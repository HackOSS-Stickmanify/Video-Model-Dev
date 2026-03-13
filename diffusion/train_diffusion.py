"""
Training Script for Pose-Conditioned Latent Diffusion

Stage 2 of the Video Model pipeline.

Features:
- DDPM training with configurable noise schedule
- Pose conditioning via concatenation or FiLM
- Mixed precision training (AMP) - supports CUDA and XPU
- EMA (Exponential Moving Average) for better inference
- Learning rate scheduling with warmup
- Validation and early stopping
- TensorBoard logging
- Checkpoint saving/resuming
- XPU (Intel GPU) support

Usage:
    # Basic training
    python train_diffusion.py \
        --latent_dir ../input_stickman_video/latents \
        --annotations ../input_stickman_video/keypoint_annotations/annotations.json \
        --vae_checkpoint ../checkpoints/vae/run_name/vae_best.pt

    # With config file
    python train_diffusion.py --config ../configs/diffusion_default.yaml

    # Resume training
    python train_diffusion.py --config ../configs/diffusion_default.yaml \
        --resume ../checkpoints/diffusion/run_name/diffusion_best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Check for XPU availability
HAS_XPU = torch.xpu.is_available()
HAS_CUDA = torch.cuda.is_available()

def get_best_device() -> torch.device:
    """Get the best available device (CUDA > XPU > CPU)."""
    if HAS_CUDA:
        return torch.device("cuda")
    elif HAS_XPU:
        return torch.device("xpu")
    else:
        return torch.device("cpu")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.unet import UNetSimple
from diffusion.scheduler import NoiseScheduler
from diffusion.dataset import DiffusionDataset, LatentCache, create_dataloaders
from vae.model import EMA


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path):
    """Save configuration to YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Create cosine learning rate schedule with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DiffusionTrainer:
    """
    Trainer for pose-conditioned latent diffusion.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: torch.device,
        output_dir: Path,
        config: dict,
    ):
        self.model = model
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        self.config = config

        # Training settings
        self.num_steps = config.get('num_steps', 50000)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.grad_accum = config.get('grad_accum', 1)
        self.use_amp = not config.get('no_amp', False)

        # EMA
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema = EMA(
                model,
                decay=config.get('ema_decay', 0.9999),
                update_after_step=config.get('ema_update_after', 100),
                update_every=config.get('ema_update_every', 10),
            )
        else:
            self.ema = None

        # Logging
        self.log_every = config.get('log_every', 100)
        self.save_every = config.get('save_every', 5000)
        self.sample_every = config.get('sample_every', 2000)
        self.val_every = config.get('val_every', 1000)

        # Early stopping
        self.early_stopping = config.get('early_stopping', True)
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 1e-4)

        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # AMP scaler - handle XPU vs CUDA
        if self.use_amp:
            if device.type == 'xpu':
                self.scaler = GradScaler(device="xpu")
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Store device type for autocast
        self.device_type = device.type


        # TensorBoard
        self.writer = SummaryWriter(output_dir / 'logs')

        # Move scheduler to device
        self.scheduler.to(device)

    def _get_autocast(self):
        """Get appropriate autocast context for the device."""
        from contextlib import nullcontext

        if not self.use_amp:
            return nullcontext()

        if self.device_type == 'xpu':
            return torch.amp.autocast('xpu', enabled=True)
        elif self.device_type == 'cuda':
            return autocast(enabled=True)
        else:
            return nullcontext()

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        latent = batch['latent'].to(self.device)
        pose = batch['pose'].to(self.device)

        # Add noise
        noisy, noise, timesteps = self.scheduler.add_noise(latent)

        # Get target
        target = self.scheduler.get_target(latent, noise, timesteps)

        # Forward pass with device-appropriate autocast
        with self._get_autocast():
            pred = self.model(noisy, timesteps, pose)
            loss = F.mse_loss(pred, target)

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss / self.grad_accum).backward()
        else:
            (loss / self.grad_accum).backward()

        return {'loss': loss.item()}

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.use_ema:
            self.ema.update()

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        if self.val_loader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            latent = batch['latent'].to(self.device)
            pose = batch['pose'].to(self.device)

            noisy, noise, timesteps = self.scheduler.add_noise(latent)
            target = self.scheduler.get_target(latent, noise, timesteps)

            with self._get_autocast():
                pred = self.model(noisy, timesteps, pose)
                loss = F.mse_loss(pred, target)

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def sample(self, num_samples: int = 4) -> torch.Tensor:
        """Generate samples using DDIM."""
        self.model.eval()

        # Get a batch of poses from validation set
        if self.val_loader is not None:
            batch = next(iter(self.val_loader))
            pose = batch['pose'][:num_samples].to(self.device)
        else:
            batch = next(iter(self.train_loader))
            pose = batch['pose'][:num_samples].to(self.device)

        # Get latent shape
        z_channels = self.config.get('z_channels', 4)
        shape = (num_samples, z_channels, 80, 60)

        # Sample with DDIM
        if self.use_ema:
            with self.ema.average_parameters():
                samples = self.scheduler.ddim_sample(
                    self.model, shape, pose, self.device,
                    num_steps=50, show_progress=False
                )
        else:
            samples = self.scheduler.ddim_sample(
                self.model, shape, pose, self.device,
                num_steps=50, show_progress=False
            )

        self.model.train()
        return samples

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': self.model.get_config(),
            'scheduler_config': self.scheduler.get_config(),
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.parent / 'diffusion_best.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed from step {self.global_step}")

    def train(self):
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()

        # Training loop
        pbar = tqdm(total=self.num_steps, initial=self.global_step, desc='Training')
        data_iter = iter(self.train_loader)
        accum_loss = 0

        while self.global_step < self.num_steps:
            # Get batch (with cycling)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Training step
            metrics = self.train_step(batch)
            accum_loss += metrics['loss']

            # Gradient accumulation
            if (self.global_step + 1) % self.grad_accum == 0:
                self.optimizer_step()

            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.log_every == 0:
                avg_loss = accum_loss / self.log_every
                lr = self.optimizer.param_groups[0]['lr']

                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
                accum_loss = 0

            # Validation
            if self.global_step % self.val_every == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)

                # Early stopping check
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(
                        self.output_dir / 'diffusion_best.pt',
                        is_best=True
                    )
                    print(f"\n[Step {self.global_step}] New best val loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                    if self.early_stopping and self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at step {self.global_step}")
                        break

            # Sampling
            if self.global_step % self.sample_every == 0:
                samples = self.sample(num_samples=4)
                # Log sample statistics
                self.writer.add_scalar('samples/mean', samples.mean().item(), self.global_step)
                self.writer.add_scalar('samples/std', samples.std().item(), self.global_step)

            # Checkpointing
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(
                    self.output_dir / f'diffusion_step_{self.global_step:06d}.pt'
                )

        pbar.close()

        # Save final checkpoint
        self.save_checkpoint(self.output_dir / 'diffusion_final.pt')

        # Save EMA model separately
        if self.use_ema:
            ema_checkpoint = {
                'model_state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                'config': self.config,
                'model_config': self.model.get_config(),
            }
            self.ema.copy_to(self.model.parameters())
            ema_checkpoint['model_state_dict'] = self.model.state_dict()
            torch.save(ema_checkpoint, self.output_dir / 'diffusion_ema.pt')

        self.writer.close()
        print(f"Training complete. Best val loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train pose-conditioned latent diffusion')

    # Data
    parser.add_argument('--latent_dir', type=str, help='Directory containing pre-encoded latents')
    parser.add_argument('--annotations', type=str, help='Path to keypoint annotations JSON')
    parser.add_argument('--image_dir', type=str, help='Image directory (for pre-encoding)')
    parser.add_argument('--vae_checkpoint', type=str, help='VAE checkpoint for pre-encoding')

    # Config
    parser.add_argument('--config', type=str, help='Path to config YAML file')

    # Model - use None defaults so config file values are respected
    parser.add_argument('--z_channels', type=int, default=None, help='Latent channels')
    parser.add_argument('--base_channels', type=int, default=None, help='UNet base channels')
    parser.add_argument('--pose_channels', type=int, default=None, help='Pose heatmap channels')

    # Scheduler
    parser.add_argument('--num_timesteps', type=int, default=None, help='Diffusion timesteps')
    parser.add_argument('--schedule', type=str, default=None,
                        choices=['linear', 'cosine', 'scaled_linear'])
    parser.add_argument('--prediction_type', type=str, default=None,
                        choices=['epsilon', 'v_prediction'])

    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--grad_accum', type=int, default=None)
    parser.add_argument('--val_split', type=float, default=None)

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=None)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=None)

    # Logging
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--sample_every', type=int, default=None)
    parser.add_argument('--val_every', type=int, default=None)

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=None)
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=None)

    # Other
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override with command line args (only if explicitly provided)
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        # No config file - use defaults
        config = vars(args)
        # Set defaults for required values
        defaults = {
            'z_channels': 4,
            'base_channels': 128,
            'pose_channels': 35,
            'num_timesteps': 1000,
            'schedule': 'cosine',
            'prediction_type': 'epsilon',
            'batch_size': 8,
            'lr': 1e-4,
            'num_steps': 50000,
            'warmup_steps': 1000,
            'grad_clip': 1.0,
            'grad_accum': 1,
            'val_split': 0.1,
            'use_ema': True,
            'ema_decay': 0.9999,
            'log_every': 100,
            'save_every': 5000,
            'sample_every': 2000,
            'val_every': 1000,
            'early_stopping': True,
            'patience': 10,
            'output_dir': '../checkpoints/diffusion',
            'num_workers': 4,
        }
        for key, default_value in defaults.items():
            if config.get(key) is None:
                config[key] = default_value

    # Handle boolean flags
    if config.get('no_ema'):
        config['use_ema'] = False
    if config.get('no_early_stopping'):
        config['early_stopping'] = False

    # Set seed
    if config.get('seed') is not None:
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])

    # Device
    if config.get('device'):
        device = torch.device(config['device'])
    else:
        device = get_best_device()
    print(f"Using device: {device}")

    # Print device info
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif device.type == 'xpu':
        print(f"XPU device: {torch.xpu.get_device_name(0)}")

    # Output directory
    run_name = config.get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config['run_name'] = run_name

    # Save config
    save_config(config, output_dir / 'config.yaml')

    # Pre-encode images if needed
    latent_dir = config.get('latent_dir')
    if latent_dir is None and config.get('image_dir') and config.get('vae_checkpoint'):
        latent_dir = Path(config['image_dir']).parent / 'latents'
        print(f"Pre-encoding images to {latent_dir}...")
        cache = LatentCache(
            config['vae_checkpoint'],
            config['image_dir'],
            latent_dir,
            device=device,
        )
        cache.encode_all(batch_size=config.get('batch_size', 8))

        # Get VAE config for z_channels
        vae_config = cache.get_latent_config()
        config['z_channels'] = vae_config['z_channels']

    if latent_dir is None:
        raise ValueError("Must provide either --latent_dir or --image_dir + --vae_checkpoint")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _, _ = create_dataloaders(
        latent_dir=latent_dir,
        annotations_file=config['annotations'],
        batch_size=config.get('batch_size', 8),
        val_split=config.get('val_split', 0.1),
        num_workers=config.get('num_workers', 4),
        include_limbs=True,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    model = UNetSimple(
        in_channels=config.get('z_channels', 4),
        base_channels=config.get('base_channels', 128),
        channel_mults=tuple(config.get('channel_mults', [1, 2, 4])),
        num_res_blocks=config.get('num_res_blocks', 2),
        attention_levels=tuple(config.get('attention_levels', [1, 2])),
        pose_channels=config.get('pose_channels', 35),
        dropout=config.get('dropout', 0.0),
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=config.get('num_timesteps', 1000),
        schedule=config.get('schedule', 'cosine'),
        prediction_type=config.get('prediction_type', 'epsilon'),
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Create scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get('warmup_steps', 1000),
        num_training_steps=config.get('num_steps', 50000),
        min_lr_ratio=config.get('min_lr', 1e-6) / config.get('lr', 1e-4),
    )

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=output_dir,
        config=config,
    )

    # Resume if specified
    if config.get('resume'):
        trainer.load_checkpoint(Path(config['resume']))

    # Train
    print("Starting training...")
    trainer.train()


if __name__ == '__main__':
    main()
