"""
VAE Training Script for Grayscale Images

Features:
- L1 reconstruction + KL divergence + optional LPIPS perceptual loss
- Mixed precision (AMP) training
- Learning rate scheduler with warmup and cosine decay
- Validation loop with early stopping
- EMA (Exponential Moving Average) for better inference
- YAML config file support
- TensorBoard logging
- Checkpoint management with best model tracking

Usage:
    # Using command line arguments
    python train_vae.py --data_dir ../input_stickman_video/all_bw_images_480p --preprocessed

    # Using config file
    python train_vae.py --config config.yaml
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


HAS_XPU = torch.xpu.is_available()
HAS_CUDA = torch.cuda.is_available()


def get_device() -> torch.device:
    """Get the best available device (CUDA > XPU > CPU)."""
    if HAS_CUDA:
        return torch.device("cuda")
    elif HAS_XPU:
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def get_device_type(device: torch.device) -> str:
    """Get device type string for autocast."""
    if device.type == "cuda":
        return "cuda"
    elif device.type == "xpu":
        return "xpu"
    else:
        return "cpu"


def get_scaler(device: torch.device):
    """Get appropriate GradScaler for the device."""
    if device.type == "xpu":
        return GradScaler(device="xpu")
    elif device.type == "cuda":
        return GradScaler()
    return None


def get_device_name(device: torch.device) -> str:
    """Get human-readable device name."""
    if device.type == "cuda":
        return torch.cuda.get_device_name()
    elif device.type == "xpu":
        return torch.xpu.get_device_name() if hasattr(torch.xpu, 'get_device_name') else "Intel XPU"
    return "CPU"


def get_device_memory(device: torch.device) -> float:
    """Get device memory in GB."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'get_device_properties'):
            return torch.xpu.get_device_properties(0).total_memory / 1e9
        return 0.0
    return 0.0

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

from dataset import ImageDataset, PreprocessedDataset
from model import EMA, VAELoss, VAE


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""

    # Data
    data_dir: str = "../input_stickman_video/all_bw_images_480p"
    preprocessed: bool = True
    target_width: int = 480
    target_height: int = 640
    val_split: float = 0.1  # Fraction of data for validation
    color: bool = False  # If True, train on RGB images (3 channels); else grayscale (1 channel)

    # Model
    base_channels: int = 32
    channel_mults: tuple[int, ...] = (1, 2, 4)
    z_channels: int = 4
    num_res_blocks: int = 2

    # Loss
    kl_weight: float = 0.01
    lpips_weight: float = 0.1
    use_lpips: bool = False  # Enable for better perceptual quality

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    min_lr: float = 1e-6
    num_steps: int = 50000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accum: int = 1

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 100
    ema_update_every: int = 10

    # Logging & saving
    log_every: int = 100
    save_every: int = 5000
    sample_every: int = 1000
    val_every: int = 1000
    output_dir: str = "../checkpoints/vae"
    run_name: Union[str, None] = None

    # Early stopping
    early_stopping: bool = True
    patience: int = 10  # Number of validations without improvement
    min_delta: float = 1e-4  # Minimum improvement to reset patience

    # Hardware
    num_workers: int = 4
    no_amp: bool = False
    compile: bool = False
    seed: Union[int, None] = None
    device: Union[str, None] = None  # Auto-detect if None, or "cuda", "xpu", "cpu"

    # Resume
    resume: Union[str, None] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v if not isinstance(v, tuple) else list(v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        """Create config from dictionary."""
        if "channel_mults" in d and isinstance(d["channel_mults"], list):
            d["channel_mults"] = tuple(d["channel_mults"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        """Load config from YAML file."""
        if not YAML_AVAILABLE or yaml is None:
            raise ImportError("PyYAML is required for config file support: pip install pyyaml")
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        if not YAML_AVAILABLE or yaml is None:
            raise ImportError("PyYAML is required for config file support: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Union[GradScaler, None],
    ema: Union[EMA, None],
    step: int,
    val_loss: float,
    config: TrainConfig,
    path: Path,
    is_best: bool = False,
):
    """Save training checkpoint with all state."""
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "val_loss": val_loss,
        "train_config": config.to_dict(),
    }
    torch.save(checkpoint, path)
    print(f"{'✓ ' if is_best else ''}Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
    scaler: Union[GradScaler, None] = None,
    ema: Union[EMA, None] = None,
) -> tuple[int, float]:
    """Load training checkpoint and return step and val_loss."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if ema and checkpoint.get("ema_state_dict"):
        ema.load_state_dict(checkpoint["ema_state_dict"])

    return checkpoint.get("step", 0), checkpoint.get("val_loss", float("inf"))


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: VAELoss,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run validation and return average losses."""
    model.eval()
    total_losses = {}
    num_batches = 0

    device_type = get_device_type(device)
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            recon, mean, logvar = model(batch)
            _, loss_dict = loss_fn(recon, batch, mean, logvar)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

    # Average
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    model.train()
    return avg_losses


@torch.no_grad()
def log_reconstructions(
    writer: SummaryWriter,
    model: nn.Module,
    batch: torch.Tensor,
    step: int,
    device: torch.device,
    prefix: str = "",
    ema: Union[EMA, None] = None,
):
    """Log original and reconstructed images to TensorBoard."""
    model.eval()

    # Take first 4 images
    x = batch[:4].to(device)

    # Regular reconstruction
    recon, _, _ = model(x)

    # Denormalize from [-1, 1] to [0, 1]
    x_vis = (x + 1) / 2
    recon_vis = torch.clamp((recon + 1) / 2, 0, 1)

    # Concatenate original and reconstruction side by side
    comparison = torch.cat([x_vis, recon_vis], dim=3)
    writer.add_images(f"{prefix}reconstructions", comparison, step)

    # EMA reconstruction
    if ema is not None:
        with ema.average_parameters():
            recon_ema, _, _ = model(x)
        recon_ema_vis = torch.clamp((recon_ema + 1) / 2, 0, 1)
        comparison_ema = torch.cat([x_vis, recon_ema_vis], dim=3)
        writer.add_images(f"{prefix}reconstructions_ema", comparison_ema, step)

    model.train()


def parse_args() -> TrainConfig:
    """Parse command line arguments into TrainConfig."""
    parser = argparse.ArgumentParser(description="Train Grayscale VAE")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data
    parser.add_argument("--data_dir", type=str, help="Path to image directory")
    parser.add_argument("--preprocessed", action="store_true", help="Use preprocessed images")
    parser.add_argument("--target_width", type=int)
    parser.add_argument("--target_height", type=int)
    parser.add_argument("--color", action="store_true", default=None, help="Train on RGB color images (3 channels)")
    parser.add_argument("--grayscale", action="store_true", default=None, help="Train on grayscale images (1 channel)")
    parser.add_argument("--val_split", type=float, help="Validation split fraction")

    # Model
    parser.add_argument("--base_channels", type=int)
    parser.add_argument("--z_channels", type=int)
    parser.add_argument("--num_res_blocks", type=int)

    # Loss
    parser.add_argument("--kl_weight", type=float, help="Weight for KL divergence (β)")
    parser.add_argument("--lpips_weight", type=float, help="Weight for LPIPS loss")
    parser.add_argument("--use_lpips", action="store_true", help="Enable LPIPS perceptual loss")

    # Training
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--grad_accum", type=int, help="Gradient accumulation steps")

    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--ema_decay", type=float)

    # Logging & saving
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--sample_every", type=int)
    parser.add_argument("--val_every", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--run_name", type=str)

    # Early stopping
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int)

    # Hardware
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, choices=["cuda", "xpu", "cpu"],
                        help="Device to use (auto-detect if not specified)")

    # Resume
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Start with defaults or config file
    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    # Handle color/grayscale flags explicitly (store_true defaults to False, not None)
    if args.color:
        config.color = True
    elif args.grayscale:
        config.color = False
    # Otherwise keep config.color from YAML or default

    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ("config", "color", "grayscale"):
            if key == "no_ema":
                config.use_ema = not value
            elif key == "no_early_stopping":
                config.early_stopping = not value
            elif hasattr(config, key):
                setattr(config, key, value)

    return config


def main():
    config = parse_args()

    # Set seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Setup device
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = get_device()
    device_type = get_device_type(device)
    print(f"Using device: {device}")

    if device.type in ("cuda", "xpu"):
        print(f"GPU: {get_device_name(device)}")
        mem = get_device_memory(device)
        if mem > 0:
            print(f"Memory: {mem:.1f} GB")

    # Setup output directory
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Save config
    config_path = output_dir / "config.yaml"
    if YAML_AVAILABLE:
        config.save_yaml(str(config_path))
        print(f"Saved config to {config_path}")

    # Setup TensorBoard
    writer = SummaryWriter(log_dir)
    writer.add_text("config", str(config.to_dict()))

    # Create dataset
    print(f"\nLoading data from: {config.data_dir}")

    # Determine color mode
    color_mode = "RGB" if config.color else "L"
    in_channels = 3 if config.color else 1
    print(f"Color mode: {color_mode} ({in_channels} channel{'s' if in_channels > 1 else ''})")

    if config.preprocessed:
        full_dataset = PreprocessedDataset(config.data_dir, augment=True, color_mode=color_mode)
    else:
        full_dataset = ImageDataset(
            config.data_dir,
            target_size=(config.target_width, config.target_height),
            augment=True,
            color_mode=color_mode,
        )

    # Train/Val split
    val_size = max(int(len(full_dataset) * config.val_split), 1)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Disable augmentation for validation (create a fresh dataset)
    if config.preprocessed:
        val_dataset_clean = PreprocessedDataset(config.data_dir, augment=False, color_mode=color_mode)
    else:
        val_dataset_clean = ImageDataset(
            config.data_dir,
            target_size=(config.target_width, config.target_height),
            augment=False,
            color_mode=color_mode,
        )

    # Use same indices as val split
    val_indices = val_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(val_dataset_clean, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    print(f"Dataset size: {len(full_dataset)} (train: {train_size}, val: {val_size})")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")

    # Create model
    model = VAE(
        in_channels=in_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        z_channels=config.z_channels,
        num_res_blocks=config.num_res_blocks,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Compile model for speed
    if config.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore
        print("Model compiled!")

    # Loss function
    loss_fn = VAELoss(
        kl_weight=config.kl_weight,
        lpips_weight=config.lpips_weight,
        use_lpips=config.use_lpips,
    ).to(device)

    print(f"Loss: L1 + {config.kl_weight}×KL" + (f" + {config.lpips_weight}×LPIPS" if config.use_lpips else ""))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    # Learning rate scheduler
    min_lr_ratio = config.min_lr / config.lr
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.num_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # Mixed precision
    use_amp = not config.no_amp and device.type in ("cuda", "xpu")
    scaler = get_scaler(device) if use_amp else None
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    # EMA
    ema = None
    if config.use_ema:
        ema = EMA(
            model,
            decay=config.ema_decay,
            update_after_step=config.ema_update_after,
            update_every=config.ema_update_every,
        )
        print(f"EMA: enabled (decay={config.ema_decay})")

    print(f"Gradient accumulation: {config.grad_accum} steps")
    print(f"Gradient clipping: {config.grad_clip}")

    # Resume if specified
    start_step = 0
    best_val_loss = float("inf")
    if config.resume:
        print(f"\nResuming from: {config.resume}")
        start_step, best_val_loss = load_checkpoint(
            config.resume, model, optimizer, scheduler, scaler, ema
        )
        print(f"Resumed at step {start_step}, best val loss: {best_val_loss:.4f}")

    # Early stopping state
    patience_counter = 0

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.num_steps} steps")
    print(f"Warmup: {config.warmup_steps} steps")
    if config.early_stopping:
        print(f"Early stopping: patience={config.patience}")
    print(f"{'='*60}\n")

    model.train()
    running_loss = {}
    optimizer.zero_grad()

    # Create infinite dataloader
    def infinite_loader():
        while True:
            yield from train_loader

    data_iter = iter(infinite_loader())
    pbar = tqdm(range(start_step, config.num_steps), desc="Training", initial=start_step, total=config.num_steps)

    for step in pbar:
        # Get batch
        batch = next(data_iter).to(device, non_blocking=True)

        # Forward pass with AMP
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            recon, mean, logvar = model(batch)
            loss, loss_dict = loss_fn(recon, batch, mean, logvar)
            loss = loss / config.grad_accum  # Scale for accumulation

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum steps
        if (step + 1) % config.grad_accum == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            # Gradient clipping
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update()

        # Update running loss
        for k, v in loss_dict.items():
            running_loss[k] = running_loss.get(k, 0) + v

        # Logging
        if (step + 1) % config.log_every == 0:
            avg_loss = {k: v / config.log_every for k, v in running_loss.items()}

            pbar_dict = {
                "loss": f"{avg_loss['total']:.4f}",
                "recon": f"{avg_loss['recon']:.4f}",
                "kl": f"{avg_loss['kl']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if "lpips" in avg_loss:
                pbar_dict["lpips"] = f"{avg_loss['lpips']:.4f}"

            pbar.set_postfix(pbar_dict)

            # TensorBoard
            for k, v in avg_loss.items():
                writer.add_scalar(f"train/{k}", v, step + 1)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step + 1)

            # Reset running loss
            running_loss = {}

        # Sample reconstructions
        if (step + 1) % config.sample_every == 0:
            log_reconstructions(writer, model, batch, step + 1, device, prefix="train/", ema=ema)

        # Validation
        if (step + 1) % config.val_every == 0:
            val_losses = validate(model, val_loader, loss_fn, device, use_amp)

            # Log validation
            for k, v in val_losses.items():
                writer.add_scalar(f"val/{k}", v, step + 1)

            print(f"\nStep {step + 1} - Val loss: {val_losses['total']:.4f} "
                  f"(recon: {val_losses['recon']:.4f}, kl: {val_losses['kl']:.4f})")

            # Check for best model
            if val_losses["total"] < best_val_loss - config.min_delta:
                improvement = best_val_loss - val_losses["total"]
                best_val_loss = val_losses["total"]
                patience_counter = 0

                # Save best model
                best_path = output_dir / "vae_best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema,
                    step + 1, best_val_loss, config, best_path, is_best=True
                )
                print(f"New best! (improved by {improvement:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{config.patience})")

                # Early stopping
                if config.early_stopping and patience_counter >= config.patience:
                    print(f"\nEarly stopping triggered at step {step + 1}")
                    break

        # Save checkpoint
        if (step + 1) % config.save_every == 0:
            ckpt_path = output_dir / f"vae_step_{step + 1:06d}.pt"
            save_checkpoint(
                model, optimizer, scheduler, scaler, ema,
                step + 1, best_val_loss, config, ckpt_path
            )

    # Final save
    final_path = output_dir / "vae_final.pt"
    save_checkpoint(
        model, optimizer, scheduler, scaler, ema,
        config.num_steps, best_val_loss, config, final_path
    )

    # Save model weights only (for inference)
    model_path = output_dir / "vae_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
    }, model_path)
    print(f"\nSaved final model to {model_path}")

    # Save EMA weights
    if ema is not None:
        ema_model_path = output_dir / "vae_model_ema.pt"
        with ema.average_parameters():
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
            }, ema_model_path)
        print(f"Saved EMA model to {ema_model_path}")

    writer.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
