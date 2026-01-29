"""
VAE Training Script for Grayscale Images
- L1 reconstruction + tiny KL (β=0.01)
- Mixed precision (AMP)
- TensorBoard logging
- Checkpoint saving
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GrayscaleVAE, compute_vae_loss
from dataset import GrayscaleImageDataset, PreprocessedDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Grayscale VAE')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Use preprocessed (pre-resized) images')
    parser.add_argument('--target_width', type=int, default=480)
    parser.add_argument('--target_height', type=int, default=640)
    
    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='Weight for KL divergence term (β)')
    
    # Logging & saving
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='../checkpoints/vae')
    parser.add_argument('--run_name', type=str, default=None)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for speed')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    
    return parser.parse_args()


def save_checkpoint(model, optimizer, scaler, step, path):
    """Save training checkpoint."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('step', 0)


def log_reconstructions(writer, model, batch, step, device):
    """Log original and reconstructed images to TensorBoard."""
    model.eval()
    with torch.no_grad():
        # Take first 4 images
        x = batch[:4].to(device)
        recon, _, _ = model(x)
        
        # Denormalize from [-1, 1] to [0, 1]
        x_vis = (x + 1) / 2
        recon_vis = (recon + 1) / 2
        recon_vis = torch.clamp(recon_vis, 0, 1)
        
        # Concatenate original and reconstruction side by side
        comparison = torch.cat([x_vis, recon_vis], dim=3)  # Concat along width
        
        writer.add_images('reconstructions', comparison, step)
    
    model.train()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup output directory
    if args.run_name is None:
        args.run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Log hyperparameters
    writer.add_text('hyperparameters', str(vars(args)))
    
    # Create dataset
    print(f"\nLoading data from: {args.data_dir}")
    
    if args.preprocessed:
        dataset = PreprocessedDataset(args.data_dir, augment=True)
    else:
        dataset = GrayscaleImageDataset(
            args.data_dir,
            target_size=(args.target_width, args.target_height),
            augment=True,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Create model
    model = GrayscaleVAE(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        z_channels=args.z_channels,
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Compile model for speed
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled!")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Mixed precision
    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"Gradient accumulation: {args.grad_accum} steps")
    
    # Resume if specified
    start_step = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed at step {start_step}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.num_steps} steps")
    print(f"{'='*60}\n")
    
    model.train()
    step = start_step
    running_loss = {'total': 0, 'recon': 0, 'kl': 0}
    
    # Create infinite dataloader
    def infinite_loader():
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = iter(infinite_loader())
    pbar = tqdm(range(start_step, args.num_steps), desc='Training')
    
    for step in pbar:
        # Get batch
        batch = next(data_iter).to(device, non_blocking=True)
        
        # Forward pass with AMP
        if use_amp:
            with autocast('cuda'):
                recon, mean, logvar = model(batch)
                loss, loss_dict = compute_vae_loss(
                    recon, batch, mean, logvar,
                    kl_weight=args.kl_weight
                )
                loss = loss / args.grad_accum  # Scale for accumulation
            
            scaler.scale(loss).backward()
            
            # Step optimizer every grad_accum steps
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            recon, mean, logvar = model(batch)
            loss, loss_dict = compute_vae_loss(
                recon, batch, mean, logvar,
                kl_weight=args.kl_weight
            )
            loss = loss / args.grad_accum
            
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Update running loss
        for k, v in loss_dict.items():
            running_loss[k] += v
        
        # Logging
        if (step + 1) % args.log_every == 0:
            avg_loss = {k: v / args.log_every for k, v in running_loss.items()}
            
            pbar.set_postfix({
                'loss': f"{avg_loss['total']:.4f}",
                'recon': f"{avg_loss['recon']:.4f}",
                'kl': f"{avg_loss['kl']:.4f}",
            })
            
            # TensorBoard
            writer.add_scalar('loss/total', avg_loss['total'], step + 1)
            writer.add_scalar('loss/reconstruction', avg_loss['recon'], step + 1)
            writer.add_scalar('loss/kl', avg_loss['kl'], step + 1)
            
            # Reset running loss
            running_loss = {'total': 0, 'recon': 0, 'kl': 0}
        
        # Sample reconstructions
        if (step + 1) % args.sample_every == 0:
            log_reconstructions(writer, model, batch, step + 1, device)
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = output_dir / f'vae_step_{step + 1:06d}.pt'
            save_checkpoint(model, optimizer, scaler, step + 1, ckpt_path)
    
    # Final save
    final_path = output_dir / 'vae_final.pt'
    save_checkpoint(model, optimizer, scaler, args.num_steps, final_path)
    
    # Save model only (for inference)
    model_path = output_dir / 'vae_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")
    
    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
