"""
VAE Model Testing Script

Generate side-by-side comparison images (original vs reconstruction)
to evaluate model quality.

Supports both grayscale and color (RGB) images.
Color mode is auto-detected from the model checkpoint.

Usage:
    # Auto-detect color mode from checkpoint (recommended)
    python evaluate.py reconstruct -c ../checkpoints/vae/run_name/vae_best.pt

    # Force grayscale mode
    python evaluate.py reconstruct -c ../checkpoints/vae/run_name/vae_best.pt --grayscale

    # Force color mode
    python evaluate.py reconstruct -c ../checkpoints/vae/run_name/vae_best.pt --color

    # Interpolation
    python evaluate.py interpolate img1.jpg img2.jpg -c checkpoint.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Union

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Handle imports for both module and script usage
try:
    from .inference import VAEInference
except ImportError:
    from inference import VAEInference


def create_comparison_image(
    original: Image.Image,
    reconstruction: Image.Image,
    label: str = "",
    padding: int = 10,
    label_height: int = 30,
) -> Image.Image:
    """
    Create a side-by-side comparison image.

    Args:
        original: Original input image
        reconstruction: Reconstructed image from VAE
        label: Optional label text to add at the top
        padding: Padding between images
        label_height: Height reserved for label

    Returns:
        Combined image with original on left, reconstruction on right
    """
    w, h = original.size

    # Calculate dimensions
    total_width = w * 2 + padding * 3
    total_height = h + padding * 2 + (label_height if label else 0)

    # Create canvas (white background)
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # Convert grayscale to RGB if needed
    if original.mode == "L":
        original = original.convert("RGB")
    if reconstruction.mode == "L":
        reconstruction = reconstruction.convert("RGB")

    # Paste images
    y_offset = label_height if label else 0
    canvas.paste(original, (padding, padding + y_offset))
    canvas.paste(reconstruction, (w + padding * 2, padding + y_offset))

    # Add labels
    draw = ImageDraw.Draw(canvas)

    # Try to use a nicer font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font

    # Add "Original" and "Reconstruction" labels below images
    draw.text((padding + w // 2 - 30, padding + y_offset + h + 2), "Original", fill=(100, 100, 100), font=small_font)
    draw.text((w + padding * 2 + w // 2 - 50, padding + y_offset + h + 2), "Reconstruction", fill=(100, 100, 100), font=small_font)

    # Add main label if provided
    if label:
        draw.text((padding, 5), label, fill=(50, 50, 50), font=font)

    return canvas


def compute_metrics(original: torch.Tensor, reconstruction: torch.Tensor) -> dict:
    """
    Compute reconstruction quality metrics.

    Args:
        original: Original image tensor (1, C, H, W) in [-1, 1]
        reconstruction: Reconstructed tensor (1, C, H, W) in [-1, 1]

    Returns:
        Dictionary with metrics (MSE, PSNR, L1)
    """
    # Ensure same device
    reconstruction = reconstruction.to(original.device)

    # Convert to [0, 1] range
    orig = (original + 1) / 2
    recon = (reconstruction + 1) / 2
    recon = torch.clamp(recon, 0, 1)

    # MSE
    mse = torch.mean((orig - recon) ** 2).item()

    # PSNR
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # L1
    l1 = torch.mean(torch.abs(orig - recon)).item()

    return {
        "mse": mse,
        "psnr": psnr,
        "l1": l1,
    }


# Default image directories
DEFAULT_GRAYSCALE_DIR = "../input_stickman_video/all_bw_images_480p"
DEFAULT_COLOR_DIR = "../input_stickman_video/all_colored_images"


def test_model(
    checkpoint_path: str,
    image_dir: str = None,
    output_dir: str = "test_output",
    num_samples: int = 5,
    seed: int = 42,
    create_grid: bool = True,
    color: bool = None,
):
    """
    Test VAE model on sample images.

    Args:
        checkpoint_path: Path to model checkpoint
        image_dir: Directory containing test images (auto-selected based on color mode if None)
        output_dir: Directory to save comparison images
        num_samples: Number of random samples to test
        seed: Random seed for reproducibility
        create_grid: Whether to create a grid of all comparisons
        color: If True, force RGB; if False, force grayscale; if None, auto-detect from model
    """
    random.seed(seed)

    # Load model (color mode auto-detected if not specified)
    color_mode = "RGB" if color is True else ("L" if color is False else None)
    print(f"Loading model from: {checkpoint_path}")
    vae = VAEInference.from_checkpoint(checkpoint_path, color_mode=color_mode)

    # Auto-select image directory based on model's color mode if not specified
    if image_dir is None:
        script_dir = Path(__file__).parent
        if vae.color_mode == "RGB":
            image_dir = script_dir / DEFAULT_COLOR_DIR
            print(f"Auto-selected color image directory: {image_dir}")
        else:
            image_dir = script_dir / DEFAULT_GRAYSCALE_DIR
            print(f"Auto-selected grayscale image directory: {image_dir}")
    
    # Setup paths
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Select random samples
    num_samples = min(num_samples, len(image_files))
    selected = random.sample(image_files, num_samples)

    print(f"\nTesting on {num_samples} random samples...")
    print("=" * 60)

    comparisons = []
    all_metrics = []

    for i, img_path in enumerate(selected):
        print(f"\n[{i+1}/{num_samples}] {img_path.name}")

        # Get reconstruction
        x = vae.preprocess(img_path)
        recon_tensor, mean, logvar = vae.model(x)
        recon_pil = vae.postprocess(recon_tensor)

        # Compute metrics
        metrics = compute_metrics(x, recon_tensor)
        all_metrics.append(metrics)

        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  L1:   {metrics['l1']:.6f}")

        # Get latent stats
        print(f"  Latent mean: {mean.mean().item():.4f} (std: {mean.std().item():.4f})")
        print(f"  Latent var:  {torch.exp(logvar).mean().item():.4f}")

        # Create comparison image
        # Use preprocessed original for fair comparison (same size/padding)
        original_pil_resized = vae.preprocess_for_display(img_path)

        label = f"{img_path.name} | PSNR: {metrics['psnr']:.1f}dB | L1: {metrics['l1']:.4f}"
        comparison = create_comparison_image(original_pil_resized, recon_pil, label=label)
        comparisons.append(comparison)

        # Save individual comparison
        comparison.save(output_dir / f"comparison_{i:02d}_{img_path.stem}.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_mse = np.mean([m["mse"] for m in all_metrics])
    avg_psnr = np.mean([m["psnr"] for m in all_metrics])
    avg_l1 = np.mean([m["l1"] for m in all_metrics])

    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average L1:   {avg_l1:.6f}")

    # Create grid if requested
    if create_grid and len(comparisons) > 1:
        print(f"\nCreating comparison grid...")

        # Calculate grid dimensions
        n = len(comparisons)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        # Get single comparison size
        comp_w, comp_h = comparisons[0].size

        # Create grid canvas
        grid_w = cols * comp_w + (cols + 1) * 5
        grid_h = rows * comp_h + (rows + 1) * 5
        grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))

        for idx, comp in enumerate(comparisons):
            row = idx // cols
            col = idx % cols
            x = col * comp_w + (col + 1) * 5
            y = row * comp_h + (row + 1) * 5
            grid.paste(comp, (x, y))

        grid_path = output_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"Grid saved to: {grid_path}")

    print(f"\nAll comparisons saved to: {output_dir}")

    return {
        "avg_mse": avg_mse,
        "avg_psnr": avg_psnr,
        "avg_l1": avg_l1,
        "num_samples": num_samples,
    }


def test_interpolation(
    checkpoint_path: str,
    image1_path: str,
    image2_path: str,
    output_dir: str = "test_interpolation",
    steps: int = 10,
    color: bool = None,
):
    """
    Test latent space interpolation between two images.
    """
    # Load model (color mode auto-detected if not specified)
    color_mode = "RGB" if color is True else ("L" if color is False else None)
    print(f"Loading model from: {checkpoint_path}")
    vae = VAEInference.from_checkpoint(checkpoint_path, color_mode=color_mode)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate interpolation
    print(f"Interpolating between:")
    print(f"  {image1_path}")
    print(f"  {image2_path}")
    print(f"  Steps: {steps}")

    frames = vae.interpolate(image1_path, image2_path, steps=steps)

    # Save frames
    for i, frame in enumerate(frames):
        frame.save(output_dir / f"interp_{i:03d}.png")

    # Create strip
    frame_w, frame_h = frames[0].size
    strip_w = frame_w * len(frames) + (len(frames) + 1) * 2
    strip = Image.new("RGB", (strip_w, frame_h + 4), (255, 255, 255))

    for i, frame in enumerate(frames):
        if frame.mode == "L":
            frame = frame.convert("RGB")
        x = i * frame_w + (i + 1) * 2
        strip.paste(frame, (x, 2))

    strip_path = output_dir / "interpolation_strip.png"
    strip.save(strip_path)

    print(f"\nInterpolation frames saved to: {output_dir}")
    print(f"Strip saved to: {strip_path}")


def main():
    parser = argparse.ArgumentParser(description="Test VAE Model")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Reconstruct command
    recon_parser = subparsers.add_parser("reconstruct", help="Test reconstruction quality")
    recon_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    recon_parser.add_argument("--image_dir", "-i",
                              default=None,
                              help="Directory with test images (auto-selected based on model's color mode if not specified)")
    recon_parser.add_argument("--output_dir", "-o", default="../test_output",
                              help="Output directory for comparisons")
    recon_parser.add_argument("--num_samples", "-n", type=int, default=5,
                              help="Number of random samples to test")
    recon_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    recon_parser.add_argument("--no_grid", action="store_true", help="Don't create grid")
    color_group = recon_parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", action="store_true",
                              help="Force color (RGB) mode")
    color_group.add_argument("--grayscale", action="store_true",
                              help="Force grayscale mode")

    # Interpolate command
    interp_parser = subparsers.add_parser("interpolate", help="Test latent interpolation")
    interp_parser.add_argument("image1", help="First image")
    interp_parser.add_argument("image2", help="Second image")
    interp_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    interp_parser.add_argument("--output_dir", "-o", default="../test_interpolation",
                              help="Output directory")
    interp_parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    interp_color_group = interp_parser.add_mutually_exclusive_group()
    interp_color_group.add_argument("--color", action="store_true",
                              help="Force color (RGB) mode")
    interp_color_group.add_argument("--grayscale", action="store_true",
                              help="Force grayscale mode")

    # Quick test (default)
    quick_parser = subparsers.add_parser("quick", help="Quick test with defaults")
    quick_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    quick_color_group = quick_parser.add_mutually_exclusive_group()
    quick_color_group.add_argument("--color", action="store_true",
                              help="Force color (RGB) mode")
    quick_color_group.add_argument("--grayscale", action="store_true",
                              help="Force grayscale mode")

    args = parser.parse_args()

    # Handle no command (default to quick if checkpoint provided via main args)
    if args.command is None:
        parser.print_help()
        print("\n\nExamples:")
        print("  python evaluate.py reconstruct -c ../checkpoints/vae/run/vae_best.pt  # auto-detect color")
        print("  python evaluate.py reconstruct -c ../checkpoints/vae/run/vae_best.pt --color  # force RGB")
        print("  python evaluate.py reconstruct -c ../checkpoints/vae/run/vae_best.pt --grayscale  # force L")
        print("  python evaluate.py quick -c ../checkpoints/vae/run/vae_model_ema.pt")
        print("  python evaluate.py interpolate img1.jpg img2.jpg -c checkpoint.pt")
        return

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent

    if args.command == "reconstruct":
        image_dir = Path(args.image_dir)
        if not image_dir.is_absolute():
            image_dir = script_dir / image_dir

        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir

        test_model(
            checkpoint_path=args.checkpoint,
            image_dir=str(image_dir),
            output_dir=str(output_dir),
            num_samples=args.num_samples,
            seed=args.seed,
            create_grid=not args.no_grid,
            color=True if args.color else (False if args.grayscale else None),
        )

    elif args.command == "interpolate":
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir

        test_interpolation(
            checkpoint_path=args.checkpoint,
            image1_path=args.image1,
            image2_path=args.image2,
            output_dir=str(output_dir),
            steps=args.steps,
            color=True if args.color else (False if args.grayscale else None),
        )

    elif args.command == "quick":
        output_dir = script_dir / "../test_output"

        test_model(
            checkpoint_path=args.checkpoint,
            image_dir=None,  # Auto-select based on model's color mode
            output_dir=str(output_dir),
            num_samples=5,
            seed=42,
            create_grid=True,
            color=True if args.color else (False if args.grayscale else None),
        )


if __name__ == "__main__":
    main()
