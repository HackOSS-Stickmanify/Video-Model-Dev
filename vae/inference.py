"""
VAE Inference Utilities

Provides easy-to-use functions for:
- Loading trained VAE models
- Encoding images to latent space
- Decoding latents to images
- Batch processing directories
- Visualizing reconstructions

Supports CUDA, Intel XPU, and CPU devices.
Supports both grayscale and color (RGB) images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


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


def get_device_name(device: torch.device) -> str:
    """Get human-readable device name."""
    if device.type == "cuda":
        return torch.cuda.get_device_name()
    elif device.type == "xpu":
        return torch.xpu.get_device_name() if hasattr(torch.xpu, 'get_device_name') else "Intel XPU"
    return "CPU"

try:
    from .model import VAE
except ImportError:
    from model import VAE


class VAEInference:
    """
    Inference wrapper for VAE.

    Example:
        vae = VAEInference.from_checkpoint("checkpoints/vae_best.pt")

        # Encode/decode single image
        latent = vae.encode("image.jpg")
        recon = vae.decode(latent)

        # Reconstruct image
        recon_pil = vae.reconstruct("image.jpg")
        recon_pil.save("reconstruction.jpg")

        # Batch process directory
        vae.process_directory("input/", "output/")
    """

    def __init__(
        self,
        model: VAE,
        device: torch.device,
        target_size: tuple[int, int] = (480, 640),  # (width, height)
        color_mode: str = "L",  # "L" for grayscale, "RGB" for color
    ):
        self.model = model
        self.device = device
        self.target_size = target_size
        self.color_mode = color_mode.upper()
        self.model.eval()

        if self.color_mode not in ("L", "RGB"):
            raise ValueError(f"color_mode must be 'L' or 'RGB', got {color_mode}")

        # Number of channels based on color mode
        self.num_channels = 3 if self.color_mode == "RGB" else 1

        # Standard transforms
        self.to_tensor = transforms.ToTensor()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
        color_mode: Optional[str] = None,
    ) -> "VAEInference":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model on (auto-detected if None)
            use_ema: If True and EMA weights exist, use them
            color_mode: "L" for grayscale, "RGB" for color. If None, auto-detect from model config.
        """
        if device is None:
            device = get_best_device()

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get model config
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            model = VAE.from_config(config)
        else:
            # Legacy checkpoint without config
            model = VAE()
            config = {}

        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)

        model = model.to(device)

        # Get model's actual input channels
        model_in_channels = config.get("in_channels", 1)
        model_color_mode = "RGB" if model_in_channels == 3 else "L"
        
        # Auto-detect color mode from model config if not specified
        if color_mode is None:
            color_mode = model_color_mode
            print(f"  Auto-detected color mode: {color_mode} (in_channels={model_in_channels})")
        else:
            # Validate that requested color mode matches model
            requested_channels = 3 if color_mode == "RGB" else 1
            if requested_channels != model_in_channels:
                print(f"\n  WARNING: Requested color mode '{color_mode}' ({requested_channels} channels) "
                      f"does not match model's in_channels={model_in_channels}")
                print(f"  Using model's native color mode: {model_color_mode}")
                color_mode = model_color_mode
        
        print(f"Loaded VAE from {checkpoint_path}")
        print(f"  Device: {device}" + (f" ({get_device_name(device)})" if device.type != "cpu" else ""))
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Color mode: {color_mode} (model in_channels={model_in_channels})")

        return cls(model, device, color_mode=color_mode)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: Optional[torch.device] = None,
        prefer_ema: bool = True,
        color_mode: Optional[str] = None,
    ) -> "VAEInference":
        """
        Load best model from training directory.

        Looks for models in order: vae_model_ema.pt, vae_best.pt, vae_final.pt
        Color mode is auto-detected from model config if not specified.
        """
        model_dir = Path(model_dir)

        # Find model file
        candidates = []
        if prefer_ema:
            candidates.append(model_dir / "vae_model_ema.pt")
        candidates.extend([
            model_dir / "vae_best.pt",
            model_dir / "vae_model.pt",
            model_dir / "vae_final.pt",
        ])

        for path in candidates:
            if path.exists():
                return cls.from_checkpoint(str(path), device, color_mode=color_mode)

        raise FileNotFoundError(f"No model found in {model_dir}")

    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Path to image or PIL Image

        Returns:
            Tensor of shape (1, C, H, W) in [-1, 1] where C=1 (grayscale) or C=3 (RGB)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to appropriate color mode
        image = image.convert(self.color_mode)

        # Resize to target size (width, height)
        target_w, target_h = self.target_size
        orig_w, orig_h = image.size

        # Scale to match target height, then pad width
        scale = target_h / orig_h
        new_w = int(orig_w * scale)
        new_h = target_h

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Pad width if needed
        if new_w < target_w:
            pad_left = (target_w - new_w) // 2
            # Use white padding (255 for grayscale, (255,255,255) for RGB)
            pad_color = 255 if self.color_mode == "L" else (255, 255, 255)
            padded = Image.new(self.color_mode, (target_w, target_h), pad_color)
            padded.paste(image, (pad_left, 0))
            image = padded
        elif new_w > target_w:
            left = (new_w - target_w) // 2
            image = image.crop((left, 0, left + target_w, target_h))

        # To tensor and normalize to [-1, 1]
        tensor = self.to_tensor(image)  # [C, H, W] in [0, 1]
        tensor = tensor * 2 - 1  # [-1, 1]

        return tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert model output to PIL Image.

        Args:
            tensor: Tensor of shape (1, C, H, W) in [-1, 1] where C=1 or C=3

        Returns:
            PIL Image (grayscale or RGB depending on color_mode)
        """
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)

        # To numpy and convert to PIL
        array = tensor.squeeze(0).cpu().detach().numpy()  # [C, H, W]

        if self.color_mode == "RGB":
            # [C, H, W] -> [H, W, C]
            array = array.transpose(1, 2, 0)
            array = (array * 255).astype("uint8")
            return Image.fromarray(array, mode="RGB")
        else:
            # Grayscale: [1, H, W] -> [H, W]
            array = array.squeeze()
            array = (array * 255).astype("uint8")
            return Image.fromarray(array, mode="L")

    def preprocess_for_display(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Preprocess image and return as PIL Image (for display/comparison).

        This applies the same preprocessing as preprocess() but returns a PIL Image
        instead of a tensor. Useful for creating comparison images.

        Args:
            image: Path to image or PIL Image

        Returns:
            Preprocessed PIL Image
        """
        tensor = self.preprocess(image)
        return self.postprocess(tensor)

    @torch.no_grad()
    def encode(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: Image path, PIL Image, or preprocessed tensor
            deterministic: If True, return mean; else sample from distribution

        Returns:
            Latent tensor of shape (1, z_channels, H/8, W/8)
        """
        if isinstance(image, torch.Tensor):
            x = image.to(self.device)
        else:
            x = self.preprocess(image)

        return self.model.get_latent(x, deterministic=deterministic)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image tensor.

        Args:
            latent: Latent tensor of shape (B, z_channels, H, W)

        Returns:
            Image tensor of shape (B, 1, H*8, W*8) in [-1, 1]
        """
        return self.model.decode(latent.to(self.device))

    @torch.no_grad()
    def reconstruct(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        return_tensor: bool = False,
        deterministic: bool = True,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Encode and decode an image.

        Args:
            image: Input image
            return_tensor: If True, return tensor instead of PIL Image
            deterministic: If True, use mean latent

        Returns:
            Reconstructed image as PIL Image or tensor
        """
        if isinstance(image, torch.Tensor):
            x = image.to(self.device)
        else:
            x = self.preprocess(image)

        # Full forward pass
        recon, mean, logvar = self.model(x)

        if return_tensor:
            return recon
        return self.postprocess(recon)

    @torch.no_grad()
    def sample(self, batch_size: int = 1) -> Union[Image.Image, list[Image.Image]]:
        """
        Sample random images from the prior.

        Args:
            batch_size: Number of images to generate

        Returns:
            PIL Image if batch_size=1, else list of PIL Images
        """
        samples = self.model.sample(batch_size=batch_size, device=self.device)

        if batch_size == 1:
            return self.postprocess(samples)

        return [self.postprocess(samples[i:i+1]) for i in range(batch_size)]

    @torch.no_grad()
    def interpolate(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
        steps: int = 5,
    ) -> list[Image.Image]:
        """
        Interpolate between two images in latent space.

        Args:
            image1: First image
            image2: Second image
            steps: Number of interpolation steps (including endpoints)

        Returns:
            List of PIL Images
        """
        z1 = self.encode(image1, deterministic=True)
        z2 = self.encode(image2, deterministic=True)

        images = []
        for i in range(steps):
            t = i / (steps - 1)
            z = z1 * (1 - t) + z2 * t
            recon = self.decode(z)
            images.append(self.postprocess(recon))

        return images

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        save_latents: bool = False,
    ):
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory with images
            output_dir: Output directory for reconstructions
            extensions: File extensions to process
            save_latents: If True, also save latent tensors
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_latents:
            latents_path = output_path / "latents"
            latents_path.mkdir(exist_ok=True)

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        print(f"Processing {len(image_files)} images...")

        for img_path in tqdm(image_files):
            # Reconstruct
            recon = self.reconstruct(img_path)

            # Save reconstruction
            out_name = img_path.stem + "_recon.png"
            recon.save(output_path / out_name)

            # Save latent if requested
            if save_latents:
                latent = self.encode(img_path)
                latent_name = img_path.stem + "_latent.pt"
                torch.save(latent.cpu(), latents_path / latent_name)

        print(f"Done! Saved reconstructions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VAE Inference Tool")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Reconstruct command
    recon_parser = subparsers.add_parser("reconstruct", help="Reconstruct image(s)")
    recon_parser.add_argument("input", help="Input image or directory")
    recon_parser.add_argument("--output", "-o", default="output", help="Output path")
    recon_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    recon_parser.add_argument("--save-latents", action="store_true", help="Save latent tensors")

    # Interpolate command
    interp_parser = subparsers.add_parser("interpolate", help="Interpolate between two images")
    interp_parser.add_argument("image1", help="First image")
    interp_parser.add_argument("image2", help="Second image")
    interp_parser.add_argument("--output", "-o", default="interpolation", help="Output directory")
    interp_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    interp_parser.add_argument("--steps", type=int, default=10, help="Number of steps")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample random images")
    sample_parser.add_argument("--output", "-o", default="samples", help="Output directory")
    sample_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    sample_parser.add_argument("--num", "-n", type=int, default=10, help="Number of samples")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode images to latents")
    encode_parser.add_argument("input", help="Input image or directory")
    encode_parser.add_argument("--output", "-o", default="latents", help="Output directory")
    encode_parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Load model
    vae = VAEInference.from_checkpoint(args.checkpoint)

    if args.command == "reconstruct":
        input_path = Path(args.input)

        if input_path.is_dir():
            vae.process_directory(args.input, args.output, save_latents=args.save_latents)
        else:
            recon = vae.reconstruct(args.input)
            output_path = Path(args.output)

            if output_path.suffix:
                recon.save(output_path)
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                recon.save(output_path / f"{input_path.stem}_recon.png")

            print(f"Saved reconstruction to {output_path}")

    elif args.command == "interpolate":
        images = vae.interpolate(args.image1, args.image2, steps=args.steps)

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            img.save(output_path / f"interpolation_{i:03d}.png")

        print(f"Saved {len(images)} interpolation frames to {output_path}")

    elif args.command == "sample":
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = vae.sample(args.num)
        if not isinstance(samples, list):
            samples = [samples]

        for i, img in enumerate(samples):
            img.save(output_path / f"sample_{i:03d}.png")

        print(f"Saved {len(samples)} samples to {output_path}")

    elif args.command == "encode":
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_dir():
            extensions = (".jpg", ".jpeg", ".png", ".bmp")
            image_files = []
            for ext in extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            image_files = sorted(set(image_files))
        else:
            image_files = [input_path]

        print(f"Encoding {len(image_files)} images...")

        for img_path in tqdm(image_files):
            latent = vae.encode(img_path)
            torch.save(latent.cpu(), output_path / f"{img_path.stem}_latent.pt")

        print(f"Saved latents to {output_path}")


if __name__ == "__main__":
    main()
