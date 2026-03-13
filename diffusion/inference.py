"""
Inference Utilities for Pose-Conditioned Latent Diffusion

Provides easy-to-use API for:
- Sampling from diffusion model given pose
- Decoding latents to images with VAE
- Visualizing pose heatmaps
- Batch processing

Usage:
    from diffusion.inference import DiffusionInference
    
    # Load models
    inference = DiffusionInference.from_checkpoints(
        diffusion_checkpoint='checkpoints/diffusion/run_name/diffusion_best.pt',
        vae_checkpoint='checkpoints/vae/run_name/vae_best.pt',
    )
    
    # Generate image from pose
    image = inference.generate_from_keypoints(keypoints)
    
    # Generate from pose heatmap
    image = inference.generate_from_heatmap(pose_heatmap)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.unet import UNetSimple
from diffusion.scheduler import NoiseScheduler, DPMSolverScheduler


class DiffusionInference:
    """
    Inference wrapper for pose-conditioned latent diffusion.
    
    Handles loading models, sampling, and decoding to images.
    """
    
    def __init__(
        self,
        diffusion_model: UNetSimple,
        scheduler: NoiseScheduler,
        vae: Optional[torch.nn.Module] = None,
        heatmap_generator: Optional['PoseHeatmapGenerator'] = None,
        device: Optional[torch.device] = None,
        vae_config: Optional[dict] = None,
    ):
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.vae = vae
        self.heatmap_generator = heatmap_generator
        self.vae_config = vae_config or {}
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
        
        if self.vae is not None:
            self.vae.to(device)
            self.vae.eval()
        
        self.scheduler.to(device)
    
    @classmethod
    def from_checkpoints(
        cls,
        diffusion_checkpoint: Union[str, Path],
        vae_checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
    ) -> 'DiffusionInference':
        """
        Load inference from checkpoint files.
        
        Args:
            diffusion_checkpoint: Path to diffusion model checkpoint
            vae_checkpoint: Path to VAE checkpoint (optional, for decoding)
            device: Device to use
            use_ema: Use EMA weights if available
        
        Returns:
            DiffusionInference instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load diffusion checkpoint
        diff_ckpt = torch.load(diffusion_checkpoint, map_location='cpu', weights_only=False)
        
        # Get model config
        model_config = diff_ckpt.get('model_config', {})
        scheduler_config = diff_ckpt.get('scheduler_config', {})
        
        # Create diffusion model
        diffusion_model = UNetSimple(
            in_channels=model_config.get('in_channels', 4),
            base_channels=model_config.get('base_channels', 128),
            channel_mults=tuple(model_config.get('channel_mults', [1, 2, 4])),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            attention_levels=tuple(model_config.get('attention_levels', [1, 2])),
            pose_channels=model_config.get('pose_channels', 35),
        )
        
        # Load weights (prefer EMA)
        if use_ema and 'ema_state_dict' in diff_ckpt:
            # EMA state dict contains the shadow params, need to load them into model
            ema_state = diff_ckpt['ema_state_dict']
            if 'shadow_params' in ema_state:
                # Reconstruct state dict from shadow params
                state_dict = {}
                for name, shadow in zip(
                    diffusion_model.state_dict().keys(),
                    ema_state['shadow_params']
                ):
                    state_dict[name] = shadow
                diffusion_model.load_state_dict(state_dict)
            else:
                diffusion_model.load_state_dict(diff_ckpt['model_state_dict'])
        elif 'model_state_dict' in diff_ckpt:
            diffusion_model.load_state_dict(diff_ckpt['model_state_dict'])
        else:
            diffusion_model.load_state_dict(diff_ckpt)
        
        # Create scheduler
        scheduler = NoiseScheduler(
            num_timesteps=scheduler_config.get('num_timesteps', 1000),
            schedule=scheduler_config.get('schedule', 'cosine'),
            prediction_type=scheduler_config.get('prediction_type', 'epsilon'),
            clip_sample=scheduler_config.get('clip_sample', True),
        )
        
        # Load VAE if provided
        vae = None
        vae_config = None
        if vae_checkpoint is not None:
            from vae.model import VAE
            
            vae_ckpt = torch.load(vae_checkpoint, map_location='cpu', weights_only=False)
            vae_config = vae_ckpt.get('config', {})
            
            vae = VAE(
                in_channels=vae_config.get('in_channels', 1),
                base_channels=vae_config.get('base_channels', 32),
                channel_mults=tuple(vae_config.get('channel_mults', [1, 2, 4])),
                z_channels=vae_config.get('z_channels', 4),
                num_res_blocks=vae_config.get('num_res_blocks', 2),
            )
            
            # Load VAE weights (prefer EMA)
            if 'ema_state_dict' in vae_ckpt:
                vae.load_state_dict(vae_ckpt['ema_state_dict'])
            elif 'model_state_dict' in vae_ckpt:
                vae.load_state_dict(vae_ckpt['model_state_dict'])
            else:
                vae.load_state_dict(vae_ckpt)
        
        # Create heatmap generator
        from tools.pose_heatmaps import PoseHeatmapGenerator
        heatmap_generator = PoseHeatmapGenerator(
            output_size=(80, 60),
            sigma=2.0,
            normalize=True,
        )
        
        return cls(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            vae=vae,
            heatmap_generator=heatmap_generator,
            device=device,
            vae_config=vae_config,
        )
    
    @torch.no_grad()
    def generate_latent(
        self,
        pose: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate latent from pose heatmap using DDIM sampling.
        
        Args:
            pose: (B, pose_channels, H, W) pose heatmaps
            num_steps: Number of DDIM steps (more = better quality, slower)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
            guidance_scale: CFG scale (not implemented yet)
            show_progress: Show progress bar
        
        Returns:
            (B, z_channels, H, W) sampled latents
        """
        pose = pose.to(self.device)
        batch_size = pose.shape[0]
        z_channels = self.diffusion_model.in_channels
        
        shape = (batch_size, z_channels, 80, 60)
        
        latent = self.scheduler.ddim_sample(
            self.diffusion_model,
            shape,
            pose,
            self.device,
            num_steps=num_steps,
            eta=eta,
            show_progress=show_progress,
        )
        
        return latent
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image using VAE.
        
        Args:
            latent: (B, z_channels, H, W) latents
        
        Returns:
            (B, C, H, W) images in [-1, 1] range
        """
        if self.vae is None:
            raise ValueError("VAE not loaded. Provide vae_checkpoint when creating inference.")
        
        latent = latent.to(self.device)
        image = self.vae.decode(latent)
        return image
    
    @torch.no_grad()
    def generate_from_heatmap(
        self,
        pose_heatmap: Union[torch.Tensor, np.ndarray],
        num_steps: int = 50,
        eta: float = 0.0,
        return_pil: bool = True,
    ) -> Union[Image.Image, torch.Tensor, list[Image.Image]]:
        """
        Generate image(s) from pose heatmap(s).
        
        Args:
            pose_heatmap: (pose_channels, H, W) or (B, pose_channels, H, W)
            num_steps: DDIM steps
            eta: DDIM stochasticity
            return_pil: Return PIL Image(s) instead of tensor
        
        Returns:
            Generated image(s)
        """
        # Convert to tensor
        if isinstance(pose_heatmap, np.ndarray):
            pose_heatmap = torch.from_numpy(pose_heatmap).float()
        
        # Add batch dimension if needed
        if pose_heatmap.dim() == 3:
            pose_heatmap = pose_heatmap.unsqueeze(0)
        
        # Generate latent
        latent = self.generate_latent(pose_heatmap, num_steps, eta)
        
        # Decode to image
        image = self.decode_latent(latent)
        
        if return_pil:
            return self._tensor_to_pil(image)
        return image
    
    @torch.no_grad()
    def generate_from_keypoints(
        self,
        keypoints: Union[list, np.ndarray],
        original_size: tuple[int, int] = (480, 640),
        include_limbs: bool = True,
        num_steps: int = 50,
        eta: float = 0.0,
        return_pil: bool = True,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Generate image from keypoints.
        
        Args:
            keypoints: List of [x, y, visibility] for 18 keypoints
            original_size: (width, height) of coordinate system
            include_limbs: Include limb heatmaps
            num_steps: DDIM steps
            eta: DDIM stochasticity
            return_pil: Return PIL Image instead of tensor
        
        Returns:
            Generated image
        """
        if self.heatmap_generator is None:
            raise ValueError("Heatmap generator not available")
        
        # Generate heatmap
        heatmap = self.heatmap_generator.generate(
            keypoints,
            original_size=original_size,
            include_limbs=include_limbs,
            return_tensor=False,
        )
        
        return self.generate_from_heatmap(heatmap, num_steps, eta, return_pil)
    
    @torch.no_grad()
    def generate_batch_from_annotations(
        self,
        annotations_file: Union[str, Path],
        output_dir: Union[str, Path],
        num_steps: int = 50,
        image_size: tuple[int, int] = (480, 640),
        include_limbs: bool = True,
        show_progress: bool = True,
    ):
        """
        Generate images for all annotations in a file.
        
        Args:
            annotations_file: Path to annotations JSON
            output_dir: Directory to save generated images
            num_steps: DDIM steps
            image_size: Original image size (width, height)
            include_limbs: Include limb heatmaps
            show_progress: Show progress bar
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        items = list(annotations.items())
        if show_progress:
            items = tqdm(items, desc='Generating images')
        
        for filename, ann in items:
            keypoints = ann.get('keypoints', [])
            if not keypoints:
                continue
            
            # Generate image
            image = self.generate_from_keypoints(
                keypoints,
                original_size=image_size,
                include_limbs=include_limbs,
                num_steps=num_steps,
                return_pil=True,
            )
            
            # Save
            stem = Path(filename).stem
            output_path = output_dir / f"{stem}_generated.png"
            image.save(output_path)
    
    def _tensor_to_pil(
        self,
        tensor: torch.Tensor,
    ) -> Union[Image.Image, list[Image.Image]]:
        """Convert tensor to PIL Image(s)."""
        # Move to CPU and convert to numpy
        tensor = tensor.cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        if tensor.dim() == 4:
            # Batch
            images = []
            for i in range(tensor.shape[0]):
                img = tensor[i]
                if img.shape[0] == 1:
                    # Grayscale
                    img = img.squeeze(0).numpy()
                    img = (img * 255).astype(np.uint8)
                    images.append(Image.fromarray(img, mode='L'))
                else:
                    # RGB
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    images.append(Image.fromarray(img, mode='RGB'))
            
            return images if len(images) > 1 else images[0]
        else:
            # Single image
            if tensor.shape[0] == 1:
                img = tensor.squeeze(0).numpy()
                img = (img * 255).astype(np.uint8)
                return Image.fromarray(img, mode='L')
            else:
                img = tensor.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                return Image.fromarray(img, mode='RGB')
    
    @staticmethod
    def visualize_pose_heatmap(
        heatmap: Union[torch.Tensor, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Visualize pose heatmap as an image.
        
        Args:
            heatmap: (pose_channels, H, W) heatmap
            output_path: Optional path to save image
        
        Returns:
            Visualization as PIL Image
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        
        # Sum all channels and normalize
        combined = heatmap.sum(axis=0)
        combined = combined / combined.max() if combined.max() > 0 else combined
        
        # Convert to image
        img_array = (combined * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Resize for better visualization
        img = img.resize((480, 640), Image.Resampling.NEAREST)
        
        if output_path:
            img.save(output_path)
        
        return img
    
    @staticmethod
    def create_comparison_grid(
        pose_image: Image.Image,
        generated_image: Image.Image,
        ground_truth: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Create a comparison grid of pose, generated, and optionally ground truth.
        
        Args:
            pose_image: Pose visualization
            generated_image: Generated image
            ground_truth: Optional ground truth image
        
        Returns:
            Grid image
        """
        # Resize all to same size
        size = (480, 640)
        pose_image = pose_image.resize(size, Image.Resampling.BILINEAR)
        generated_image = generated_image.resize(size, Image.Resampling.BILINEAR)
        
        if ground_truth:
            ground_truth = ground_truth.resize(size, Image.Resampling.BILINEAR)
            width = size[0] * 3
            grid = Image.new('RGB', (width, size[1]))
            grid.paste(pose_image.convert('RGB'), (0, 0))
            grid.paste(generated_image.convert('RGB'), (size[0], 0))
            grid.paste(ground_truth.convert('RGB'), (size[0] * 2, 0))
        else:
            width = size[0] * 2
            grid = Image.new('RGB', (width, size[1]))
            grid.paste(pose_image.convert('RGB'), (0, 0))
            grid.paste(generated_image.convert('RGB'), (size[0], 0))
        
        return grid


def main():
    """Command-line interface for diffusion inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diffusion inference')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate samples from annotations')
    sample_parser.add_argument('--diffusion_checkpoint', '-d', type=str, required=True,
                               help='Diffusion model checkpoint')
    sample_parser.add_argument('--vae_checkpoint', '-v', type=str, required=True,
                               help='VAE checkpoint')
    sample_parser.add_argument('--annotations', '-a', type=str, required=True,
                               help='Keypoint annotations JSON')
    sample_parser.add_argument('--output_dir', '-o', type=str, default='generated',
                               help='Output directory')
    sample_parser.add_argument('--num_steps', type=int, default=50,
                               help='DDIM sampling steps')
    sample_parser.add_argument('--num_samples', type=int, default=None,
                               help='Number of samples to generate (default: all)')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive generation')
    interactive_parser.add_argument('--diffusion_checkpoint', '-d', type=str, required=True)
    interactive_parser.add_argument('--vae_checkpoint', '-v', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'sample':
        print("Loading models...")
        inference = DiffusionInference.from_checkpoints(
            args.diffusion_checkpoint,
            args.vae_checkpoint,
        )
        
        print("Generating samples...")
        inference.generate_batch_from_annotations(
            args.annotations,
            args.output_dir,
            num_steps=args.num_steps,
        )
        print(f"Saved to {args.output_dir}")
    
    elif args.command == 'interactive':
        print("Loading models...")
        inference = DiffusionInference.from_checkpoints(
            args.diffusion_checkpoint,
            args.vae_checkpoint,
        )
        print("Models loaded. Use inference.generate_from_keypoints() to generate images.")
        # Start interactive session
        import code
        code.interact(local={'inference': inference, 'torch': torch, 'np': np})


if __name__ == '__main__':
    main()