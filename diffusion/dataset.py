"""
Dataset for Pose-Conditioned Latent Diffusion Training

Provides paired (latent, pose_heatmap) samples for training the diffusion model.

Features:
- Pre-encode images to latents using frozen VAE (LatentCache)
- Load corresponding pose heatmaps
- Data augmentation (horizontal flip with pose mirroring)
- Efficient caching for fast training

Usage:
    from diffusion.dataset import DiffusionDataset, LatentCache
    
    # Pre-encode all images to latents
    cache = LatentCache(vae_checkpoint, image_dir, output_dir)
    cache.encode_all()
    
    # Create dataset
    dataset = DiffusionDataset(
        latent_dir='latents/',
        pose_dir='pose_heatmaps/',
        annotations_file='annotations.json',
    )
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Import pose heatmap generator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pose_heatmaps import PoseHeatmapGenerator, SKELETON_CONNECTIONS


class LatentCache:
    """
    Pre-encode images to latents using a frozen VAE.
    
    This speeds up training by avoiding VAE encoding at each iteration.
    
    Args:
        vae_checkpoint: Path to trained VAE checkpoint
        image_dir: Directory containing images
        output_dir: Directory to save latents
        device: Device to use for encoding
    """
    
    def __init__(
        self,
        vae_checkpoint: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        self.vae_checkpoint = Path(vae_checkpoint)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.vae = None
        self.vae_config = None
    
    def _load_vae(self):
        """Load VAE model from checkpoint."""
        if self.vae is not None:
            return
        
        from vae.model import VAE
        
        checkpoint = torch.load(self.vae_checkpoint, map_location='cpu', weights_only=False)
        
        # Get config from checkpoint (try different keys)
        if 'config' in checkpoint:
            self.vae_config = checkpoint['config']
        elif 'model_config' in checkpoint:
            self.vae_config = checkpoint['model_config']
        elif 'train_config' in checkpoint:
            # Extract VAE-specific config from train_config
            train_cfg = checkpoint['train_config']
            self.vae_config = {
                'in_channels': 3 if train_cfg.get('color', False) else 1,
                'base_channels': train_cfg.get('base_channels', 32),
                'channel_mults': tuple(train_cfg.get('channel_mults', [1, 2, 4])),
                'z_channels': train_cfg.get('z_channels', 4),
                'num_res_blocks': train_cfg.get('num_res_blocks', 2),
            }
        else:
            # Default config
            self.vae_config = {
                'in_channels': 1,
                'base_channels': 32,
                'channel_mults': (1, 2, 4),
                'z_channels': 4,
                'num_res_blocks': 2,
            }
        
        # Create model
        self.vae = VAE(**self.vae_config)
        
        # Load weights - handle different checkpoint formats
        loaded = False
        
        # Try EMA state dict first (preferred for inference)
        if 'ema_state_dict' in checkpoint:
            ema_state = checkpoint['ema_state_dict']
            # EMA stores shadow_params - can be dict or list
            if 'shadow_params' in ema_state:
                shadow_params = ema_state['shadow_params']
                if isinstance(shadow_params, dict):
                    # shadow_params is already a state dict
                    self.vae.load_state_dict(shadow_params)
                    loaded = True
                    print("Loaded VAE from EMA shadow params (dict)")
                elif isinstance(shadow_params, list):
                    # shadow_params is a list, need to reconstruct state dict
                    state_dict = {}
                    param_names = [name for name, _ in self.vae.named_parameters()]
                    for name, shadow in zip(param_names, shadow_params):
                        state_dict[name] = shadow
                    self.vae.load_state_dict(state_dict)
                    loaded = True
                    print("Loaded VAE from EMA shadow params (list)")
            elif isinstance(ema_state, dict) and 'encoder.conv_in.weight' in ema_state:
                # Direct state dict format
                self.vae.load_state_dict(ema_state)
                loaded = True
                print("Loaded VAE from EMA state dict")
        
        if not loaded and 'model_state_dict' in checkpoint:
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            loaded = True
            print("Loaded VAE from model_state_dict")
        
        if not loaded:
            # Try loading checkpoint directly as state dict
            self.vae.load_state_dict(checkpoint)
            print("Loaded VAE from direct checkpoint")
        
        self.vae.eval()
        self.vae.to(self.device)
        
        print(f"VAE config: in_channels={self.vae_config.get('in_channels', 1)}, "
              f"z_channels={self.vae_config.get('z_channels', 4)}")
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess an image."""
        img = Image.open(path)
        
        # Determine if grayscale or color
        in_channels = self.vae_config.get('in_channels', 1)
        
        if in_channels == 1:
            img = img.convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        else:
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
        
        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2 - 1
        
        return img_tensor
    
    @torch.no_grad()
    def encode_single(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Encode a single image to latent."""
        self._load_vae()
        
        img = self._load_image(Path(image_path))
        img = img.unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        # Encode to latent (use mean, not sampled)
        mean, _ = self.vae.encode(img)
        return mean.squeeze(0).cpu()  # (z_channels, H/8, W/8)
    
    @torch.no_grad()
    def encode_all(
        self,
        batch_size: int = 8,
        force: bool = False,
        extensions: tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
    ) -> dict[str, Path]:
        """
        Encode all images in image_dir to latents.
        
        Args:
            batch_size: Batch size for encoding
            force: Re-encode even if latent exists
            extensions: Image file extensions to process
        
        Returns:
            Dict mapping image filename to latent path
        """
        self._load_vae()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.image_dir.glob(f'*{ext}'))
            image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        print(f"Found {len(image_paths)} images in {self.image_dir}")
        
        # Filter out already encoded (unless force)
        if not force:
            to_encode = []
            for path in image_paths:
                latent_path = self.output_dir / f"{path.stem}.pt"
                if not latent_path.exists():
                    to_encode.append(path)
            print(f"Encoding {len(to_encode)} new images (skipping {len(image_paths) - len(to_encode)} existing)")
            image_paths = to_encode
        
        # Encode in batches
        latent_map = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Encoding images'):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load batch
            batch = []
            for path in batch_paths:
                img = self._load_image(path)
                batch.append(img)
            
            if not batch:
                continue
            
            batch = torch.stack(batch).to(self.device)
            
            # Encode
            means, _ = self.vae.encode(batch)
            
            # Save latents
            for path, latent in zip(batch_paths, means):
                latent_path = self.output_dir / f"{path.stem}.pt"
                torch.save(latent.cpu(), latent_path)
                latent_map[path.name] = latent_path
        
        # Build full map including pre-existing
        for ext in extensions:
            for path in self.image_dir.glob(f'*{ext}'):
                latent_path = self.output_dir / f"{path.stem}.pt"
                if latent_path.exists():
                    latent_map[path.name] = latent_path
        
        print(f"Total latents: {len(latent_map)}")
        return latent_map
    
    def get_latent_config(self) -> dict:
        """Get latent space configuration."""
        self._load_vae()
        return {
            'z_channels': self.vae_config.get('z_channels', 4),
            'in_channels': self.vae_config.get('in_channels', 1),
            'latent_height': 80,  # 640 / 8
            'latent_width': 60,   # 480 / 8
        }


class DiffusionDataset(Dataset):
    """
    Dataset for pose-conditioned latent diffusion training.
    
    Provides (latent, pose_heatmap) pairs for training.
    
    Args:
        latent_dir: Directory containing pre-encoded latents (.pt files)
        annotations_file: Path to keypoint annotations JSON
        pose_heatmap_dir: Directory containing pre-generated pose heatmaps (optional)
        image_size: Original image size (width, height) for coordinate scaling
        latent_size: Latent/heatmap size (height, width)
        sigma: Gaussian sigma for heatmap generation
        include_limbs: Include limb heatmaps
        augment: Enable data augmentation
    """
    
    def __init__(
        self,
        latent_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        pose_heatmap_dir: Optional[Union[str, Path]] = None,
        image_size: tuple[int, int] = (480, 640),  # (width, height)
        latent_size: tuple[int, int] = (80, 60),   # (height, width)
        sigma: float = 2.0,
        include_limbs: bool = True,
        augment: bool = True,
    ):
        self.latent_dir = Path(latent_dir)
        self.annotations_file = Path(annotations_file)
        self.pose_heatmap_dir = Path(pose_heatmap_dir) if pose_heatmap_dir else None
        self.image_size = image_size
        self.latent_size = latent_size
        self.sigma = sigma
        self.include_limbs = include_limbs
        self.augment = augment
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Find matching latent files
        self.samples = []
        latent_files = {p.stem: p for p in self.latent_dir.glob('*.pt')}
        
        for filename, ann in self.annotations.items():
            stem = Path(filename).stem
            if stem in latent_files:
                # Handle both formats:
                # 1. {filename: keypoints_list} (direct list)
                # 2. {filename: {'keypoints': keypoints_list}} (dict with keypoints key)
                if isinstance(ann, list):
                    keypoints = ann
                elif isinstance(ann, dict):
                    keypoints = ann.get('keypoints', [])
                else:
                    keypoints = []
                
                self.samples.append({
                    'filename': filename,
                    'latent_path': latent_files[stem],
                    'keypoints': keypoints,
                })
        
        print(f"Found {len(self.samples)} samples with both latent and annotations")
        
        # Create heatmap generator (if not using pre-generated)
        if self.pose_heatmap_dir is None:
            self.heatmap_generator = PoseHeatmapGenerator(
                output_size=latent_size,
                sigma=sigma,
                normalize=True,
            )
        else:
            self.heatmap_generator = None
        
        # Keypoint indices for horizontal flip
        # Left-right pairs in OpenPose format
        self.flip_pairs = [
            (2, 5),   # shoulders
            (3, 6),   # elbows
            (4, 7),   # wrists
            (8, 11),  # hips
            (9, 12),  # knees
            (10, 13), # ankles
            (14, 15), # eyes
            (16, 17), # ears
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _flip_keypoints(self, keypoints: list, width: int) -> list:
        """Horizontally flip keypoints."""
        flipped = [list(kp) for kp in keypoints]
        
        # Flip x coordinates
        for kp in flipped:
            if kp[2] > 0:  # If visible
                kp[0] = width - 1 - kp[0]
        
        # Swap left-right pairs
        for left, right in self.flip_pairs:
            flipped[left], flipped[right] = flipped[right], flipped[left]
        
        return flipped
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a training sample.
        
        Returns:
            dict with:
                - latent: (z_channels, H, W) tensor
                - pose: (pose_channels, H, W) tensor
                - filename: str
        """
        sample = self.samples[idx]
        
        # Load latent
        latent = torch.load(sample['latent_path'], weights_only=True)
        
        # Get keypoints
        keypoints = sample['keypoints']
        
        # Data augmentation: horizontal flip
        do_flip = self.augment and random.random() < 0.5
        
        if do_flip:
            # Flip latent
            latent = torch.flip(latent, dims=[-1])
            
            # Flip keypoints
            keypoints = self._flip_keypoints(keypoints, self.image_size[0])
        
        # Generate or load pose heatmap
        if self.pose_heatmap_dir is not None:
            # Load pre-generated heatmap
            stem = Path(sample['filename']).stem
            heatmap_path = self.pose_heatmap_dir / f"{stem}.pt"
            
            if heatmap_path.exists():
                pose = torch.load(heatmap_path, weights_only=True)
                if do_flip:
                    pose = torch.flip(pose, dims=[-1])
                    # Also need to swap left-right channels
                    # This is complex for pre-generated, so regenerate instead
                    pose = self._generate_heatmap(keypoints)
            else:
                pose = self._generate_heatmap(keypoints)
        else:
            pose = self._generate_heatmap(keypoints)
        
        return {
            'latent': latent,
            'pose': pose,
            'filename': sample['filename'],
        }
    
    def _generate_heatmap(self, keypoints: list) -> torch.Tensor:
        """Generate pose heatmap from keypoints."""
        heatmap = self.heatmap_generator.generate(
            keypoints,
            original_size=self.image_size,
            include_limbs=self.include_limbs,
            return_tensor=False,
        )
        return torch.from_numpy(heatmap)


class DiffusionDatasetFromImages(Dataset):
    """
    Dataset that loads images and encodes on-the-fly (for validation/testing).
    
    Slower but doesn't require pre-caching latents.
    
    Args:
        image_dir: Directory containing images
        annotations_file: Path to keypoint annotations JSON
        vae_checkpoint: Path to VAE checkpoint
        latent_size: Latent/heatmap size (height, width)
        sigma: Gaussian sigma for heatmap generation
        include_limbs: Include limb heatmaps
        device: Device for VAE encoding
    """
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        vae_checkpoint: Union[str, Path],
        latent_size: tuple[int, int] = (80, 60),
        sigma: float = 2.0,
        include_limbs: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.image_dir = Path(image_dir)
        self.annotations_file = Path(annotations_file)
        self.latent_size = latent_size
        self.sigma = sigma
        self.include_limbs = include_limbs
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Find matching images
        self.samples = []
        extensions = ('.png', '.jpg', '.jpeg')
        image_files = {}
        
        for ext in extensions:
            for p in self.image_dir.glob(f'*{ext}'):
                image_files[p.name] = p
        
        for filename, ann in self.annotations.items():
            if filename in image_files:
                # Handle both formats:
                # 1. {filename: keypoints_list} (direct list)
                # 2. {filename: {'keypoints': keypoints_list}} (dict with keypoints key)
                if isinstance(ann, list):
                    keypoints = ann
                elif isinstance(ann, dict):
                    keypoints = ann.get('keypoints', [])
                else:
                    keypoints = []
                
                self.samples.append({
                    'filename': filename,
                    'image_path': image_files[filename],
                    'keypoints': keypoints,
                })
        
        print(f"Found {len(self.samples)} samples")
        
        # Create latent cache (for encoding)
        self.cache = LatentCache(vae_checkpoint, image_dir, '/tmp/latents', device)
        self.cache._load_vae()
        
        # Get image size from first image
        if self.samples:
            img = Image.open(self.samples[0]['image_path'])
            self.image_size = img.size  # (width, height)
        else:
            self.image_size = (480, 640)
        
        # Create heatmap generator
        self.heatmap_generator = PoseHeatmapGenerator(
            output_size=latent_size,
            sigma=sigma,
            normalize=True,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @torch.no_grad()
    def __getitem__(self, idx: int) -> dict:
        """Get a sample with on-the-fly encoding."""
        sample = self.samples[idx]
        
        # Encode image
        latent = self.cache.encode_single(sample['image_path'])
        
        # Generate pose heatmap
        pose = self.heatmap_generator.generate(
            sample['keypoints'],
            original_size=self.image_size,
            include_limbs=self.include_limbs,
            return_tensor=False,
        )
        pose = torch.from_numpy(pose)
        
        return {
            'latent': latent,
            'pose': pose,
            'filename': sample['filename'],
        }


def create_dataloaders(
    latent_dir: Union[str, Path],
    annotations_file: Union[str, Path],
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 4,
    **dataset_kwargs,
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        latent_dir: Directory containing latents
        annotations_file: Path to annotations
        batch_size: Batch size
        val_split: Validation split ratio
        num_workers: DataLoader workers
        **dataset_kwargs: Additional args for DiffusionDataset
    
    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = DiffusionDataset(
        latent_dir=latent_dir,
        annotations_file=annotations_file,
        augment=True,
        **dataset_kwargs,
    )
    
    # Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == '__main__':
    """Test dataset creation."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='latents')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    # Pre-encode images
    print("Pre-encoding images to latents...")
    cache = LatentCache(
        args.vae_checkpoint,
        args.image_dir,
        args.output_dir,
    )
    cache.encode_all(batch_size=args.batch_size)
    
    # Test dataset
    print("\nTesting dataset...")
    dataset = DiffusionDataset(
        latent_dir=args.output_dir,
        annotations_file=args.annotations,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Latent shape: {sample['latent'].shape}")
    print(f"Pose shape: {sample['pose'].shape}")
    print(f"Filename: {sample['filename']}")