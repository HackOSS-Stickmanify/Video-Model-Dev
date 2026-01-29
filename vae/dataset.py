"""
Dataset for grayscale VAE training with strong augmentation.
Handles resizing/padding portrait images to 480×640 (W×H).
"""

import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class GrayscaleImageDataset(Dataset):
    """
    Dataset for grayscale images with augmentation.
    
    Target size: 480×640 (width×height, portrait)
    Images are resized to fit height, then padded width if needed.
    """
    
    def __init__(
        self,
        image_dir: str,
        target_size: tuple[int, int] = (480, 640),  # (width, height)
        augment: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.target_size = target_size  # (W, H)
        self.augment = augment
        
        # Collect image paths
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _resize_and_pad(self, img: Image.Image) -> Image.Image:
        """Resize to fit target height, then pad width to target."""
        target_w, target_h = self.target_size
        orig_w, orig_h = img.size
        
        # Scale to match target height
        scale = target_h / orig_h
        new_w = int(orig_w * scale)
        new_h = target_h
        
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad width if needed (center padding)
        if new_w < target_w:
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            # Create new image with padding (white = 255)
            padded = Image.new('L', (target_w, target_h), 255)
            padded.paste(img, (pad_left, 0))
            img = padded
        elif new_w > target_w:
            # Center crop width
            left = (new_w - target_w) // 2
            img = img.crop((left, 0, left + target_w, target_h))
        
        return img
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply augmentations for small dataset."""
        
        # Random horizontal flip (50%)
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        # Random slight rotation (-5 to +5 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=255)
        
        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Random small affine (scale + translate)
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            translate_x = random.uniform(-0.02, 0.02)
            translate_y = random.uniform(-0.02, 0.02)
            img = TF.affine(
                img,
                angle=0,
                translate=(int(translate_x * self.target_size[0]), int(translate_y * self.target_size[1])),
                scale=scale,
                shear=0,
                fill=255,
            )
        
        return img
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        
        # Load as grayscale
        img = Image.open(img_path).convert('L')
        
        # Resize and pad to target size
        img = self._resize_and_pad(img)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self._augment(img)
        
        # Convert to tensor [0, 1] then normalize to [-1, 1]
        tensor = TF.to_tensor(img)  # [1, H, W] in [0, 1]
        tensor = tensor * 2 - 1      # [-1, 1]
        
        return tensor


class PreprocessedDataset(Dataset):
    """
    Dataset for pre-resized images (faster loading).
    """
    
    def __init__(
        self,
        image_dir: str,
        augment: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.augment = augment
        
        # Collect image paths
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} preprocessed images")
        
        # Get size from first image
        with Image.open(self.image_paths[0]) as img:
            self.target_size = img.size  # (W, H)
        print(f"Image size: {self.target_size}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply augmentations."""
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=255)
        
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_contrast(img, contrast_factor)
        
        return img
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert('L')
        
        if self.augment:
            img = self._augment(img)
        
        tensor = TF.to_tensor(img) * 2 - 1
        return tensor


if __name__ == '__main__':
    # Test dataset
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = '../input_stickman_video/all_bw_images'
    
    dataset = GrayscaleImageDataset(image_dir, augment=True)
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
