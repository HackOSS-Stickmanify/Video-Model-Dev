"""
Dataset for VAE training with strong augmentation.
Handles resizing/padding portrait images to 480×640 (W×H).

Supports both grayscale (1 channel) and color (3 channel RGB) images.
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


class ImageDataset(Dataset):
    """
    Dataset for grayscale or color images with augmentation.
    
    Target size: 480×640 (width×height, portrait)
    Images are resized to fit height, then padded width if needed.
    
    Args:
        image_dir: Directory containing images
        target_size: Target (width, height) tuple
        augment: Whether to apply data augmentation
        extensions: File extensions to look for
        color_mode: "L" for grayscale, "RGB" for color
    """
    
    def __init__(
        self,
        image_dir: str,
        target_size: tuple[int, int] = (480, 640),  # (width, height)
        augment: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        color_mode: str = "L",  # "L" for grayscale, "RGB" for color
    ):
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.target_size = target_size  # (W, H)
        self.augment = augment
        self.color_mode = color_mode.upper()
        
        if self.color_mode not in ("L", "RGB"):
            raise ValueError(f"color_mode must be 'L' or 'RGB', got {color_mode}")
        
        # Padding color (white)
        self.pad_color = 255 if self.color_mode == "L" else (255, 255, 255)
        
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
            # Create new image with padding
            padded = Image.new(self.color_mode, (target_w, target_h), self.pad_color)
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
            img = TF.rotate(img, angle, fill=self.pad_color)
        
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
                fill=self.pad_color,
            )
        
        # Color-specific augmentations
        if self.color_mode == "RGB":
            # Random saturation adjustment
            if random.random() > 0.5:
                saturation_factor = random.uniform(0.8, 1.2)
                img = TF.adjust_saturation(img, saturation_factor)
            
            # Random hue adjustment (small)
            if random.random() > 0.5:
                hue_factor = random.uniform(-0.05, 0.05)
                img = TF.adjust_hue(img, hue_factor)
        
        return img
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        
        # Load in appropriate color mode
        img = Image.open(img_path).convert(self.color_mode)
        
        # Resize and pad to target size
        img = self._resize_and_pad(img)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self._augment(img)
        
        # Convert to tensor [0, 1] then normalize to [-1, 1]
        # Grayscale: [1, H, W], RGB: [3, H, W]
        tensor = TF.to_tensor(img)  # [C, H, W] in [0, 1]
        tensor = tensor * 2 - 1      # [-1, 1]
        
        return tensor


# Backwards compatibility alias
GrayscaleImageDataset = ImageDataset


class PreprocessedDataset(Dataset):
    """
    Dataset for pre-resized images (faster loading).
    
    Args:
        image_dir: Directory containing preprocessed images
        augment: Whether to apply data augmentation
        extensions: File extensions to look for
        color_mode: "L" for grayscale, "RGB" for color
    """
    
    def __init__(
        self,
        image_dir: str,
        augment: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        color_mode: str = "L",  # "L" for grayscale, "RGB" for color
    ):
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.color_mode = color_mode.upper()
        
        if self.color_mode not in ("L", "RGB"):
            raise ValueError(f"color_mode must be 'L' or 'RGB', got {color_mode}")
        
        self.pad_color = 255 if self.color_mode == "L" else (255, 255, 255)
        
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
            img = TF.rotate(img, angle, fill=self.pad_color)
        
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Color-specific augmentations
        if self.color_mode == "RGB":
            if random.random() > 0.5:
                saturation_factor = random.uniform(0.8, 1.2)
                img = TF.adjust_saturation(img, saturation_factor)
            
            if random.random() > 0.5:
                hue_factor = random.uniform(-0.05, 0.05)
                img = TF.adjust_hue(img, hue_factor)
        
        return img
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert(self.color_mode)
        
        if self.augment:
            img = self._augment(img)
        
        # Grayscale: [1, H, W], RGB: [3, H, W]
        tensor = TF.to_tensor(img) * 2 - 1
        return tensor


if __name__ == '__main__':
    # Test dataset
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = '../input_stickman_video/all_bw_images'
    
    # Test grayscale
    print("Testing grayscale dataset:")
    dataset = ImageDataset(image_dir, augment=True, color_mode="L")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
    
    # Test color
    print("\nTesting color dataset:")
    dataset_rgb = ImageDataset(image_dir, augment=True, color_mode="RGB")
    print(f"Dataset size: {len(dataset_rgb)}")
    sample_rgb = dataset_rgb[0]
    print(f"Sample shape: {sample_rgb.shape}")
    print(f"Sample range: [{sample_rgb.min():.2f}, {sample_rgb.max():.2f}]")
