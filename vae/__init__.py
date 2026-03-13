"""
VAE module for video generation.

This module provides a spatial convolutional VAE for compressing
images (C×640×480) to a latent space (z×80×60).
Supports both grayscale (1 channel) and color (3 channel RGB) images.

Components:
- VAE: Main VAE model with encoder and decoder (supports grayscale and color)
- VAELoss: Combined loss function with L1, KL, and optional LPIPS
- EMA: Exponential Moving Average for smoother inference weights
- VAEInference: High-level inference API for encoding/decoding

Device Support:
- CUDA (NVIDIA GPUs)
- XPU (Intel GPUs via intel_extension_for_pytorch)
- CPU (fallback)
"""

from .model import (
    VAE,
    GrayscaleVAE,  # Backwards compatibility alias
    VAELoss,
    EMA,
    compute_vae_loss,
    get_best_device,
    HAS_CUDA,
    HAS_XPU,
)
from .dataset import GrayscaleImageDataset, PreprocessedDataset
from .inference import VAEInference

__version__ = "0.1.0"
__all__ = [
    # Model
    "VAE",
    "GrayscaleVAE",  # Backwards compatibility alias
    "VAELoss",
    "EMA",
    "compute_vae_loss",
    # Dataset
    "ImageDataset",
    "GrayscaleImageDataset",  # Backwards compatibility alias
    "PreprocessedDataset",
    # Inference
    "VAEInference",
    # Device utilities
    "get_best_device",
    "HAS_CUDA",
    "HAS_XPU",
]