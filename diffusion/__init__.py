"""
Diffusion Module for Pose-Conditioned Latent Diffusion

Stage 2 of the Video Model pipeline.

Components:
- UNet: Latent diffusion model with FiLM conditioning
- NoiseScheduler: DDPM/DDIM noise scheduling
- DiffusionDataset: Latent + pose heatmap pairs
- Training and inference utilities
"""

# Lazy imports to avoid cv2 dependency at module import time
# Import specific classes when needed:
#   from diffusion.unet import UNetSimple
#   from diffusion.scheduler import NoiseScheduler
#   from diffusion.dataset import DiffusionDataset, LatentCache

__all__ = [
    "UNet",
    "UNetSimple",
    "FiLMConditioner",
    "NoiseScheduler",
    "DPMSolverScheduler",
    "DiffusionDataset",
    "LatentCache",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    if name in ("UNet", "UNetSimple", "FiLMConditioner"):
        from diffusion.unet import UNet, UNetSimple, FiLMConditioner
        return {"UNet": UNet, "UNetSimple": UNetSimple, "FiLMConditioner": FiLMConditioner}[name]
    elif name in ("NoiseScheduler", "DPMSolverScheduler"):
        from diffusion.scheduler import NoiseScheduler, DPMSolverScheduler
        return {"NoiseScheduler": NoiseScheduler, "DPMSolverScheduler": DPMSolverScheduler}[name]
    elif name in ("DiffusionDataset", "LatentCache"):
        from diffusion.dataset import DiffusionDataset, LatentCache
        return {"DiffusionDataset": DiffusionDataset, "LatentCache": LatentCache}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")