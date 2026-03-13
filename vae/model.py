"""
Spatial Convolutional VAE for Images

Supports both grayscale (1 channel) and color (3 channel RGB) images.

- Input: C×640×480 (C×H×W, portrait) where C=1 (grayscale) or C=3 (RGB)
- Latent: 4×80×60 (C×H×W)
- Downsample factor: ×8
- Uses GroupNorm + SiLU
- Supports EMA and LPIPS perceptual loss

Examples:
    # Grayscale VAE (default)
    model = VAE(in_channels=1)

    # Color VAE
    model = VAE(in_channels=3)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by factor of 2 using strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2 using nearest + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder: C×640×480 → z×80×60
    Channel progression: C → 32 → 64 → 128 → z (z_channels)
    3 downsamples = ×8

    Args:
        in_channels: Input image channels (1 for grayscale, 3 for RGB)
        base_channels: Base channel count (default 32)
        channel_mults: Channel multipliers for each level
        z_channels: Latent space channels
        num_res_blocks: Number of residual blocks per level
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        z_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = base_channels
        self.down_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(channels, out_channels))
                channels = out_channels

            # Downsample (except last level)
            if i < len(channel_mults) - 1:
                self.down_blocks.append(Downsample(channels))

        # Final downsample to get ×8 total
        self.down_blocks.append(Downsample(channels))

        # Mid blocks
        self.mid_block1 = ResBlock(channels, channels)
        self.mid_block2 = ResBlock(channels, channels)

        # Output conv to latent (mean and logvar)
        self.norm_out = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(
            channels, z_channels * 2, 3, padding=1
        )  # *2 for mean + logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    """
    Decoder: z×80×60 → C×640×480
    Channel progression: z → 128 → 64 → 32 → C
    3 upsamples = ×8

    Args:
        out_channels: Output image channels (1 for grayscale, 3 for RGB)
        base_channels: Base channel count (default 32)
        channel_mults: Channel multipliers for each level
        z_channels: Latent space channels
        num_res_blocks: Number of residual blocks per level
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        z_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        # Start from highest channel count
        channels = base_channels * channel_mults[-1]

        self.conv_in = nn.Conv2d(z_channels, channels, 3, padding=1)

        # Mid blocks
        self.mid_block1 = ResBlock(channels, channels)
        self.mid_block2 = ResBlock(channels, channels)

        # Initial upsample
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(Upsample(channels))

        # Reverse channel mults for decoder
        reversed_mults = list(reversed(channel_mults))

        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock(channels, out_ch))
                channels = out_ch

            # Upsample (except last level)
            if i < len(reversed_mults) - 1:
                self.up_blocks.append(Upsample(channels))

        # Output
        self.norm_out = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h


class VAE(nn.Module):
    """
    Spatial VAE for grayscale or color images.

    - Input/Output: C×640×480 where C=1 (grayscale) or C=3 (RGB)
    - Latent: z×80×60 (default z=4)
    - Downsample: ×8

    Supports EMA (Exponential Moving Average) for better inference.

    Args:
        in_channels: Input/output channels (1 for grayscale, 3 for RGB)
        base_channels: Base channel count (default 32)
        channel_mults: Channel multipliers for encoder/decoder levels
        z_channels: Latent space channels (default 4)
        num_res_blocks: Number of residual blocks per level

    Examples:
        # Grayscale VAE (default)
        model = VAE(in_channels=1)

        # Color VAE
        model = VAE(in_channels=3)

        # Larger latent space for color
        model = VAE(in_channels=3, z_channels=8)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        z_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
        )

        self.z_channels = z_channels
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (reconstruction, mean, logvar)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get latent representation (for inference)."""
        mean, logvar = self.encode(x)
        if deterministic:
            return mean
        return self.reparameterize(mean, logvar)

    def sample(
        self, batch_size: int = 1, device: Union[torch.device, None] = None
    ) -> torch.Tensor:
        """Sample from the prior and decode."""
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)

        # Sample from standard normal (latent space is 4×80×60)
        z = torch.randn(batch_size, self.z_channels, 80, 60, device=device)
        return self.decode(z)

    def get_config(self) -> dict:
        """Return model configuration for saving."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "channel_mults": self.channel_mults,
            "z_channels": self.z_channels,
            "num_res_blocks": self.num_res_blocks,
        }

    @classmethod
    def from_config(cls, config: dict) -> "VAE":
        """Create model from configuration dict."""
        return cls(**config)


# Backwards compatibility alias
GrayscaleVAE = VAE


class EMA:
    """
    Exponential Moving Average for model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # During training:
        ema.update()
        # For inference:
        with ema.average_parameters():
            output = model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0

        # Create shadow parameters
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup_params = {}

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        self.step += 1

        if self.step < self.update_after_step:
            return

        if self.step % self.update_every != 0:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].lerp_(param.data, 1 - self.decay)

    def store(self):
        """Store current model parameters."""
        self.backup_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

    def restore(self):
        """Restore model parameters from backup."""
        for name, param in self.model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def copy_to(self):
        """Copy shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def average_parameters(self):
        """Context manager to temporarily use EMA parameters."""

        class _EMAContext:
            def __init__(self, ema: "EMA"):
                self.ema = ema

            def __enter__(self):
                self.ema.store()
                self.ema.copy_to()
                return self.ema.model

            def __exit__(self, *args):
                self.ema.restore()

        return _EMAContext(self)  # type: ignore[return-value]

    def state_dict(self) -> dict:
        """Return EMA state for saving."""
        return {
            "shadow_params": self.shadow_params,
            "step": self.step,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state."""
        self.shadow_params = state_dict["shadow_params"]
        self.step = state_dict["step"]
        self.decay = state_dict.get("decay", self.decay)


class VAELoss(nn.Module):
    """
    Combined VAE loss with optional perceptual loss.

    Loss = L1 + kl_weight * KL + lpips_weight * LPIPS

    Args:
        kl_weight: Weight for KL divergence term (β in β-VAE)
        lpips_weight: Weight for LPIPS perceptual loss
        use_lpips: Whether to use LPIPS loss (requires lpips package)
    """

    def __init__(
        self,
        kl_weight: float = 0.01,
        lpips_weight: float = 0.1,
        use_lpips: bool = False,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_weight = lpips_weight
        self.use_lpips = use_lpips

        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips

                # Use VGG for perceptual loss (grayscale will be converted to 3-channel)
                self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
                # Freeze LPIPS network
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips package not found. Disabling perceptual loss.")
                self.use_lpips = False

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute VAE loss.

        Args:
            recon: Reconstructed image (B, 1, H, W) in [-1, 1]
            target: Original image (B, 1, H, W) in [-1, 1]
            mean: Latent mean
            logvar: Latent log variance

        Returns:
            total_loss, loss_dict
        """
        # L1 reconstruction loss
        recon_loss = F.l1_loss(recon, target)

        # KL divergence (per-element, then mean)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Start with base losses
        total_loss = recon_loss + self.kl_weight * kl_loss

        loss_dict = {
            "total": 0.0,  # Will be updated
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
        }

        # LPIPS perceptual loss (optional)
        if self.use_lpips and self.lpips_fn is not None:
            # Convert grayscale to 3-channel for LPIPS
            recon_rgb = recon.repeat(1, 3, 1, 1)
            target_rgb = target.repeat(1, 3, 1, 1)

            # Move LPIPS to same device if needed
            if next(self.lpips_fn.parameters()).device != recon.device:
                self.lpips_fn = self.lpips_fn.to(recon.device)

            lpips_loss = self.lpips_fn(recon_rgb, target_rgb).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss
            loss_dict["lpips"] = lpips_loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute VAE loss: L1 reconstruction + KL divergence.

    This is a simple function for backward compatibility.
    For more options, use the VAELoss class.

    Args:
        recon: Reconstructed image
        target: Original image
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL term (β in β-VAE)

    Returns:
        total_loss, loss_dict
    """
    # L1 reconstruction loss
    recon_loss = F.l1_loss(recon, target)

    # KL divergence (per-element, then mean)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, {
        "total": total_loss.item(),
        "recon": recon_loss.item(),
        "kl": kl_loss.item(),
    }


if __name__ == "__main__":
    # Test the model
    device = get_best_device()
    print(f"Using device: {device}")

    model = VAE().to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass (1×640×480 portrait)
    x = torch.randn(2, 1, 640, 480).to(device)
    recon, mean, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {mean.shape}")
    print(f"Output shape: {recon.shape}")

    # Test simple loss function
    loss, loss_dict = compute_vae_loss(recon, x, mean, logvar)
    print(f"Loss (simple): {loss_dict}")

    # Test VAELoss class
    vae_loss = VAELoss(kl_weight=0.01, lpips_weight=0.1, use_lpips=False)
    loss, loss_dict = vae_loss(recon, x, mean, logvar)
    print(f"Loss (class): {loss_dict}")

    # Test EMA
    ema = EMA(model, decay=0.999)
    ema.update()
    print("EMA update successful")

    # Test with EMA parameters
    with ema.average_parameters():
        recon_ema, _, _ = model(x)
    print(f"EMA inference output shape: {recon_ema.shape}")

    # Test config save/load
    config = model.get_config()
    print(f"Model config: {config}")

    model2 = VAE.from_config(config).to(device)
    print("Config load successful")

    # Test sampling
    samples = model.sample(batch_size=2, device=device)
    print(f"Sample shape: {samples.shape}")

    print("\nAll tests passed!")
