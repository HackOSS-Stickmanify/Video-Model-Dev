"""
Spatial Convolutional VAE for Grayscale Images
- Input: 1×640×480 (C×H×W, portrait)
- Latent: 4×80×60 (C×H×W)
- Downsample factor: ×8
- Uses GroupNorm + SiLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
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
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder: 1×640×480 → 4×80×60
    Channel progression: 1 → 32 → 64 → 128 → 4 (z_channels)
    3 downsamples = ×8
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
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
        self.conv_out = nn.Conv2d(channels, z_channels * 2, 3, padding=1)  # *2 for mean + logvar
    
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
    Decoder: 4×80×60 → 1×640×480
    Channel progression: 4 → 128 → 64 → 32 → 1
    3 upsamples = ×8
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
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


class GrayscaleVAE(nn.Module):
    """
    Grayscale Spatial VAE
    - Input/Output: 1×640×480
    - Latent: 4×80×60
    - Downsample: ×8
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Compute VAE loss: L1 reconstruction + KL divergence.
    
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
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
    }


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GrayscaleVAE().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass (1×640×480 portrait)
    x = torch.randn(2, 1, 640, 480).to(device)
    recon, mean, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {mean.shape}")
    print(f"Output shape: {recon.shape}")
    
    # Test loss
    loss, loss_dict = compute_vae_loss(recon, x, mean, logvar)
    print(f"Loss: {loss_dict}")
