"""
UNet Model for Pose-Conditioned Latent Diffusion

Architecture:
- Input: noisy latent (z_channels × 80 × 60) + timestep embedding
- Conditioning: pose heatmaps (35 × 80 × 60) via FiLM (Feature-wise Linear Modulation)
- Output: predicted noise (same shape as input)

FiLM Conditioning:
- Pose heatmaps are projected to scale/shift parameters
- Applied at each resolution level: output = scale * features + shift

Design choices:
- GroupNorm for batch-size independence
- SiLU activation (smooth, works well with diffusion)
- Self-attention at lower resolutions (40×30, 20×15)
- Sinusoidal timestep embedding
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.
    
    Args:
        timesteps: (batch_size,) tensor of timesteps
        dim: Embedding dimension
    
    Returns:
        (batch_size, dim) embedding tensor
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning from pose heatmaps.
    
    Projects pose heatmaps to scale and shift parameters for each feature channel.
    
    Args:
        pose_channels: Number of pose heatmap channels (default 35: 18 keypoints + 17 limbs)
        out_channels: Number of output channels for scale/shift
        hidden_dim: Hidden dimension for projection
    """
    
    def __init__(
        self,
        pose_channels: int = 35,
        out_channels: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # Encode pose heatmaps
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Project to scale and shift
        self.to_scale = nn.Conv2d(hidden_dim, out_channels, 1)
        self.to_shift = nn.Conv2d(hidden_dim, out_channels, 1)
        
        # Initialize scale to 1, shift to 0
        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)
    
    def forward(self, pose_heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose_heatmaps: (B, pose_channels, H, W) pose heatmaps
        
        Returns:
            scale: (B, out_channels, H, W)
            shift: (B, out_channels, H, W)
        """
        h = self.encoder(pose_heatmaps)
        scale = self.to_scale(h)
        shift = self.to_shift(h)
        return scale, shift


class FiLMBlock(nn.Module):
    """
    FiLM modulation block - applies scale and shift from conditioning.
    """
    
    def __init__(self, channels: int, pose_channels: int = 35):
        super().__init__()
        self.film = FiLMConditioner(pose_channels, channels)
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: (B, C, H, W) features
            pose: (B, pose_channels, H_pose, W_pose) pose heatmaps
        
        Returns:
            Modulated features (B, C, H, W)
        """
        # Resize pose to match x if needed
        if pose.shape[-2:] != x.shape[-2:]:
            pose = F.interpolate(pose, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        scale, shift = self.film(pose)
        return scale * self.norm(x) + shift


class ResBlock(nn.Module):
    """
    Residual block with timestep embedding and optional FiLM conditioning.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_emb_dim: Timestep embedding dimension
        pose_channels: Pose heatmap channels (if using FiLM)
        use_film: Whether to use FiLM conditioning
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        pose_channels: int = 35,
        use_film: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.use_film = use_film
        
        # First conv
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Timestep embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # Second conv
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # FiLM conditioning
        if use_film:
            self.film = FiLMConditioner(pose_channels, out_channels)
        
        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.act = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            time_emb: (B, time_emb_dim) timestep embedding
            pose: (B, pose_channels, H, W) pose heatmaps (optional)
        
        Returns:
            (B, out_channels, H, W) output features
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        
        # Add timestep embedding
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Apply FiLM conditioning
        if self.use_film and pose is not None:
            # Resize pose to match h if needed
            if pose.shape[-2:] != h.shape[-2:]:
                pose = F.interpolate(pose, size=h.shape[-2:], mode='bilinear', align_corners=False)
            scale, shift = self.film(pose)
            h = scale * h + shift
        
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Self-attention block for capturing global dependencies.
    
    Args:
        channels: Number of channels
        num_heads: Number of attention heads
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out


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


class UNet(nn.Module):
    """
    UNet for pose-conditioned latent diffusion.
    
    Architecture:
    - Encoder: progressively downsample with ResBlocks + attention
    - Bottleneck: ResBlocks + attention at lowest resolution
    - Decoder: progressively upsample with skip connections
    - Conditioning: FiLM from pose heatmaps at each level
    
    Args:
        in_channels: Latent channels (4 for grayscale VAE, 8 for color VAE)
        base_channels: Base channel count
        channel_mults: Channel multipliers for each level
        num_res_blocks: ResBlocks per level
        attention_levels: Which levels to apply attention (0=highest res)
        pose_channels: Pose heatmap channels (default 35)
        dropout: Dropout rate
        time_emb_dim: Timestep embedding dimension
    
    Examples:
        # For grayscale VAE (z=4)
        model = UNet(in_channels=4)
        
        # For color VAE (z=8)
        model = UNet(in_channels=8)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (1, 2, 3),  # Apply attention at levels 1, 2, 3
        pose_channels: int = 35,
        dropout: float = 0.0,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.pose_channels = pose_channels
        
        num_levels = len(channel_mults)
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = base_channels
        encoder_channels = [ch]
        
        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]
            use_attn = level in attention_levels
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(
                        ch, out_ch, time_emb_dim, pose_channels,
                        use_film=True, dropout=dropout
                    )
                )
                ch = out_ch
                encoder_channels.append(ch)
                
                if use_attn:
                    self.encoder_blocks.append(SelfAttention(ch))
                    encoder_channels.append(ch)
            
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)
        
        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, pose_channels, use_film=True, dropout=dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, pose_channels, use_film=True, dropout=dropout)
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]
            use_attn = level in attention_levels
            
            for i in range(num_res_blocks + 1):
                # Pop skip connection channel count
                skip_ch = encoder_channels.pop()
                
                self.decoder_blocks.append(
                    ResBlock(
                        ch + skip_ch, out_ch, time_emb_dim, pose_channels,
                        use_film=True, dropout=dropout
                    )
                )
                ch = out_ch
                
                if use_attn:
                    self.decoder_blocks.append(SelfAttention(ch))
                    if i < num_res_blocks:
                        encoder_channels.pop()  # Account for attention in encoder
            
            if level > 0:
                self.upsamplers.append(Upsample(ch))
                encoder_channels.pop()  # Account for downsampler in encoder
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)
        
        # Initialize output conv to zero (better initial predictions)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, in_channels, H, W) noisy latent
            timestep: (B,) timestep values
            pose: (B, pose_channels, H, W) pose heatmaps
        
        Returns:
            (B, in_channels, H, W) predicted noise
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(timestep, self.base_channels)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skips = [h]
        
        down_idx = 0
        for block in self.encoder_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb, pose)
            elif isinstance(block, SelfAttention):
                h = block(h)
            skips.append(h)
            
            # Check if we need to downsample
            if isinstance(block, ResBlock) and down_idx < len(self.downsamplers):
                # Count ResBlocks at current level
                # Downsample after num_res_blocks ResBlocks
                pass
        
        # Apply downsamplers in order
        down_idx = 0
        h = self.conv_in(x)
        skips = [h]
        
        block_idx = 0
        level_block_count = 0
        current_level = 0
        
        for block in self.encoder_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb, pose)
                level_block_count += 1
            elif isinstance(block, SelfAttention):
                h = block(h)
            skips.append(h)
            
        # Simplified: apply downsamplers between levels
        # Re-implement forward pass more cleanly
        
        # Actually let me rewrite the forward pass properly
        h = self.conv_in(x)
        skips = [h]
        
        num_levels = len(self.channel_mults)
        block_idx = 0
        
        for level in range(num_levels):
            use_attn = level in self.attention_levels
            
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb, pose)
                block_idx += 1
                skips.append(h)
                
                if use_attn:
                    h = self.encoder_blocks[block_idx](h)
                    block_idx += 1
                    skips.append(h)
            
            if level < num_levels - 1:
                h = self.downsamplers[level](h)
                skips.append(h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb, pose)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, pose)
        
        # Decoder
        block_idx = 0
        up_idx = 0
        
        for level in reversed(range(num_levels)):
            use_attn = level in self.attention_levels
            
            for i in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb, pose)
                block_idx += 1
                
                if use_attn:
                    h = self.decoder_blocks[block_idx](h)
                    block_idx += 1
                    if i < self.num_res_blocks:
                        skips.pop()  # Pop attention skip
            
            if level > 0:
                h = self.upsamplers[up_idx](h)
                up_idx += 1
                skips.pop()  # Pop downsampler skip
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "channel_mults": self.channel_mults,
            "num_res_blocks": self.num_res_blocks,
            "attention_levels": self.attention_levels,
            "pose_channels": self.pose_channels,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "UNet":
        """Create model from config dict."""
        return cls(**config)


class UNetSimple(nn.Module):
    """
    Simplified UNet for pose-conditioned latent diffusion.
    
    Cleaner implementation with explicit encoder/decoder stages.
    
    Args:
        in_channels: Latent channels (4 for grayscale VAE, 8 for color VAE)
        base_channels: Base channel count
        channel_mults: Channel multipliers for each level
        num_res_blocks: ResBlocks per level
        attention_levels: Which levels to apply attention (0=highest res)
        pose_channels: Pose heatmap channels (default 35)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (1, 2),
        pose_channels: int = 35,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.pose_channels = pose_channels
        
        time_emb_dim = base_channels * 4
        num_levels = len(channel_mults)
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Pose encoder - encode once and cache
        self.pose_encoder = nn.Sequential(
            nn.Conv2d(pose_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        
        # Initial conv (concatenate pose features)
        self.conv_in = nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1)
        
        # Build encoder - track skip connection channels
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        # Track channels for skip connections
        self.skip_channels = []
        ch = base_channels
        
        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]
            use_attn = level in attention_levels
            
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(ch, out_ch, time_emb_dim, pose_channels, use_film=False, dropout=dropout)
                )
                ch = out_ch
                self.skip_channels.append(ch)
                
                if use_attn:
                    level_attns.append(SelfAttention(ch))
                else:
                    level_attns.append(nn.Identity())
            
            self.encoder.append(level_blocks)
            self.encoder_attns.append(level_attns)
            
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(ch))
        
        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, pose_channels, use_film=False, dropout=dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, pose_channels, use_film=False, dropout=dropout)
        
        # Build decoder - use skip_channels in reverse
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        skip_idx = len(self.skip_channels) - 1
        
        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]
            use_attn = level in attention_levels
            
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels[skip_idx]
                skip_idx -= 1
                level_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_emb_dim, pose_channels, use_film=False, dropout=dropout)
                )
                ch = out_ch
                
                if use_attn:
                    level_attns.append(SelfAttention(ch))
                else:
                    level_attns.append(nn.Identity())
            
            self.decoder.append(level_blocks)
            self.decoder_attns.append(level_attns)
            
            if level > 0:
                self.upsamplers.append(Upsample(ch))
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)
        
        # Zero init output
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, in_channels, H, W) noisy latent
            timestep: (B,) timestep values
            pose: (B, pose_channels, H, W) pose heatmaps
        
        Returns:
            (B, in_channels, H, W) predicted noise
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(timestep, self.base_channels)
        t_emb = self.time_mlp(t_emb)
        
        # Encode pose and concatenate with input
        pose_feat = self.pose_encoder(pose)
        h = torch.cat([x, pose_feat], dim=1)
        h = self.conv_in(h)
        
        # Encoder - collect skip connections
        skips = []
        
        for level, (blocks, attns) in enumerate(zip(self.encoder, self.encoder_attns)):
            for block, attn in zip(blocks, attns):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            
            if level < len(self.downsamplers):
                h = self.downsamplers[level](h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder - use skip connections in reverse
        for level, (blocks, attns) in enumerate(zip(self.decoder, self.decoder_attns)):
            if level > 0:
                h = self.upsamplers[level - 1](h)
            
            for block, attn in zip(blocks, attns):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "channel_mults": self.channel_mults,
            "num_res_blocks": self.num_res_blocks,
            "attention_levels": self.attention_levels,
            "pose_channels": self.pose_channels,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "UNetSimple":
        """Create model from config dict."""
        return cls(**config)