"""
Tests for VAE model and training utilities.

Run with: pytest tests/test_vae.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.model import (
    VAE,
    VAELoss,
    EMA,
    ResBlock,
    Encoder,
    Decoder,
    compute_vae_loss,
)


class TestResBlock:
    """Tests for residual block."""
    
    def test_same_channels(self):
        """Test ResBlock with same input/output channels."""
        block = ResBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape
    
    def test_different_channels(self):
        """Test ResBlock with different input/output channels."""
        block = ResBlock(32, 64)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestEncoder:
    """Tests for VAE encoder."""
    
    def test_output_shape(self):
        """Test encoder outputs correct latent shape."""
        encoder = Encoder(in_channels=1, base_channels=32, z_channels=4)
        x = torch.randn(2, 1, 640, 480)
        mean, logvar = encoder(x)
        
        # 640/8 = 80, 480/8 = 60
        assert mean.shape == (2, 4, 80, 60)
        assert logvar.shape == (2, 4, 80, 60)
    
    def test_custom_channels(self):
        """Test encoder with custom channel configuration."""
        encoder = Encoder(
            in_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4, 8),
            z_channels=8,
        )
        x = torch.randn(2, 1, 256, 256)
        mean, logvar = encoder(x)
        
        # 256/8 = 32 (due to 3 downsamples + 1 final)
        assert mean.shape[1] == 8  # z_channels


class TestDecoder:
    """Tests for VAE decoder."""
    
    def test_output_shape(self):
        """Test decoder outputs correct image shape."""
        decoder = Decoder(out_channels=1, base_channels=32, z_channels=4)
        z = torch.randn(2, 4, 80, 60)
        out = decoder(z)
        
        # 80*8 = 640, 60*8 = 480
        assert out.shape == (2, 1, 640, 480)


class TestVAE:
    """Tests for full VAE model."""
    
    @pytest.fixture
    def model(self):
        """Create a small VAE for testing."""
        return VAE(
            in_channels=1,
            base_channels=16,  # Smaller for faster tests
            channel_mults=(1, 2, 4),
            z_channels=4,
            num_res_blocks=1,
        )
    
    def test_forward_pass(self, model):
        """Test full forward pass."""
        x = torch.randn(2, 1, 640, 480)
        recon, mean, logvar = model(x)
        
        assert recon.shape == x.shape
        assert mean.shape == (2, 4, 80, 60)
        assert logvar.shape == (2, 4, 80, 60)
    
    def test_encode(self, model):
        """Test encoding to latent space."""
        x = torch.randn(2, 1, 640, 480)
        mean, logvar = model.encode(x)
        
        assert mean.shape == (2, 4, 80, 60)
        assert logvar.shape == (2, 4, 80, 60)
    
    def test_decode(self, model):
        """Test decoding from latent space."""
        z = torch.randn(2, 4, 80, 60)
        out = model.decode(z)
        
        assert out.shape == (2, 1, 640, 480)
    
    def test_reparameterize(self, model):
        """Test reparameterization trick."""
        mean = torch.zeros(2, 4, 80, 60)
        logvar = torch.zeros(2, 4, 80, 60)
        
        z = model.reparameterize(mean, logvar)
        
        assert z.shape == mean.shape
        # With zero mean and logvar=0 (std=1), values should be roughly standard normal
        assert z.std().item() > 0.5  # Should have some variance
    
    def test_get_latent_deterministic(self, model):
        """Test deterministic latent extraction."""
        x = torch.randn(2, 1, 640, 480)
        
        z1 = model.get_latent(x, deterministic=True)
        z2 = model.get_latent(x, deterministic=True)
        
        # Should be identical when deterministic
        assert torch.allclose(z1, z2)
    
    def test_get_latent_stochastic(self, model):
        """Test stochastic latent extraction."""
        x = torch.randn(2, 1, 640, 480)
        
        torch.manual_seed(42)
        z1 = model.get_latent(x, deterministic=False)
        torch.manual_seed(123)
        z2 = model.get_latent(x, deterministic=False)
        
        # Should be different when stochastic (different seeds)
        assert not torch.allclose(z1, z2)
    
    def test_sample(self, model):
        """Test sampling from prior."""
        samples = model.sample(batch_size=4)
        assert samples.shape == (4, 1, 640, 480)
    
    def test_get_config(self, model):
        """Test config export."""
        config = model.get_config()
        
        assert config["in_channels"] == 1
        assert config["base_channels"] == 16
        assert config["z_channels"] == 4
    
    def test_from_config(self, model):
        """Test model creation from config."""
        config = model.get_config()
        model2 = VAE.from_config(config)
        
        # Check same architecture
        assert model2.z_channels == model.z_channels
        assert model2.base_channels == model.base_channels
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, model):
        """Test forward pass on CUDA."""
        model = model.cuda()
        x = torch.randn(2, 1, 640, 480).cuda()
        
        recon, mean, logvar = model(x)
        
        assert recon.device.type == "cuda"
        assert mean.device.type == "cuda"


class TestVAELoss:
    """Tests for VAE loss function."""
    
    def test_basic_loss(self):
        """Test basic loss computation."""
        loss_fn = VAELoss(kl_weight=0.01, use_lpips=False)
        
        recon = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        mean = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        
        loss, loss_dict = loss_fn(recon, target, mean, logvar)
        
        assert loss.ndim == 0  # Scalar
        assert "total" in loss_dict
        assert "recon" in loss_dict
        assert "kl" in loss_dict
    
    def test_kl_weight(self):
        """Test that KL weight affects loss."""
        recon = torch.randn(2, 1, 64, 64)
        target = recon.clone()  # Perfect reconstruction
        mean = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        
        loss_fn_low = VAELoss(kl_weight=0.001, use_lpips=False)
        loss_fn_high = VAELoss(kl_weight=1.0, use_lpips=False)
        
        loss_low, _ = loss_fn_low(recon, target, mean, logvar)
        loss_high, _ = loss_fn_high(recon, target, mean, logvar)
        
        # Higher KL weight should give higher loss (with same KL term)
        assert loss_high > loss_low
    
    def test_compute_vae_loss_function(self):
        """Test standalone loss function."""
        recon = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        mean = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        
        loss, loss_dict = compute_vae_loss(recon, target, mean, logvar)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    @pytest.fixture
    def model_and_ema(self):
        """Create a simple model and EMA wrapper."""
        model = torch.nn.Linear(10, 10)
        ema = EMA(model, decay=0.9, update_after_step=0, update_every=1)
        return model, ema
    
    def test_initialization(self, model_and_ema):
        """Test EMA initializes with model parameters."""
        model, ema = model_and_ema
        
        # Shadow params should match model params initially
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ema.shadow_params[name], param)
    
    def test_update(self, model_and_ema):
        """Test EMA update moves towards model parameters."""
        model, ema = model_and_ema
        
        # Get original shadow params
        original_shadow = {k: v.clone() for k, v in ema.shadow_params.items()}
        
        # Update model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))
        
        # Update EMA
        ema.update()
        
        # Shadow params should have moved towards model params
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Shadow should be between original and current
                # (0.9 * original + 0.1 * current)
                expected = 0.9 * original_shadow[name] + 0.1 * param
                assert torch.allclose(ema.shadow_params[name], expected)
    
    def test_average_parameters_context(self, model_and_ema):
        """Test average_parameters context manager."""
        model, ema = model_and_ema
        
        # Store original model params
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))
        
        # Update EMA
        ema.update()
        
        # Inside context, model should have EMA params
        with ema.average_parameters():
            for name, param in model.named_parameters():
                assert torch.allclose(param, ema.shadow_params[name])
        
        # Outside context, model should be restored
        # (to the modified params, not original)
        for name, param in model.named_parameters():
            assert not torch.allclose(param, original_params[name])
    
    def test_state_dict(self, model_and_ema):
        """Test EMA state saving and loading."""
        model, ema = model_and_ema
        
        # Update a few times
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema.update()
        
        # Save state
        state = ema.state_dict()
        
        # Create new EMA and load state
        ema2 = EMA(model, decay=0.9)
        ema2.load_state_dict(state)
        
        # Should match
        assert ema2.step == ema.step
        for name in ema.shadow_params:
            assert torch.allclose(ema2.shadow_params[name], ema.shadow_params[name])


class TestModelGradients:
    """Tests for gradient flow."""
    
    def test_encoder_gradients(self):
        """Test gradients flow through encoder."""
        encoder = Encoder(in_channels=1, base_channels=16, z_channels=4)
        x = torch.randn(1, 1, 128, 128, requires_grad=True)
        
        mean, logvar = encoder(x)
        loss = mean.sum() + logvar.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_decoder_gradients(self):
        """Test gradients flow through decoder."""
        decoder = Decoder(out_channels=1, base_channels=16, z_channels=4)
        z = torch.randn(1, 4, 16, 16, requires_grad=True)
        
        out = decoder(z)
        loss = out.sum()
        loss.backward()
        
        assert z.grad is not None
        assert z.grad.abs().sum() > 0
    
    def test_full_vae_gradients(self):
        """Test gradients flow through full VAE."""
        model = VAE(base_channels=16, num_res_blocks=1)
        x = torch.randn(1, 1, 128, 128, requires_grad=True)
        
        recon, mean, logvar = model(x)
        loss, _ = compute_vae_loss(recon, x, mean, logvar)
        loss.backward()
        
        # Check model gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])