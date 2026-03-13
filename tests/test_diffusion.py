"""
Tests for diffusion model and scheduler.

Run with: pytest tests/test_diffusion.py -v
"""

import pytest
import torch

from diffusion.unet import UNetSimple, ResBlock, SelfAttention, get_timestep_embedding
from diffusion.scheduler import NoiseScheduler


class TestTimestepEmbedding:
    """Tests for timestep embedding."""

    def test_embedding_shape(self):
        """Test timestep embedding output shape."""
        timesteps = torch.tensor([0, 100, 500, 999])
        emb = get_timestep_embedding(timesteps, dim=128)
        assert emb.shape == (4, 128)

    def test_embedding_different_timesteps(self):
        """Test that different timesteps produce different embeddings."""
        t1 = get_timestep_embedding(torch.tensor([0]), dim=64)
        t2 = get_timestep_embedding(torch.tensor([500]), dim=64)
        assert not torch.allclose(t1, t2)

    def test_embedding_odd_dim(self):
        """Test embedding with odd dimension."""
        timesteps = torch.tensor([100])
        emb = get_timestep_embedding(timesteps, dim=65)
        assert emb.shape == (1, 65)


class TestResBlock:
    """Tests for residual block."""

    def test_resblock_same_channels(self):
        """Test ResBlock with same input/output channels."""
        block = ResBlock(64, 64, time_emb_dim=256, pose_channels=35, use_film=False)
        x = torch.randn(2, 64, 20, 15)
        t_emb = torch.randn(2, 256)
        out = block(x, t_emb)
        assert out.shape == x.shape

    def test_resblock_different_channels(self):
        """Test ResBlock with different input/output channels."""
        block = ResBlock(64, 128, time_emb_dim=256, pose_channels=35, use_film=False)
        x = torch.randn(2, 64, 20, 15)
        t_emb = torch.randn(2, 256)
        out = block(x, t_emb)
        assert out.shape == (2, 128, 20, 15)


class TestSelfAttention:
    """Tests for self-attention block."""

    def test_attention_output_shape(self):
        """Test that attention preserves spatial dimensions."""
        attn = SelfAttention(channels=64, num_heads=4)
        x = torch.randn(2, 64, 20, 15)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_residual(self):
        """Test that attention includes residual connection."""
        attn = SelfAttention(channels=64, num_heads=4)
        x = torch.randn(2, 64, 20, 15)
        out = attn(x)
        # Output should be different from input (attention applied)
        # but not completely different (residual connection)
        assert not torch.allclose(out, x)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestUNetSimple:
    """Tests for UNetSimple model."""

    @pytest.fixture
    def small_model(self):
        """Create a small UNet for testing."""
        return UNetSimple(
            in_channels=4,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_levels=(1,),
            pose_channels=35,
        )

    @pytest.fixture
    def default_model(self):
        """Create default UNet configuration."""
        return UNetSimple(
            in_channels=4,
            base_channels=128,
            channel_mults=(1, 2, 4),
            num_res_blocks=2,
            attention_levels=(1, 2),
            pose_channels=35,
        )

    def test_forward_shape_small(self, small_model):
        """Test forward pass output shape with small model."""
        x = torch.randn(2, 4, 80, 60)
        t = torch.randint(0, 1000, (2,))
        pose = torch.randn(2, 35, 80, 60)

        out = small_model(x, t, pose)
        assert out.shape == x.shape

    def test_forward_shape_default(self, default_model):
        """Test forward pass output shape with default model."""
        x = torch.randn(2, 4, 80, 60)
        t = torch.randint(0, 1000, (2,))
        pose = torch.randn(2, 35, 80, 60)

        out = default_model(x, t, pose)
        assert out.shape == x.shape

    def test_color_vae_channels(self):
        """Test UNet with color VAE (8 channels)."""
        model = UNetSimple(
            in_channels=8,
            base_channels=64,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_levels=(1,),
            pose_channels=35,
        )

        x = torch.randn(2, 8, 80, 60)
        t = torch.randint(0, 1000, (2,))
        pose = torch.randn(2, 35, 80, 60)

        out = model(x, t, pose)
        assert out.shape == x.shape

    def test_batch_size_one(self, small_model):
        """Test with batch size of 1."""
        x = torch.randn(1, 4, 80, 60)
        t = torch.randint(0, 1000, (1,))
        pose = torch.randn(1, 35, 80, 60)

        out = small_model(x, t, pose)
        assert out.shape == x.shape

    def test_get_config(self, small_model):
        """Test get_config returns correct values."""
        config = small_model.get_config()
        assert config["in_channels"] == 4
        assert config["base_channels"] == 32
        assert config["channel_mults"] == (1, 2)
        assert config["num_res_blocks"] == 1
        assert config["pose_channels"] == 35

    def test_from_config(self, small_model):
        """Test creating model from config."""
        config = small_model.get_config()
        new_model = UNetSimple.from_config(config)

        # Test that new model has same config
        assert new_model.get_config() == config


class TestNoiseScheduler:
    """Tests for noise scheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create default scheduler."""
        return NoiseScheduler(
            num_timesteps=1000,
            schedule="cosine",
            prediction_type="epsilon",
        )

    def test_add_noise_shape(self, scheduler):
        """Test that add_noise preserves shape."""
        x = torch.randn(2, 4, 80, 60)
        noisy, noise, timesteps = scheduler.add_noise(x)

        assert noisy.shape == x.shape
        assert noise.shape == x.shape
        assert timesteps.shape == (2,)

    def test_add_noise_specific_timestep(self, scheduler):
        """Test adding noise at specific timesteps."""
        x = torch.randn(2, 4, 80, 60)
        t = torch.tensor([0, 999])
        noisy, noise, timesteps = scheduler.add_noise(x, timesteps=t)

        assert torch.equal(timesteps, t)

    def test_add_noise_with_provided_noise(self, scheduler):
        """Test adding specific noise."""
        x = torch.randn(2, 4, 80, 60)
        noise = torch.randn_like(x)
        noisy, returned_noise, _ = scheduler.add_noise(x, noise=noise)

        assert torch.equal(noise, returned_noise)

    def test_predict_start_from_noise(self, scheduler):
        """Test predicting x_0 from noise."""
        x = torch.randn(2, 4, 80, 60)
        noise = torch.randn_like(x)
        t = torch.tensor([500, 500])

        noisy, _, _ = scheduler.add_noise(x, noise=noise, timesteps=t)
        x_pred = scheduler.predict_start_from_noise(noisy, t, noise)

        # Should approximately recover x
        assert torch.allclose(x_pred, x, atol=1e-5)

    def test_get_target_epsilon(self, scheduler):
        """Test target for epsilon prediction."""
        x = torch.randn(2, 4, 80, 60)
        noise = torch.randn_like(x)
        t = torch.tensor([500, 500])

        target = scheduler.get_target(x, noise, t)
        assert torch.equal(target, noise)

    def test_get_target_v_prediction(self):
        """Test target for v prediction."""
        scheduler = NoiseScheduler(prediction_type="v_prediction")
        x = torch.randn(2, 4, 80, 60)
        noise = torch.randn_like(x)
        t = torch.tensor([500, 500])

        target = scheduler.get_target(x, noise, t)
        assert target.shape == x.shape
        assert not torch.equal(target, noise)

    def test_linear_schedule(self):
        """Test linear beta schedule."""
        scheduler = NoiseScheduler(schedule="linear")
        assert len(scheduler.betas) == 1000
        assert scheduler.betas[0] < scheduler.betas[-1]

    def test_scaled_linear_schedule(self):
        """Test scaled linear beta schedule."""
        scheduler = NoiseScheduler(schedule="scaled_linear")
        assert len(scheduler.betas) == 1000

    def test_cosine_schedule(self):
        """Test cosine beta schedule."""
        scheduler = NoiseScheduler(schedule="cosine")
        assert len(scheduler.betas) == 1000

    def test_get_config(self, scheduler):
        """Test get_config returns correct values."""
        config = scheduler.get_config()
        assert config["num_timesteps"] == 1000
        assert config["schedule"] == "cosine"
        assert config["prediction_type"] == "epsilon"

    def test_to_device(self, scheduler):
        """Test moving scheduler to device."""
        scheduler.to(torch.device("cpu"))
        assert scheduler.betas.device.type == "cpu"
        assert scheduler.alphas_cumprod.device.type == "cpu"


class TestDDIMSampling:
    """Tests for DDIM sampling."""

    @pytest.fixture
    def small_model(self):
        """Create small model for sampling tests."""
        return UNetSimple(
            in_channels=4,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_levels=(),  # No attention for speed
            pose_channels=35,
        )

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        return NoiseScheduler(num_timesteps=1000, schedule="cosine")

    def test_ddim_sample_shape(self, small_model, scheduler):
        """Test DDIM sampling output shape."""
        small_model.eval()
        scheduler.to(torch.device("cpu"))

        shape = (1, 4, 80, 60)
        pose = torch.randn(1, 35, 80, 60)

        with torch.no_grad():
            sample = scheduler.ddim_sample(
                small_model, shape, pose,
                device=torch.device("cpu"),
                num_steps=5,
                show_progress=False,
            )

        assert sample.shape == shape

    def test_ddim_deterministic(self, small_model, scheduler):
        """Test DDIM with eta=0 is deterministic."""
        small_model.eval()
        scheduler.to(torch.device("cpu"))

        shape = (1, 4, 80, 60)
        pose = torch.randn(1, 35, 80, 60)

        torch.manual_seed(42)
        with torch.no_grad():
            sample1 = scheduler.ddim_sample(
                small_model, shape, pose,
                device=torch.device("cpu"),
                num_steps=5, eta=0.0,
                show_progress=False,
            )

        torch.manual_seed(42)
        with torch.no_grad():
            sample2 = scheduler.ddim_sample(
                small_model, shape, pose,
                device=torch.device("cpu"),
                num_steps=5, eta=0.0,
                show_progress=False,
            )

        assert torch.allclose(sample1, sample2)


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_training_step(self):
        """Test a single training step."""
        model = UNetSimple(
            in_channels=4,
            base_channels=32,
            channel_mults=(1,),
            num_res_blocks=1,
            attention_levels=(),
            pose_channels=35,
        )
        scheduler = NoiseScheduler(num_timesteps=1000)

        # Simulate training data
        latent = torch.randn(2, 4, 80, 60)
        pose = torch.randn(2, 35, 80, 60)

        # Add noise
        noisy, noise, timesteps = scheduler.add_noise(latent)
        target = scheduler.get_target(latent, noise, timesteps)

        # Forward pass
        pred = model(noisy, timesteps, pose)
        loss = torch.nn.functional.mse_loss(pred, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None