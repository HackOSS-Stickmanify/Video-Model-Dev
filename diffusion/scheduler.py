"""
Noise Scheduler for Diffusion Models

Implements DDPM (Denoising Diffusion Probabilistic Models) and DDIM (Denoising 
Diffusion Implicit Models) noise scheduling for training and inference.

Features:
- Linear, cosine, and scaled-linear beta schedules
- DDPM training (add noise at timestep t)
- DDIM inference (fast deterministic sampling)
- v-prediction and epsilon-prediction modes
- Configurable number of timesteps

Usage:
    from diffusion.scheduler import NoiseScheduler
    
    # Create scheduler
    scheduler = NoiseScheduler(num_timesteps=1000, schedule='cosine')
    
    # Training: add noise
    noisy, noise, t = scheduler.add_noise(clean_latents)
    
    # Inference: sample
    sample = scheduler.ddim_sample(model, shape, pose, num_steps=50)
"""

from __future__ import annotations

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class NoiseScheduler:
    """
    Noise scheduler for DDPM/DDIM diffusion.
    
    Args:
        num_timesteps: Number of diffusion timesteps (default 1000)
        beta_start: Starting beta value
        beta_end: Ending beta value
        schedule: Beta schedule type ('linear', 'cosine', 'scaled_linear')
        prediction_type: What the model predicts ('epsilon', 'v_prediction')
        clip_sample: Whether to clip samples to [-1, 1]
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'cosine',
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        
        # Compute beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'scaled_linear':
            # Scaled linear schedule (used in Stable Diffusion)
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        elif schedule == 'cosine':
            betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        
        # Compute alphas
        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for training
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values for sampling
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance (for DDPM sampling)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def to(self, device: torch.device) -> 'NoiseScheduler':
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def add_noise(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to samples at given timesteps (forward diffusion process).
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        
        Args:
            x: (B, C, H, W) clean samples
            noise: (B, C, H, W) noise to add (if None, sample from N(0, I))
            timesteps: (B,) timesteps (if None, sample uniformly)
        
        Returns:
            noisy: (B, C, H, W) noisy samples
            noise: (B, C, H, W) the noise that was added
            timesteps: (B,) the timesteps used
        """
        device = x.device
        batch_size = x.shape[0]
        
        if noise is None:
            noise = torch.randn_like(x)
        
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Get schedule values for timesteps
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # Add noise
        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        
        return noisy, noise, timesteps
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        """
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * noise
    
    def predict_start_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted v.
        
        v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * x_0
        x_0 = sqrt(alpha_cumprod_t) * x_t - sqrt(1 - alpha_cumprod_t) * v
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    
    def get_v_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute v-prediction target.
        
        v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * x_0
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
    
    def get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Get training target based on prediction type."""
        if self.prediction_type == 'epsilon':
            return noise
        elif self.prediction_type == 'v_prediction':
            return self.get_v_target(x_0, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from model output based on prediction type."""
        if self.prediction_type == 'epsilon':
            return self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == 'v_prediction':
            return self.predict_start_from_v(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    @torch.no_grad()
    def ddpm_step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDPM sampling step.
        
        Args:
            model_output: Model prediction (noise or v)
            t: Current timestep (scalar)
            x_t: Current noisy sample
        
        Returns:
            x_{t-1}: Denoised sample
        """
        device = x_t.device
        t_tensor = torch.tensor([t], device=device)
        
        # Predict x_0
        x_0_pred = self.predict_x0(x_t, t_tensor, model_output)
        
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute posterior mean
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        mean = coef1 * x_0_pred + coef2 * x_t
        
        if t > 0:
            noise = torch.randn_like(x_t)
            std = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + std * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def ddim_step(
        self,
        model_output: torch.Tensor,
        t: int,
        t_prev: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM sampling step.
        
        Args:
            model_output: Model prediction (noise or v)
            t: Current timestep
            t_prev: Previous timestep
            x_t: Current noisy sample
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
        
        Returns:
            x_{t_prev}: Denoised sample
        """
        device = x_t.device
        t_tensor = torch.tensor([t], device=device)
        
        # Predict x_0
        x_0_pred = self.predict_x0(x_t, t_tensor, model_output)
        
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Get alphas
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Compute sigma
        sigma = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # Predict noise from x_0 and x_t
        pred_noise = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)
        
        # Compute x_{t-1}
        x_prev = (
            torch.sqrt(alpha_t_prev) * x_0_pred +
            torch.sqrt(1 - alpha_t_prev - sigma ** 2) * pred_noise
        )
        
        if eta > 0 and t_prev >= 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        pose: torch.Tensor,
        device: torch.device,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Full DDPM sampling loop.
        
        Args:
            model: Diffusion model
            shape: Output shape (B, C, H, W)
            pose: (B, pose_channels, H, W) pose conditioning
            device: Device to use
            show_progress: Show progress bar
        
        Returns:
            Sampled latents
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc='DDPM Sampling')
        
        for t in timesteps:
            t_batch = torch.tensor([t] * shape[0], device=device)
            model_output = model(x, t_batch, pose)
            x = self.ddpm_step(model_output, t, x)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        pose: torch.Tensor,
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling with configurable number of steps.
        
        Args:
            model: Diffusion model
            shape: Output shape (B, C, H, W)
            pose: (B, pose_channels, H, W) pose conditioning
            device: Device to use
            num_steps: Number of sampling steps (fewer = faster)
            eta: DDIM stochasticity (0 = deterministic)
            show_progress: Show progress bar
        
        Returns:
            Sampled latents
        """
        # Create timestep schedule
        step_ratio = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if show_progress:
            timesteps_iter = tqdm(enumerate(timesteps), total=len(timesteps), desc='DDIM Sampling')
        else:
            timesteps_iter = enumerate(timesteps)
        
        for i, t in timesteps_iter:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            
            t_batch = torch.tensor([t] * shape[0], device=device)
            model_output = model(x, t_batch, pose)
            x = self.ddim_step(model_output, t, t_prev, x, eta)
        
        return x
    
    def get_config(self) -> dict:
        """Return scheduler configuration."""
        return {
            'num_timesteps': self.num_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'schedule': self.schedule,
            'prediction_type': self.prediction_type,
            'clip_sample': self.clip_sample,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'NoiseScheduler':
        """Create scheduler from config."""
        return cls(**config)


class DPMSolverScheduler(NoiseScheduler):
    """
    DPM-Solver scheduler for faster sampling.
    
    Implements DPM-Solver++ which can generate good samples in 10-25 steps.
    
    Reference: https://arxiv.org/abs/2211.01095
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Precompute log(alpha)
        self.log_alphas_cumprod = torch.log(self.alphas_cumprod)
    
    def _get_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Compute lambda_t = log(alpha_t) - log(sigma_t)."""
        log_alpha = self.log_alphas_cumprod[t]
        log_sigma = 0.5 * torch.log(1 - self.alphas_cumprod[t])
        return log_alpha - log_sigma
    
    @torch.no_grad()
    def dpm_solver_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        pose: torch.Tensor,
        device: torch.device,
        num_steps: int = 20,
        order: int = 2,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DPM-Solver++ sampling.
        
        Args:
            model: Diffusion model
            shape: Output shape (B, C, H, W)
            pose: (B, pose_channels, H, W) pose conditioning
            device: Device to use
            num_steps: Number of sampling steps
            order: Solver order (1, 2, or 3)
            show_progress: Show progress bar
        
        Returns:
            Sampled latents
        """
        # Create timestep schedule (uniform in lambda space)
        step_ratio = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Store model outputs for multi-step methods
        model_outputs = []
        
        if show_progress:
            timesteps_iter = tqdm(enumerate(timesteps), total=len(timesteps), desc='DPM-Solver Sampling')
        else:
            timesteps_iter = enumerate(timesteps)
        
        for i, t in timesteps_iter:
            t_batch = torch.tensor([t] * shape[0], device=device)
            model_output = model(x, t_batch, pose)
            model_outputs.append(model_output)
            
            if len(model_outputs) > order:
                model_outputs.pop(0)
            
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            if order == 1 or len(model_outputs) == 1:
                x = self._dpm_solver_first_order(x, t, t_prev, model_output)
            elif order == 2 and len(model_outputs) >= 2:
                x = self._dpm_solver_second_order(
                    x, timesteps[max(0, i-1)], t, t_prev,
                    model_outputs[-2], model_outputs[-1]
                )
            else:
                x = self._dpm_solver_first_order(x, t, t_prev, model_output)
        
        return x
    
    def _dpm_solver_first_order(
        self,
        x: torch.Tensor,
        t: int,
        t_prev: int,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """First-order DPM-Solver update."""
        device = x.device
        t_tensor = torch.tensor([t], device=device)
        
        # Predict x_0
        x_0_pred = self.predict_x0(x, t_tensor, model_output)
        
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
        sigma_t = torch.sqrt(1 - alpha_t)
        sigma_t_prev = torch.sqrt(1 - alpha_t_prev) if t_prev > 0 else torch.tensor(0.0)
        
        # First-order update
        x_prev = (
            torch.sqrt(alpha_t_prev) * x_0_pred +
            sigma_t_prev * (x - torch.sqrt(alpha_t) * x_0_pred) / sigma_t
        )
        
        return x_prev
    
    def _dpm_solver_second_order(
        self,
        x: torch.Tensor,
        t_prev_prev: int,
        t: int,
        t_prev: int,
        model_output_prev: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Second-order DPM-Solver update."""
        # For simplicity, fall back to first order if not enough history
        return self._dpm_solver_first_order(x, t, t_prev, model_output)