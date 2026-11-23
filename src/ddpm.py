"""Train and sample a spike-conditioned DDPM on the native image grid."""

import math

import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.nn import functional as F

from src import GLYPH_SIZE as MODEL_IMAGE_SIZE

DIFFUSION_STEPS = 64


def time_embedding(timesteps):
    """Encode the diffusion timestep with 64 sinusoidal features."""
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(32, device=timesteps.device) / 32
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=1)


class ResidualBlock(nn.Module):
    """Apply timestep and spike conditioning to a convolutional block."""

    def __init__(self, inputs, outputs):
        """Build two convolutions and an embedding that modulates their output."""
        super().__init__()
        self.norm1 = nn.GroupNorm(8, inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.modulation = nn.Linear(128, 2 * outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.shortcut = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, image, embedding):
        """Predict a residual modulated by the condition and timestep."""
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, bias = self.modulation(F.silu(embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return self.shortcut(image) + self.conv2(F.silu(hidden))


class SpikeDenoiser(nn.Module):
    """Predict the DDPM velocity image from spikes and a noisy 64×64 image."""

    def __init__(self, n_features):
        """Build a jointly trained spike encoder and a compact multiscale denoiser."""
        super().__init__()
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.spike = nn.Sequential(
            nn.Linear(n_features, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 128),
        )
        self.null_condition = nn.Parameter(torch.zeros(128))
        self.input = nn.Conv2d(1, 16, 3, padding=1)
        self.encoder1 = ResidualBlock(16, 16)
        self.down1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder2 = ResidualBlock(32, 32)
        self.down2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.middle1 = ResidualBlock(64, 64)
        self.middle2 = ResidualBlock(64, 64)
        self.decoder2 = ResidualBlock(64 + 32, 32)
        self.decoder1 = ResidualBlock(32 + 16, 16)
        self.output = nn.Sequential(
            nn.GroupNorm(8, 16), nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, image, timesteps, counts, drop=None):
        """Return one native-size velocity image per spike condition."""
        spikes = self.spike(torch.log1p(counts))
        if drop is not None:
            spikes = torch.where(drop[:, None], self.null_condition[None], spikes)
        embedding = self.time(time_embedding(timesteps)) + spikes

        high = self.encoder1(self.input(image), embedding)
        low = self.encoder2(self.down1(high), embedding)
        hidden = self.middle1(self.down2(low), embedding)
        hidden = self.middle2(hidden, embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.decoder2(torch.cat((hidden, low), dim=1), embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.decoder1(torch.cat((hidden, high), dim=1), embedding)
        return self.output(hidden)


def build_model(n_features):
    """Build a denoiser for one mouse's spike feature dimension."""
    return SpikeDenoiser(n_features)


def diffusion_scheduler():
    """Configure the library DDPM scheduler for cosine noise and v-prediction."""
    return DDPMScheduler(
        num_train_timesteps=DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        prediction_type="v_prediction",
        clip_sample=True,
    )


def training_loss(model, scheduler, images, counts, generator=None):
    """Measure the Min-SNR weighted velocity error on images scaled to [-1, 1]."""
    device = images.device
    batch_size = len(images)
    timesteps = torch.randint(
        scheduler.config.num_train_timesteps, (batch_size,), device=device, generator=generator,
    )
    noise = torch.randn(images.shape, dtype=images.dtype, device=device, generator=generator)
    noisy = scheduler.add_noise(images, noise, timesteps)
    target = scheduler.get_velocity(images, noise, timesteps)
    drop = None
    if model.training:
        # Leave 10% of examples unconditioned for classifier-free guidance.
        drop = torch.rand(batch_size, device=device, generator=generator) < 0.1
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
        velocity = model(noisy, timesteps, counts, drop)
    alpha = scheduler.alphas_cumprod[timesteps]
    snr = alpha / (1 - alpha)
    weight = snr.clamp(max=5) / (snr + 1)
    error = (velocity.float() - target).square().flatten(1).mean(1)
    return (weight * error).mean()


@torch.inference_mode()
def sample_images(model, scheduler, counts, generator):
    """Denoise native-size Gaussian noise into grayscale images in [0, 1]."""
    device = counts.device
    batch_size = len(counts)
    image = torch.randn((batch_size, 1, *MODEL_IMAGE_SIZE), device=device, generator=generator)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    batch_counts = torch.cat((counts, counts))
    drop = torch.arange(2 * batch_size, device=device) >= batch_size
    for timestep in scheduler.timesteps.tolist():
        timesteps = torch.full((2 * batch_size,), timestep, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=counts.is_cuda):
            velocity = model(torch.cat((image, image)), timesteps, batch_counts, drop)
        conditional, unconditional = velocity.float().chunk(2)
        guided = unconditional + 1.5 * (conditional - unconditional)
        image = scheduler.step(guided, timestep, image, generator=generator).prev_sample
    return (image + 1).mul(0.5).clamp(0, 1)
