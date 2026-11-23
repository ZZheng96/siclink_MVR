"""Batch the native 64-by-64 OCF calculation on a CUDA device."""

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import cosdg, sindg
import torch
from torch.nn import functional as F

from src import GLYPH_SIZE
from src.ocf.ocf_native import ANGLES, DIRECTIONS, OFFSETS, gaussian_spectrum


@lru_cache(maxsize=32)
def kernel_spectrum(height, width, sigma, dy, dx, device):
    """Transfer the two-dimensional separable Gaussian spectrum once per shape."""
    ky = gaussian_spectrum(sigma, dy, height, False)
    kx = gaussian_spectrum(sigma, dx, width, True)
    return torch.as_tensor(ky[:, None] * kx[None, :], device=device)


def gaussian_fields(image, sigma, orders):
    """Convolve with the same zero-padded finite kernels as the CPU scorer."""
    height, width = image.shape[-2:]
    radius = int(4 * sigma + 0.5)
    shape = (next_fast_len(height + 2 * radius, real=True), next_fast_len(width + 2 * radius, real=True))
    frequency = torch.fft.rfft2(image, s=shape)
    result = []
    for dy, dx in orders:
        kernel = kernel_spectrum(*shape, sigma, dy, dx, image.device)
        filtered = torch.fft.irfft2(frequency * kernel, s=shape)
        result.append(filtered[..., radius:radius + height, radius:radius + width])
    return result


def blur(image, sigma):
    """Apply the OCF zero-padded Gaussian smoothing."""
    return gaussian_fields(image, sigma, ((0, 0),))[0]


def unit_mass(image):
    """Smooth each image and normalize it to unit mass."""
    smoothed = blur(image, 2)
    mass = smoothed.sum((-2, -1), keepdim=True, dtype=torch.float64)
    return smoothed / mass.clamp_min(1e-12).float()


@lru_cache(maxsize=4)
def grid(size, device):
    """Build a reusable coordinate grid in native pixel units."""
    values = torch.arange(size, dtype=torch.float32, device=device)
    return torch.stack(torch.meshgrid(values, values, indexing="ij"))[None]


@lru_cache(maxsize=12)
def rotation_index(size, angle, device):
    """Compute nearest-neighbor coordinates for the stock SciPy rotation."""
    c, s = cosdg(angle), sindg(angle)
    rotation = np.array([[c, s], [-s, c]])
    center = np.full(2, (size - 1) / 2)
    offset = center - rotation @ center
    points = np.indices((size, size), dtype=np.float64).reshape(2, -1)
    coords = rotation @ points + offset[:, None]
    valid = ((coords >= 0) & (coords <= size - 1)).all(axis=0)
    indices = np.floor(coords + 0.5).astype(np.int64).clip(0, size - 1)
    flat = indices[0] * size + indices[1]
    return torch.as_tensor(flat, device=device), torch.as_tensor(valid, device=device)


def nearest(image, coords):
    """Sample a binary image with the CPU's nearest-neighbor and zero-border rule."""
    size = image.shape[-1]
    valid = ((coords >= 0) & (coords <= size - 1)).all(dim=1)
    indices = torch.floor(coords + 0.5).long().clamp_(0, size - 1)
    flat = indices[:, 0] * size + indices[:, 1]
    pixels = image.flatten(1).gather(1, flat.flatten(1)).reshape_as(image)
    return pixels * valid


def bilinear(image, coords):
    """Sample the smooth fields with zero outside the original coordinate bounds."""
    size = image.shape[-1]
    valid = ((coords >= 0) & (coords <= size - 1)).all(dim=1, keepdim=True)
    normalized = (coords.flip(1).permute(0, 2, 3, 1) * (2 / (size - 1))) - 1
    result = F.grid_sample(image, normalized, mode="bilinear", padding_mode="zeros", align_corners=True)
    return result * valid


def rigid_candidates(predictions, targets):
    """Find both stock pose initializations independently for each image."""
    predictions = F.pad(predictions, (8,) * 4)
    targets = F.pad(targets, (8,) * 4)
    fixed = unit_mass(predictions)
    extent = predictions.shape[-1]
    pad = 2
    batch_size = len(predictions)
    rotations = []
    best_error = torch.full((batch_size,), torch.inf, dtype=torch.float64, device=fixed.device)
    best = torch.zeros((batch_size, 3), dtype=torch.long, device=fixed.device)
    for angle_index, angle in enumerate(ANGLES):
        if angle == 0:
            rotated = targets
        else:
            indices, valid = rotation_index(extent, angle, fixed.device)
            rotated = (targets.flatten(1)[:, indices] * valid).reshape_as(targets)
        rotations.append(rotated)
        moving = F.pad(unit_mass(rotated), (pad,) * 4)
        for dy, dx in OFFSETS:
            y, x = pad - dy, pad - dx
            shifted = moving[:, y:y + extent, x:x + extent]
            error = (fixed - shifted).abs().sum((-1, -2), dtype=torch.float64)
            improved = error < best_error - 1e-10
            best_error = torch.where(improved, error, best_error)
            best[:, 0] = torch.where(improved, angle_index, best[:, 0])
            best[:, 1] = torch.where(improved, dy, best[:, 1])
            best[:, 2] = torch.where(improved, dx, best[:, 2])
        if angle == 0:
            simple = best.clone()
    target_options = torch.stack(rotations, dim=1)
    batch = torch.arange(batch_size, device=fixed.device)
    native_grid = grid(extent, fixed.device)
    rigid_target = target_options[batch, best[:, 0]]
    rigid = nearest(rigid_target, native_grid - best[:, 1:, None, None])
    translation = nearest(targets, native_grid - simple[:, 1:, None, None])
    return predictions, fixed, rigid, translation, (best != simple).any(dim=1)


def bound_field(flow):
    """Contract each field to the stock displacement and Jacobian limits."""
    amplitude = flow.abs().amax((1, 2, 3), keepdim=True)
    flow = flow * (2 / amplitude.clamp_min(1e-12)).clamp_max(1)
    dy, dx = torch.gradient(flow, dim=(-2, -1))
    norm_squared = dy[:, 0].square() + dx[:, 0].square() + dy[:, 1].square() + dx[:, 1].square()
    norm = norm_squared.amax((-2, -1)).sqrt()
    return flow * (0.45 / norm.clamp_min(1e-12)).clamp_max(1)[:, None, None, None]


def residual_energy(fixed, moving):
    """Measure squared residuals per image."""
    return (fixed - moving).square().sum((-2, -1))


def smooth_reference(fixed, target):
    """Fit the same bounded smooth flow for 60 iterations on the native grid."""
    native_grid = grid(target.shape[-1], target.device)
    flow = torch.zeros((len(target), 2, *target.shape[-2:]), dtype=target.dtype, device=target.device)
    moving = unit_mass(target)
    dy, dx = torch.gradient(moving, dim=(-2, -1))
    fields = torch.stack((moving, dy, dx), dim=1)
    best_error = residual_energy(fixed, moving)
    best_flow = flow
    for _ in range(60):
        warped = bilinear(fields, native_grid + flow)
        grad = warped[:, 1:]
        residual = fixed - warped[:, 0]
        denominator = grad[:, 0].square() + grad[:, 1].square() + residual.square() + 1e-12
        delta = residual[:, None] * grad / denominator[:, None]
        flow = bound_field(blur(flow + 0.8 * blur(delta, 4), 1))
        fitted = bilinear(fields[:, :1], native_grid + flow)[:, 0]
        error = residual_energy(fixed, fitted)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_flow = torch.where(improved[:, None, None, None], flow, best_flow)
    return nearest(target, native_grid + best_flow)


def directional_fields(image, sigma):
    """Calculate the stock four signed second derivative directions."""
    gxx, gyy, gxy = gaussian_fields(image, sigma, ((0, 2), (2, 0), (1, 1)))
    return torch.stack([
        np.cos(theta)**2 * gxx + np.sin(theta)**2 * gyy + 2 * np.sin(theta) * np.cos(theta) * gxy
        for theta in DIRECTIONS
    ], dim=1)


def sum_directions(fields):
    """Accumulate the four directions in the same order as the CPU path."""
    return fields[:, 0] + fields[:, 1] + fields[:, 2] + fields[:, 3]


def matched_energy(predictions, targets, sigma):
    """Find the best local agreement and accumulate overlapping edge energy."""
    p = directional_fields(predictions, sigma)
    t = directional_fields(targets, sigma)
    pool = 2
    a = blur(sum_directions(p.square()), pool).clamp_min_(0)
    b = blur(sum_directions(t.square()), pool).clamp_min_(0)
    height, width = a.shape[-2:]
    wide = F.pad(t, (1,) * 4)
    wide_b = F.pad(b, (1,) * 4)
    agreement = torch.zeros_like(a)
    for dy, dx in product((-1, 0, 1), repeat=2):
        y, x = 1 - dy, 1 - dx
        shifted = wide[..., y:y + height, x:x + width]
        shifted_b = wide_b[..., y:y + height, x:x + width]
        cross = blur(sum_directions(p * shifted), pool)
        den = a * shifted_b
        cosine2 = torch.where(den > 0, cross.clamp_min_(0).square() / den, 0)
        agreement = torch.maximum(agreement, cosine2.clamp_max_(1))
    shared = torch.minimum(a, b) * agreement
    total = a + b - shared
    return shared.sum((-2, -1), dtype=torch.float64), total.sum((-2, -1), dtype=torch.float64)


@torch.inference_mode()
def score_batch(predictions, targets, device):
    """Score native 64-pixel image pairs on CUDA, retaining the stock empty cases."""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if predictions.shape != targets.shape or predictions.ndim != 3 or predictions.shape[1:] != GLYPH_SIZE:
        raise ValueError("Expected matched arrays of native 64-by-64 image pairs.")
    p_present = predictions.any(axis=(1, 2))
    t_present = targets.any(axis=(1, 2))
    equal = np.all(predictions == targets, axis=(1, 2))
    values = np.where(equal, 100.0, 0.0)
    active = p_present & t_present & ~equal
    if not active.any():
        return values
    prediction = torch.as_tensor(predictions[active], device=device)
    target = torch.as_tensor(targets[active], device=device)
    prediction, fixed, rigid, translation, use_translation = rigid_candidates(prediction, target)
    fitted = smooth_reference(fixed, rigid)
    selected = torch.where(use_translation)[0]
    if len(selected):
        alternative = smooth_reference(fixed[selected], translation[selected])
        original_error = residual_energy(fixed[selected], unit_mass(fitted[selected]))
        alternative_error = residual_energy(fixed[selected], unit_mass(alternative))
        choose = alternative_error < original_error - 1e-12
        fitted[selected] = torch.where(choose[:, None, None], alternative, fitted[selected])

    prediction = F.pad(prediction, (24,) * 4)
    target = F.pad(fitted, (24,) * 4)
    energies = [matched_energy(prediction, target, sigma) for sigma in (1, 2, 4)]
    shared = sum(a for a, _ in energies)
    total = sum(b for _, b in energies)
    scores = torch.where(total > 0, 100 * shared / total, 100)
    values[active] = scores.cpu().numpy()
    return values
