"""Evaluate OCF on the native 64-by-64 sampling grid."""

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.fft import fft, rfft
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, rotate

from src import GLYPH_SIZE


ANGLES = tuple(sorted(range(-10, 11, 2), key=lambda angle: (abs(angle), angle)))
OFFSETS = tuple(sorted(product(range(-2, 3), repeat=2), key=lambda d: (d[0]**2 + d[1]**2, d)))
DIRECTIONS = tuple(np.arange(4) * np.pi / 4)


@lru_cache(maxsize=48)
def gaussian_spectrum(sigma, order, n, real):
    """Compute the frequency response of SciPy's truncated 1D Gaussian kernel."""
    radius = int(4 * sigma + 0.5)
    impulse = np.zeros(2 * radius + 1, dtype=np.float32)
    impulse[radius] = 1
    kernel = gaussian_filter1d(impulse, sigma, order=order, mode="constant")
    return rfft(kernel, n) if real else fft(kernel, n)


def gaussian_fields(image, sigma, orders):
    """Return zero-padded Gaussian derivatives along the two image axes."""
    leading = (0,) * (image.ndim - 2)
    return [
        gaussian_filter(image, leading + (sigma, sigma), order=leading + order, mode="constant")
        for order in orders
    ]


def blur(image, sigma):
    """Apply the protocol's zero-padded Gaussian smoothing."""
    return gaussian_fields(image, sigma, ((0, 0),))[0]


def unit_mass(image):
    """Smooth an image and normalize it to unit mass."""
    smoothed = blur(image, 2)
    return smoothed / max(float(smoothed.sum(dtype=np.float64)), 1e-12)


def rigid_candidates(prediction, target):
    """Find the best rigid pose and the best translation-only starting pose."""
    prediction = np.pad(prediction, 8)
    target = np.pad(target, 8)
    fixed = unit_mass(prediction)
    extent = prediction.shape[0]
    pad = 2
    best_error = np.inf
    best = simple = None
    for angle in ANGLES:
        rotated = target if angle == 0 else rotate(
            target, angle, reshape=False, order=0, mode="constant", prefilter=False
        )
        moving = np.pad(unit_mass(rotated), pad)
        for dy, dx in OFFSETS:
            y, x = pad - dy, pad - dx
            error = np.abs(fixed - moving[y:y + extent, x:x + extent]).sum(dtype=np.float64)
            if error < best_error - 1e-10:
                best_error = error
                best = (rotated, angle, dy, dx)
        if angle == 0:
            simple = best

    def translated(candidate):
        """Apply the candidate's translation to its rotated glyph."""
        rotated, _, dy, dx = candidate
        y, x = pad - dy, pad - dx
        return np.pad(rotated, pad)[y:y + extent, x:x + extent].copy()

    candidates = [translated(best)]
    if best[1:] != simple[1:]:
        candidates.append(translated(simple))
    return prediction, fixed, candidates


def bounded_field(flow):
    """Contract the entire displacement field to the radius and gradient limits."""
    flow *= min(1.0, 2 / max(float(np.abs(flow).max()), 1e-12))
    derivatives = [g for field in flow for g in np.gradient(field)]
    maximum = np.sqrt(sum(g * g for g in derivatives)).max()
    flow *= min(1.0, 0.45 / max(float(maximum), 1e-12))
    return flow


def smooth_reference(fixed, target):
    """Apply the bounded 60-step OCF smooth-field fit on the native grid."""
    coords = np.indices(target.shape, dtype=np.float32)
    flow = np.zeros_like(coords)
    moving = unit_mass(target)
    gradients = np.gradient(moving)
    best_error = float(np.square(fixed - moving).sum())
    best_flow = flow.copy()
    for _ in range(60):
        location = coords + flow
        warped = map_coordinates(moving, location, order=1, mode="constant", prefilter=False)
        grad = np.stack([
            map_coordinates(g, location, order=1, mode="constant", prefilter=False)
            for g in gradients
        ])
        residual = fixed - warped
        delta = residual * grad / (np.square(grad).sum(axis=0) + residual**2 + 1e-12)
        flow = bounded_field(blur(flow + 0.8 * blur(delta, 4), 1))
        fitted = map_coordinates(moving, coords + flow, order=1, mode="constant", prefilter=False)
        error = float(np.square(fixed - fitted).sum())
        if error < best_error:
            best_error = error
            best_flow = flow.copy()
    return map_coordinates(target, coords + best_flow, order=0, mode="constant", prefilter=False)


def directional_fields(image, sigma):
    """Compute the four signed second derivative fields at one OCF scale."""
    gxx, gyy, gxy = gaussian_fields(image, sigma, ((0, 2), (2, 0), (1, 1)))
    return [
        np.cos(angle)**2 * gxx + np.sin(angle)**2 * gyy + 2 * np.sin(angle) * np.cos(angle) * gxy
        for angle in DIRECTIONS
    ]


def matched_energy(prediction, target, sigma):
    """Compare local signed patches and cap shared energy at the original site."""
    p = directional_fields(prediction, sigma)
    t = directional_fields(target, sigma)
    pool = 2
    a = np.maximum(blur(sum(f * f for f in p), pool), 0).astype(np.float64)
    b = np.maximum(blur(sum(f * f for f in t), pool), 0).astype(np.float64)
    height, width = a.shape
    wide = [np.pad(f, 1) for f in t]
    wide_b = np.pad(b, 1)
    agreement = np.zeros_like(a)
    for dy, dx in product((-1, 0, 1), repeat=2):
        y, x = 1 - dy, 1 - dx
        shifted = [f[y:y + height, x:x + width] for f in wide]
        shifted_b = wide_b[y:y + height, x:x + width]
        cross = blur(sum(f * g for f, g in zip(p, shifted)), pool).astype(np.float64)
        den = a * shifted_b
        cosine2 = np.divide(np.maximum(cross, 0)**2, den, out=np.zeros_like(cross), where=den > 0)
        np.maximum(agreement, np.minimum(cosine2, 1), out=agreement)
    shared = np.minimum(a, b) * agreement
    return shared.sum(), (a + b - shared).sum()


def score_image(prediction, target):
    """Score one matched grayscale prediction and binary reference at native resolution."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != GLYPH_SIZE or target.shape != GLYPH_SIZE:
        raise ValueError("Expected a matched 64-by-64 image pair.")
    if not prediction.any() or not target.any():
        return 100.0 if not prediction.any() and not target.any() else 0.0
    if np.array_equal(prediction, target):
        return 100.0
    prediction, fixed, candidates = rigid_candidates(prediction, target)
    best_error = np.inf
    for initial in candidates:
        fitted = smooth_reference(fixed, initial)
        error = float(np.square(fixed - unit_mass(fitted)).sum())
        if error < best_error - 1e-12:
            best_error = error
            aligned = fitted

    prediction = np.pad(prediction, 24)
    target = np.pad(aligned, 24)
    energies = [matched_energy(prediction, target, sigma) for sigma in (1, 2, 4)]
    shared, total = np.sum(energies, axis=0)
    return float(100 * shared / total) if total > 0 else 100.0
