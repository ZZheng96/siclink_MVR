"""Compute OCF scores for paired 64-by-64 character images."""

import numpy as np
from PIL import Image

from src import GLYPH_SIZE as IMAGE_SIZE
from src.ocf.ocf_native import score_image


def _image_batch(images):
    """Accept a single image, a batch, or a batch with one channel."""
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None]
    elif images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[1:] != IMAGE_SIZE or images.dtype.kind not in "buif":
        raise ValueError("Expected real arrays shaped (64, 64), (N, 64, 64), or (N, 1, 64, 64).")
    return images


def score(predictions, targets, *, device="cpu", batch_size=4):
    """Return one OCF score per grayscale prediction and binary reference."""
    predictions = _image_batch(predictions)
    targets = _image_batch(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same number of images.")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if device != "cpu":
        if device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("Use device='cpu', 'cuda', or 'cuda:<index>'.")
        from src.ocf.ocf_cuda import score_batch

    scores = np.empty(len(predictions), dtype=np.float64)
    for start in range(0, len(scores), batch_size):
        stop = start + batch_size
        p = predictions[start:stop]
        t = targets[start:stop]
        if not np.all((p >= 0) & (p <= 1)):
            raise ValueError("Predictions must be finite grayscale values in [0, 1].")
        if not np.all((t == 0) | (t == 1)):
            raise ValueError("Targets must be binary masks containing only 0 and 1.")
        if device == "cpu":
            scores[start:stop] = [score_image(prediction, target) for prediction, target in zip(p, t)]
        else:
            scores[start:stop] = score_batch(p, t, device)
    return scores


def score_png(path, reference, *, device="cpu"):
    """Score a submission-format PNG against its matched uint8 reference glyph."""
    reference = np.asarray(reference)
    if reference.shape != IMAGE_SIZE or reference.dtype != np.uint8:
        raise ValueError("Expected a 64-by-64 uint8 reference glyph.")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != IMAGE_SIZE:
            raise ValueError(f"Expected a 64-by-64 single-channel grayscale PNG: {path}")
        prediction = np.asarray(image, dtype=np.float32) / 255
    return float(score(prediction, reference >= 128, device=device)[0])
