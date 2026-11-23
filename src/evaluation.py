"""PNG-parity evaluation using the official Oriented Contour Fidelity metric."""

import numpy as np

from src.ocf.score import score as official_score


def _unit_images(images, name):
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3 or images.shape[1:] != (64, 64):
        raise ValueError(f"{name} must have shape (N, 64, 64).")
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255
    images = images.astype(np.float32)
    if not np.isfinite(images).all() or images.min(initial=0) < 0 or images.max(initial=0) > 1:
        raise ValueError(f"Floating-point {name} must be finite and in [0, 1].")
    return images


def png_quantize(images):
    """Match the uint8 PNG round trip used by Kaggle."""
    images = _unit_images(images, "predictions")
    return np.rint(np.clip(images, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255


def binary_targets(images):
    images = _unit_images(images, "targets")
    scaled = np.rint(images * 255).astype(np.uint8)
    if not np.all((scaled == 0) | (scaled == 255)):
        raise ValueError("Evaluation targets must be binary {0,1} or {0,255}.")
    return scaled == 255


def score_predictions(predictions, targets, *, threshold=None, device="cpu", batch_size=16):
    """Return per-image official OCF after optional binarisation and PNG quantisation."""
    predictions = np.asarray(predictions)
    if predictions.dtype != np.uint8:
        predictions = np.clip(predictions, 0, 1)
    predictions = _unit_images(predictions, "predictions")
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        predictions = predictions >= threshold
    predictions = png_quantize(predictions)
    targets = binary_targets(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")
    return official_score(predictions, targets, device=device, batch_size=batch_size)


def summarize_scores(scores, group_ids=None):
    scores = np.asarray(scores, dtype=np.float64)
    result = {"n": len(scores), "ocf_mean": float(scores.mean()), "ocf_std": float(scores.std())}
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != len(scores):
            raise ValueError("group_ids must match scores.")
        groups = {}
        for group in sorted(set(group_ids.tolist())):
            values = scores[group_ids == group]
            groups[str(group)] = {"n": len(values), "ocf_mean": float(values.mean())}
        result["groups"] = groups
        result["group_macro_mean"] = float(np.mean([row["ocf_mean"] for row in groups.values()]))
    return result


def sweep_thresholds(predictions, targets, thresholds, *, device="cpu", batch_size=16):
    """Tune thresholds only on a caller-provided training/validation tuning set."""
    rows = []
    for threshold in thresholds:
        scores = score_predictions(
            predictions, targets, threshold=float(threshold), device=device, batch_size=batch_size
        )
        rows.append({"threshold": float(threshold), "n": len(scores), "ocf_mean": float(scores.mean())})
    return rows
