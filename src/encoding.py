"""Candidate scoring for image-to-neural-response encoding models."""

import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def response_scores(observations, prototypes, metric="mse", batch_size=128, device=None):
    """Compare observed neural vectors with every candidate-character response."""
    observations = np.asarray(observations, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if observations.ndim != 2 or prototypes.ndim != 2 or observations.shape[1] != prototypes.shape[1]:
        raise ValueError("Expected matched 2-D observed and prototype neural responses.")
    if metric not in ("mse", "cosine"):
        raise ValueError("metric must be 'mse' or 'cosine'.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    prototype = torch.from_numpy(prototypes).to(device)
    if metric == "cosine":
        prototype = F.normalize(prototype, dim=1)
    prototype_square = prototype.square().sum(dim=1)
    output = []
    for start in range(0, len(observations), batch_size):
        batch = torch.from_numpy(observations[start:start + batch_size]).to(device)
        if metric == "cosine":
            scores = F.normalize(batch, dim=1) @ prototype.T
        else:
            scores = 2 * batch @ prototype.T - prototype_square - batch.square().sum(dim=1, keepdim=True)
        output.append(scores.float().cpu().numpy())
    return np.concatenate(output)
