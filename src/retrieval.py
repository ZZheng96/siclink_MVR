"""Crisp seen-character retrieval and neural prototype classification."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def unique_assignment(scores):
    """Assign each row a distinct class, using only unlabeled model scores."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 2-D array.")
    rows, columns = linear_sum_assignment(scores, maximize=True)
    assigned = np.empty(scores.shape[0], dtype=np.int64)
    assigned[rows] = columns
    return assigned


def _directional_embedding(images):
    """Compact multi-scale signed orientation fields aligned with OCF's image basis."""
    from src.ocf.ocf_cuda import directional_fields

    embeddings = []
    for sigma in (1, 2, 4):
        fields = directional_fields(images[:, 0], sigma)
        pooled = F.avg_pool2d(fields, kernel_size=4, stride=4).flatten(1)
        embeddings.append(F.normalize(pooled, dim=1))
    return torch.cat(embeddings, dim=1)


def _blur(images, sigma):
    if sigma <= 0:
        return images
    radius = int(4 * sigma + 0.5)
    coordinates = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, 1, -1)
    images = F.conv2d(images, kernel, padding=(0, radius))
    return F.conv2d(images, kernel.reshape(1, 1, -1, 1), padding=(radius, 0))


@torch.inference_mode()
def retrieve_glyphs(predictions, bank, metric="cosine", blur_sigma=1.0, batch_size=64, device=None):
    """Return the nearest binary bank glyph and its index for every prediction."""
    predictions = np.asarray(predictions, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    if predictions.ndim != 3 or bank.ndim != 3 or predictions.shape[1:] != bank.shape[1:]:
        raise ValueError("Expected matched (N, H, W) predictions and (K, H, W) bank images.")
    if metric not in ("cosine", "mse", "directional"):
        raise ValueError("metric must be 'cosine', 'mse', or 'directional'.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bank_tensor = torch.from_numpy(bank[:, None]).to(device)
    if metric == "directional":
        bank_tensor = _directional_embedding(bank_tensor)
    else:
        bank_tensor = _blur(bank_tensor, blur_sigma).flatten(1)
    if metric in ("cosine", "directional"):
        bank_tensor = F.normalize(bank_tensor, dim=1)
    bank_square = bank_tensor.square().sum(dim=1)

    indices = []
    for start in range(0, len(predictions), batch_size):
        batch = torch.from_numpy(predictions[start:start + batch_size, None]).to(device)
        if metric == "directional":
            batch = _directional_embedding(batch)
        else:
            batch = _blur(batch, blur_sigma).flatten(1)
        if metric in ("cosine", "directional"):
            scores = F.normalize(batch, dim=1) @ bank_tensor.T
        else:
            scores = 2 * batch @ bank_tensor.T - batch.square().sum(dim=1, keepdim=True) - bank_square
        indices.append(scores.argmax(dim=1).cpu().numpy())
    indices = np.concatenate(indices)
    return bank[indices].copy(), indices


class CentroidClassifier:
    """Nearest class-centroid classifier for seen-character tasks."""

    def __init__(self, metric="cosine", center=True):
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'.")
        self.metric = metric
        self.center = center

    def fit(self, features, targets):
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets)
        self.mean = features.mean(axis=0) if self.center else np.zeros(features.shape[1], np.float32)
        centered = features - self.mean
        self.classes = np.unique(targets)
        self.centroids = np.stack([centered[targets == target].mean(axis=0) for target in self.classes])
        self.scale = self.centroids.std(axis=0)
        self.scale = np.maximum(self.scale, 1e-6)
        self.centroids /= self.scale
        if self.metric == "cosine":
            self.centroids /= np.maximum(np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12)
        self.centroid_square = np.square(self.centroids).sum(axis=1)
        return self

    def decision_function(self, features, batch_size=256):
        """Return one centroid-compatibility score per class."""
        features = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        if self.metric == "cosine":
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        output = []
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size]
            scores = 2 * batch @ self.centroids.T - self.centroid_square
            if self.metric == "euclidean":
                scores -= np.square(batch).sum(axis=1, keepdims=True)
            output.append(scores.astype(np.float32))
        return np.concatenate(output)

    def predict(self, features, batch_size=256):
        features = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        if self.metric == "cosine":
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        output = []
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size]
            scores = 2 * batch @ self.centroids.T - self.centroid_square
            if self.metric == "euclidean":
                scores -= np.square(batch).sum(axis=1, keepdims=True)
            output.append(self.classes[scores.argmax(axis=1)])
        return np.concatenate(output)
