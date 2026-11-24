"""Crisp seen-character retrieval and neural prototype classification."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def gb2312_characters():
    """Return the 6,763 defined Han characters in the GB2312 B0-F7 grid."""
    characters = []
    for lead in range(0xB0, 0xF8):
        for trail in range(0xA1, 0xFF):
            try:
                characters.append(bytes((lead, trail)).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    if len(characters) != 6763:
        raise RuntimeError("Unexpected GB2312 character count.")
    return characters


def render_gb2312_bank(font_path, font_size=52, variation="Regular"):
    """Render a binary 64x64 GB2312 candidate bank without copying the font."""
    font_path = Path(font_path)
    if not font_path.is_file():
        raise FileNotFoundError(f"Font not found: {font_path}")
    font = ImageFont.truetype(str(font_path), font_size)
    if variation:
        font.set_variation_by_name(variation)
    characters = gb2312_characters()
    bank = np.zeros((len(characters), 64, 64), dtype=np.float32)
    for index, character in enumerate(characters):
        side = max(128, 3 * font_size)
        canvas = Image.new("L", (side, side), 0)
        ImageDraw.Draw(canvas).text((font_size, font_size), character, font=font, fill=255)
        binary = np.asarray(canvas, dtype=np.uint8) >= 128
        ys, xs = np.nonzero(binary)
        if not len(xs):
            continue
        glyph = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if glyph.shape[0] > 64 or glyph.shape[1] > 64:
            raise ValueError(f"Glyph does not fit 64x64 at font size {font_size}: {glyph.shape}")
        y = (64 - glyph.shape[0]) // 2
        x = (64 - glyph.shape[1]) // 2
        bank[index, y:y + glyph.shape[0], x:x + glyph.shape[1]] = glyph
    return bank


def unique_assignment(scores):
    """Assign each row a distinct class, using only unlabeled model scores."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 2-D array.")
    rows, columns = linear_sum_assignment(scores, maximize=True)
    assigned = np.empty(scores.shape[0], dtype=np.int64)
    assigned[rows] = columns
    return assigned


def _directional_embedding(images, pool_size=4):
    """Compact multi-scale signed orientation fields aligned with OCF's image basis."""
    from src.ocf.ocf_cuda import directional_fields

    embeddings = []
    for sigma in (1, 2, 4):
        fields = directional_fields(images[:, 0], sigma)
        pooled = F.avg_pool2d(fields, kernel_size=pool_size, stride=pool_size).flatten(1)
        embeddings.append(F.normalize(pooled, dim=1))
    return torch.cat(embeddings, dim=1)


def _retrieval_embedding(images, metric):
    if metric == "directional_multi":
        return torch.cat([
            _directional_embedding(images, pool_size) for pool_size in (2, 4, 8)
        ], dim=1)
    pool_size = {"directional2": 2, "directional": 4, "directional8": 8}[metric]
    return _directional_embedding(images, pool_size)


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
def glyph_similarity(bank, metric="directional2", device=None):
    """Return pairwise structural similarity for a binary glyph bank."""
    if metric not in {"directional2", "directional", "directional8", "directional_multi"}:
        raise ValueError("glyph_similarity requires a directional metric.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    images = np.asarray(bank, dtype=np.float32)
    embeddings = torch.from_numpy(images[:, None]).to(device)
    embeddings = F.normalize(_retrieval_embedding(embeddings, metric), dim=1)
    return (embeddings @ embeddings.T).cpu().numpy()


@torch.inference_mode()
def softmax_dct_similarity(logits, centered_coefficients, device=None, batch_size=64):
    """Return cosine similarity to a fixed T=1 softmax-expected continuous DCT image."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logits = np.asarray(logits, dtype=np.float32)
    centered = torch.as_tensor(centered_coefficients, dtype=torch.float32, device=device)
    if logits.ndim != 2 or centered.ndim != 2 or logits.shape[1] != centered.shape[0]:
        raise ValueError("Expected class logits and a matching centered DCT bank.")
    normalized_bank = F.normalize(centered, dim=1)
    output = []
    for start in range(0, len(logits), batch_size):
        batch = torch.from_numpy(logits[start:start + batch_size]).to(device)
        probabilities = F.softmax(batch, dim=1)
        expected = F.normalize(probabilities @ centered, dim=1)
        output.append((expected @ normalized_bank.T).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def retrieve_glyphs(predictions, bank, metric="cosine", blur_sigma=1.0, batch_size=64, device=None):
    """Return the nearest binary bank glyph and its index for every prediction."""
    predictions = np.asarray(predictions, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    if predictions.ndim != 3 or bank.ndim != 3 or predictions.shape[1:] != bank.shape[1:]:
        raise ValueError("Expected matched (N, H, W) predictions and (K, H, W) bank images.")
    directional_metrics = {"directional2", "directional", "directional8", "directional_multi"}
    if metric not in ("cosine", "mse", *directional_metrics):
        raise ValueError("Unsupported glyph retrieval metric.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bank_tensor = torch.from_numpy(bank[:, None]).to(device)
    if metric in directional_metrics:
        bank_tensor = _retrieval_embedding(bank_tensor, metric)
    else:
        bank_tensor = _blur(bank_tensor, blur_sigma).flatten(1)
    if metric == "cosine" or metric in directional_metrics:
        bank_tensor = F.normalize(bank_tensor, dim=1)
    bank_square = bank_tensor.square().sum(dim=1)

    indices = []
    for start in range(0, len(predictions), batch_size):
        batch = torch.from_numpy(predictions[start:start + batch_size, None]).to(device)
        if metric in directional_metrics:
            batch = _retrieval_embedding(batch, metric)
        else:
            batch = _blur(batch, blur_sigma).flatten(1)
        if metric == "cosine" or metric in directional_metrics:
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
