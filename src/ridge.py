"""Regularized multi-output ridge decoding with an optional fixed DCT basis."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn, idctn


class PCABasis:
    """A centered image basis fitted only on permitted training glyphs."""

    def __init__(self, mean, components):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)

    @classmethod
    def fit(cls, images, rank=256, device=None):
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3 or images.shape[1:] != (64, 64) or len(images) < rank:
            raise ValueError("PCA requires (N, 64, 64) images with N >= rank.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        flat = torch.from_numpy(images.reshape(len(images), -1)).to(device)
        _, _, components = torch.pca_lowrank(flat, q=rank, center=True, niter=4)
        mean = flat.mean(dim=0)
        return cls(mean.cpu().numpy(), components.cpu().numpy())

    def encode(self, images):
        images = np.asarray(images, dtype=np.float32)
        flat = images.reshape(len(images), -1)
        return (flat - self.mean) @ self.components

    def decode(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float32)
        images = coefficients @ self.components.T + self.mean
        return images.reshape(len(coefficients), 64, 64).astype(np.float32)


def dct_encode(images, cutoff=32):
    """Encode N×64×64 images in a fixed low-frequency 2-D DCT basis."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (64, 64):
        raise ValueError("Expected images shaped (N, 64, 64).")
    if not 1 <= cutoff <= 64:
        raise ValueError("cutoff must be between 1 and 64.")
    coefficients = dctn(images, axes=(-2, -1), norm="ortho")
    return coefficients[:, :cutoff, :cutoff].reshape(len(images), -1).copy()


def dct_decode(coefficients, cutoff=32):
    """Decode DCT coefficients to N×64×64 grayscale images."""
    coefficients = np.asarray(coefficients, dtype=np.float32)
    if coefficients.ndim != 2 or coefficients.shape[1] != cutoff * cutoff:
        raise ValueError("Coefficient shape does not match cutoff.")
    square = np.zeros((len(coefficients), 64, 64), dtype=np.float32)
    square[:, :cutoff, :cutoff] = coefficients.reshape(len(coefficients), cutoff, cutoff)
    return idctn(square, axes=(-2, -1), norm="ortho")


@dataclass
class RidgeModel:
    """A selected standardized ridge model with an intercept."""

    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        standardized = (features - self.x_mean) / self.x_scale
        return standardized @ self.weights + self.bias

    def save(self, path):
        """Save a portable model without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alpha=np.asarray(self.alpha),
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            weights=self.weights,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                alpha=float(data["alpha"]),
                x_mean=data["x_mean"],
                x_scale=data["x_scale"],
                weights=data["weights"],
                bias=data["bias"],
            )


class RidgePath:
    """Share one eigendecomposition across several regularization values."""

    def __init__(self):
        self._eigenvalues = None

    def fit(self, features, targets):
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("Expected matched 2-D feature and target matrices.")
        if len(features) < 2:
            raise ValueError("Ridge fitting requires at least two samples.")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Ridge inputs must be finite.")

        self.x_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.x_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.x_scale = np.maximum(self.x_scale, 1e-6)
        self.y_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
        x = (features - self.x_mean) / self.x_scale
        y = targets - self.y_mean
        n_samples = float(len(x))
        gram = (x.T @ x).astype(np.float64) / n_samples
        cross = (x.T @ y).astype(np.float64) / n_samples
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(gram)
        self._eigenvalues = np.maximum(self._eigenvalues, 0)
        self._projected_cross = self._eigenvectors.T @ cross
        self.n_features = features.shape[1]
        self.n_outputs = targets.shape[1]
        return self

    def model(self, alpha):
        if self._eigenvalues is None:
            raise RuntimeError("Call fit before selecting a ridge model.")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative.")
        if alpha == 0 and self._eigenvalues.min() <= 1e-12:
            raise ValueError("alpha=0 is singular; use positive regularization.")
        projected = self._projected_cross / (self._eigenvalues[:, None] + alpha)
        weights = (self._eigenvectors @ projected).astype(np.float32)
        bias = self.y_mean.copy()
        return RidgeModel(alpha, self.x_mean, self.x_scale, weights, bias.astype(np.float32))

    def predict(self, features, alpha):
        return self.model(alpha).predict(features)
