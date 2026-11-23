"""Memory-conscious trial-level spike features and repeat aggregation."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

from src.dataloader import load_blocks


@dataclass
class FeatureBlock:
    """Trial-level counts paired with the official sample grouping."""

    mouse: str
    session: str
    split: str
    task: str
    sample_ids: np.ndarray
    group_ptr: np.ndarray
    targets: np.ndarray | None
    features: np.ndarray
    n_units: int
    bin_ms: int

    @property
    def key(self):
        return f"{self.mouse}/{self.session}"

    @property
    def n_samples(self):
        return len(self.sample_ids)

    @property
    def n_trials(self):
        return len(self.features)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def trial_sample_indices(self):
        return np.repeat(np.arange(self.n_samples), np.diff(self.group_ptr))

    @property
    def trial_targets(self):
        if self.targets is None:
            raise ValueError("Test blocks do not contain training targets.")
        return self.targets[self.trial_sample_indices]

    def aggregate(self, sample_indices=None, average=True):
        return aggregate_samples(self.features, self.group_ptr, sample_indices, average)


def trial_count_features(block, bin_ms=50):
    """Return one unnormalised spike-count vector per trial."""
    if not isinstance(bin_ms, (int, np.integer)) or isinstance(bin_ms, bool) or bin_ms <= 0:
        raise ValueError("bin_ms must be a positive integer.")
    samples_per_bin = block.sample_rate_hz * bin_ms / 1000
    window_samples = block.sample_rate_hz * block.window_ms / 1000
    if samples_per_bin % 1:
        raise ValueError("bin_ms must represent a whole number of samples.")
    if block.spike_samples.min(initial=0) < 0 or block.spike_samples.max(initial=-1) >= window_samples:
        raise ValueError("Spike times are outside the trial window.")
    n_bins = ceil(window_samples / samples_per_bin)
    n_trials = len(block.trial_ptr) - 1
    events_per_trial = np.diff(block.trial_ptr)
    trial_indices = np.repeat(np.arange(n_trials), events_per_trial)
    time_bins = block.spike_samples // int(samples_per_bin)
    flat = (trial_indices * block.n_units + block.unit_index) * n_bins + time_bins
    counts = np.bincount(flat, minlength=n_trials * block.n_units * n_bins)
    return counts.reshape(n_trials, block.n_units * n_bins).astype(np.float32)


def aggregate_samples(features, group_ptr, sample_indices=None, average=True):
    """Sum or average trial features within each selected sample."""
    features = np.asarray(features, dtype=np.float32)
    group_ptr = np.asarray(group_ptr)
    n_samples = len(group_ptr) - 1
    if sample_indices is None:
        sample_indices = np.arange(n_samples)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if sample_indices.ndim != 1 or len(sample_indices) == 0:
        raise ValueError("sample_indices must be a non-empty 1-D array.")
    if sample_indices.min() < 0 or sample_indices.max() >= n_samples:
        raise IndexError("Sample index outside group_ptr.")
    repeats = np.diff(group_ptr)
    selected_repeats = repeats[sample_indices]
    if np.all(repeats == repeats[0]):
        grouped = features.reshape(n_samples, repeats[0], features.shape[1])[sample_indices]
        output = grouped.sum(axis=1, dtype=np.float32)
    else:
        output = np.stack([
            features[group_ptr[index]:group_ptr[index + 1]].sum(axis=0, dtype=np.float32)
            for index in sample_indices
        ])
    if average:
        output /= selected_repeats[:, None]
    return output


def aggregate_by_target(features, targets):
    """Average independent training rows that share a character target."""
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Expected one target per feature row.")
    if not len(targets):
        raise ValueError("Cannot aggregate an empty target array.")
    order = np.argsort(targets, kind="stable")
    sorted_targets = targets[order]
    sorted_features = features[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_targets)) + 1))
    sums = np.add.reduceat(sorted_features, starts, axis=0, dtype=np.float32)
    counts = np.diff(np.concatenate((starts, [len(targets)])))
    unique_targets = sorted_targets[starts]
    return unique_targets, sums / counts[:, None]


def trials_for_samples(group_ptr, sample_indices):
    """Return concatenated trial indices for arbitrary sample indices."""
    group_ptr = np.asarray(group_ptr)
    return np.concatenate([
        np.arange(group_ptr[index], group_ptr[index + 1], dtype=np.int64)
        for index in np.asarray(sample_indices, dtype=np.int64)
    ]) if len(sample_indices) else np.empty(0, dtype=np.int64)


def load_feature_blocks(root, split="train", task=None, bin_ms=50, mouse=None):
    """Load official blocks and convert their events to trial-level features."""
    output = []
    for block in load_blocks(Path(root), split=split, task=task):
        if mouse is not None and block.mouse != mouse:
            continue
        output.append(FeatureBlock(
            mouse=block.mouse,
            session=block.session,
            split=block.targets is None and "test" or "train",
            task=task or "train",
            sample_ids=block.sample_ids,
            group_ptr=block.group_ptr,
            targets=block.targets,
            features=trial_count_features(block, bin_ms),
            n_units=block.n_units,
            bin_ms=bin_ms,
        ))
    return sorted(output, key=lambda block: (block.mouse, block.session))
