"""Optional spike-count preprocessing for the DDPM example.

Participants can skip this step and use the per-trial binary matrices
returned by SpikeDataset.
"""

from math import ceil

import numpy as np
import torch


def count_spikes(block, bin_ms=50):
    """Return average spike counts with shape (samples, units * time bins)."""
    samples_per_bin = block.sample_rate_hz * bin_ms / 1000
    window_samples = block.sample_rate_hz * block.window_ms / 1000
    if samples_per_bin <= 0 or samples_per_bin % 1:
        raise ValueError("Bin width must be a positive whole number of samples")
    if block.spike_samples.min(initial=0) < 0 or block.spike_samples.max(initial=-1) >= window_samples:
        raise ValueError("Spike times are outside the trial window in catalog.json")

    n_time_bins = ceil(window_samples / samples_per_bin)
    n_features = block.n_units * n_time_bins
    events_per_sample = np.diff(block.trial_ptr[block.group_ptr])
    sample_indices = np.repeat(np.arange(len(block)), events_per_sample)
    time_bins = block.spike_samples // int(samples_per_bin)

    # Count events by sample, unit, and time bin in one pass.
    flat_indices = (sample_indices * block.n_units + block.unit_index) * n_time_bins + time_bins
    counts = np.bincount(flat_indices, minlength=len(block) * n_features)
    counts = counts.reshape(len(block), n_features).astype(np.float32)

    # Divide by the number of trials in each sample. Single-trial counts stay unchanged.
    counts /= np.diff(block.group_ptr)[:, None]
    return counts


class SpikeCountDataset(torch.utils.data.Dataset):
    """Pair precomputed spike counts with the target images for the example model."""

    def __init__(self, counts, targets, images):
        """Store the counts and the image index of each training sample."""
        self.counts = torch.as_tensor(counts, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.images = images

        if self.counts.ndim != 2 or self.targets.shape != (len(self.counts),):
            raise ValueError("Expected one image index per spike count vector")
        if len(self.targets) and (self.targets.min() < 0 or self.targets.max() >= len(images)):
            raise ValueError("A target index is outside the training image bank")
        self.n_features = self.counts.shape[1]

    def __len__(self):
        """Return the number of training samples."""
        return len(self.counts)

    def __getitem__(self, index):
        """Return the count vector and normalized target image of one sample."""
        spike_counts = self.counts[index]
        image = self.images[self.targets[index]].float()
        if self.images.dtype == torch.uint8:
            image = image / 255
        return spike_counts, image
