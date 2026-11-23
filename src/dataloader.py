"""Read per-trial spike matrices and training images."""

import json
from math import ceil
from operator import index
from pathlib import Path

import numpy as np
import torch

from src import GLYPH_SIZE
from src.spike_counts import count_spikes


class SpikeBlock:
    """Spike events from one mouse, session, and data split."""

    def __init__(self, path, entry, sample_rate_hz, window_ms):
        """Load spike events and sample boundaries for one session block."""
        self.mouse = entry["mouse"]
        self.session = entry["session"]
        self.n_units = entry["n_units"]
        self.sample_rate_hz = sample_rate_hz
        self.window_ms = window_ms
        with np.load(path, allow_pickle=False) as data:
            self.sample_ids = data["sample_ids"]
            self.group_ptr = data["group_ptr"]
            self.trial_ptr = data["trial_ptr"]
            self.spike_samples = data["spike_samples"]
            self.unit_index = data["unit_index"]
            self.targets = data["target"] if "target" in data else None

        if len(self) != entry["n_samples"]:
            raise ValueError(f"Sample count differs from catalog.json: {path}")
        if (self.targets is not None) != (entry["split"] == "train"):
            raise ValueError(f"Only training blocks should contain targets: {path}")

    def __len__(self):
        """Return the number of samples in this block."""
        return len(self.sample_ids)

    def trials(self, index):
        """Return each trial of one sample. Spike times are relative sample offsets."""
        if not 0 <= index < len(self):
            raise IndexError(index)

        first_trial, last_trial = self.group_ptr[index:index + 2]
        trials = []
        for trial_index in range(first_trial, last_trial):
            start, stop = self.trial_ptr[trial_index:trial_index + 2]
            trials.append({
                "spike_samples": self.spike_samples[start:stop],
                "unit_index": self.unit_index[start:stop],
            })
        return trials

    def counts(self, bin_ms=50):
        """Optionally run the example's spike counting on this block."""
        return count_spikes(self, bin_ms=bin_ms)


def load_catalog(root):
    """Read the index of data blocks and the shared recording settings."""
    return json.loads((Path(root) / "catalog.json").read_text(encoding="utf-8"))


def load_image_size(root):
    """Return the native glyph size declared for this competition dataset."""
    size = load_catalog(root).get("image_size", list(GLYPH_SIZE))
    if size != list(GLYPH_SIZE) or any(type(side) is not int for side in size):
        raise ValueError("This competition expects image_size [64, 64] in catalog.json.")
    return GLYPH_SIZE


def load_blocks(root, split, task=None):
    """Read the shared training blocks or the test blocks for a task."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if task not in (None, "task1", "task2", "task3"):
        raise ValueError("task must be 'task1', 'task2', or 'task3'")

    root = Path(root)
    catalog = load_catalog(root)
    blocks = []
    for entry in catalog["blocks"]:
        if entry["split"] != split:
            continue
        if split == "test" and task is not None and entry["task"] != task:
            continue
        block = SpikeBlock(
            root / entry["path"], entry, catalog["sample_rate_hz"], catalog["window_ms"]
        )
        blocks.append(block)
    return blocks


def load_images(root):
    """Read the shared uint8 training bank with shape (n_pictures, 64, 64)."""
    with np.load(Path(root) / "train" / "pictures.npz", allow_pickle=False) as data:
        images = data["images"]
    if images.ndim != 3 or images.shape[1:] != load_image_size(root) or images.dtype != np.uint8:
        raise ValueError("Training images must be uint8 and match the size in catalog.json")
    return images


class SpikeDataset(torch.utils.data.Dataset):
    """Return one binary spike matrix per trial."""

    def __init__(self, blocks, images=None, bin_ms=1):
        """Store the session blocks, training images, and time-bin width."""
        if bin_ms <= 0:
            raise ValueError("Bin width must be positive.")
        self._blocks = blocks
        self._images = images
        self._size = sum(len(block) for block in blocks)
        self.bin_ms = bin_ms

    def __len__(self):
        """Return the total number of samples across session blocks."""
        return self._size

    def __getitem__(self, item):
        """Return the sample and its per-trial spike matrices."""
        sample_index = index(item)
        if sample_index < 0:
            sample_index += len(self)
        if not 0 <= sample_index < len(self):
            raise IndexError(item)

        # Find the session block and the sample's row within that block.
        for block in self._blocks:
            if sample_index < len(block):
                break
            sample_index -= len(block)

        spike_trials = block.trials(sample_index)

        n_bins = ceil(block.window_ms / self.bin_ms)
        samples_per_bin = block.sample_rate_hz * self.bin_ms / 1000
        spike_matrix = np.zeros((len(spike_trials), block.n_units, n_bins), dtype=np.uint8)
        for trial_index, trial in enumerate(spike_trials):
            time_bins = np.floor(trial["spike_samples"] / samples_per_bin).astype(np.intp)
            # Each unit and time bin stores whether at least one spike occurred.
            spike_matrix[trial_index, trial["unit_index"], time_bins] = 1

        sample = {
            "sample_id": str(block.sample_ids[sample_index]),
            "mouse": block.mouse,
            "session": block.session,
            "n_units": block.n_units,
            "trials": spike_matrix,
        }
        if self._images is not None:
            sample["image"] = self._images[block.targets[sample_index]]
        return sample


def load_dataset(root="data", split="train", task=None, bin_ms=1):
    """Load training or test samples with per-trial spike matrices."""
    blocks = load_blocks(root, split, task)
    images = load_images(root) if split == "train" else None
    return SpikeDataset(blocks, images, bin_ms=bin_ms)
