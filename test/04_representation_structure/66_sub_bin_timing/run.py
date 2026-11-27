"""Experiment 66: does sub-bin spike timing carry identity information beyond counts?

Preregistered development paired test on Task 3 (single trial, the bottleneck task).
Public train blocks only; no competition test file is read and no hidden label is used.

The 250 ms window is sampled at 30 kHz (7,500 positions). Every training block uses
all 7,500 positions and all 300 positions inside a 10 ms bin, so the raw events keep
sub-bin timing that count features discard. Per trial the features are:

  counts   : per (unit, 25 ms bin) spike count, log1p transformed
  timing   : per (unit, 25 ms bin) mean normalised within-bin offset (0 if empty)

Arms share one model recipe and differ only in the input block:

  counts   : counts only
  real     : counts + timing from the recorded offsets
  shuffled : counts + timing after permuting within-bin offsets between cells that
             carry the same spike count inside the same trial

The shuffle preserves every (unit, bin) count and the per-trial multiset of offsets
exactly. The only destroyed quantity is which cell fired when inside its bin, so
`real - shuffled` isolates timing information rather than feature dimensionality.

Gate, frozen before any scoring: macro OCF delta (real - shuffled) >= +2.0, both mouse
deltas non-negative, and both seed deltas non-negative. Anything less is rejected.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_blocks, load_images
from src.evaluation import score_predictions
from src.neural import SpikeClassifier, predict_logits
from src.retrieval import unique_assignment
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
N_CLASSES = 2748
DEV_PICTURES = np.sort(np.random.default_rng(20261128).permutation(N_CLASSES)[:256])
SEEDS = (20261126, 20261127)
BIN_MS = 25
EPOCHS = 9
DROPOUT = {"mice1": 0.2, "mice2": 0.35}
WINDOW_SAMPLES = 7500
ARMS = ("counts", "real", "shuffled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/04_representation_structure/66_sub_bin_timing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _shuffle_offsets(cell, offset_fraction, rng):
    """Permute within-bin offsets between cells that hold the same spike count."""
    order = np.argsort(cell, kind="stable")
    sorted_cells = cell[order]
    sorted_fraction = offset_fraction[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    lengths = np.diff(np.r_[starts, len(cell)])
    shuffled = np.empty_like(sorted_fraction)
    for count in np.unique(lengths):
        targets = np.flatnonzero(lengths == count)
        sources = targets[rng.permutation(len(targets))]
        for target, source in zip(targets, sources):
            shuffled[starts[target]:starts[target] + count] = \
                sorted_fraction[starts[source]:starts[source] + count]
    output = np.empty_like(offset_fraction)
    output[order] = shuffled
    return output


def build_features(block, *, shuffle=False, seed=0):
    """Return log1p counts and within-bin mean offset features for every trial."""
    bin_samples = int(block.sample_rate_hz * BIN_MS / 1000)
    n_bins = WINDOW_SAMPLES // bin_samples
    n_trials = len(block.trial_ptr) - 1
    width = block.n_units * n_bins
    counts = np.zeros((n_trials, width), dtype=np.float32)
    timing = np.zeros((n_trials, width), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for trial in range(n_trials):
        start, stop = block.trial_ptr[trial], block.trial_ptr[trial + 1]
        if start == stop:
            continue
        offsets = block.spike_samples[start:stop].astype(np.int64)
        bins = offsets // bin_samples
        cell = block.unit_index[start:stop].astype(np.int64) * n_bins + bins
        fraction = (offsets - bins * bin_samples).astype(np.float32) / bin_samples
        trial_counts = np.bincount(cell, minlength=width).astype(np.float32)
        counts[trial] = trial_counts
        if shuffle:
            fraction = _shuffle_offsets(cell, fraction, rng)
        timing[trial] = np.bincount(cell, weights=fraction, minlength=width).astype(np.float32) \
            / np.maximum(trial_counts, 1)
    return np.log1p(counts), timing


class View:
    """Minimal block view exposing exactly what make_split and training need."""

    def __init__(self, block, features):
        self.mouse = block.mouse
        self.session = block.session
        self.group_ptr = block.group_ptr
        self.targets = block.targets
        self.sample_ids = block.sample_ids
        self.features = features
        self.n_units = block.n_units

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
    def trial_targets(self):
        return self.targets[np.repeat(np.arange(self.n_samples), np.diff(self.group_ptr))]


def fit_model(x, y, mouse, device):
    torch.manual_seed(42)
    model = SpikeClassifier(x.shape[1], N_CLASSES, hidden=512, dropout=DROPOUT[mouse], apply_log=False)
    model.set_normalization(x)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3,
                                  fused=device.type == "cuda")
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(42 + epoch).permutation(len(y))
        total = 0.0
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            inputs = torch.from_numpy(x[index]).to(device)
            labels = torch.from_numpy(y[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(inputs).float(), labels, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(index)
        print(f"  epoch={epoch}/{EPOCHS} loss={total / len(y):.6f}", flush=True)
    return model


class OcfCache:
    """Cache official OCF for (prediction index, target index) gallery pairs."""

    def __init__(self, images, device, batch_size):
        self.images = images
        self.device = device
        self.batch_size = batch_size
        self.store = {}

    def score(self, predictions, targets):
        pairs = np.unique(np.stack([predictions, targets], axis=1), axis=0)
        missing = [tuple(int(v) for v in pair) for pair in pairs
                   if (int(pair[0]), int(pair[1])) not in self.store]
        if missing:
            array = np.asarray(missing, dtype=np.int64)
            values = score_predictions(
                self.images[array[:, 0]], self.images[array[:, 1]],
                device=str(self.device), batch_size=self.batch_size,
            )
            for pair, value in zip(missing, values):
                self.store[pair] = float(value)
        return np.asarray([self.store[(int(p), int(t))] for p, t in zip(predictions, targets)])


def prepare(blocks, split, mouse):
    """Concatenate train rows and keep per-block validation rows with a train-only normaliser."""
    features, targets, validation = [], [], {}
    for block in blocks:
        if block.mouse != mouse:
            continue
        part = split.blocks[block.key]
        raw = np.asarray(block.features, dtype=np.float32)
        mean = raw[part.train_trials].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(raw[part.train_trials].std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
        normalised = ((raw - mean) / scale).astype(np.float32, copy=False)
        features.append(normalised[part.train_trials])
        targets.append(block.trial_targets[part.train_trials])
        validation[block.key] = (normalised[part.validation_trials],
                                 block.trial_targets[part.validation_trials])
    return np.concatenate(features), np.concatenate(targets), validation


def evaluate(validation, cache):
    """Assign every row a unique character and report official OCF on development pictures."""
    rows = {}
    for key, (logits, targets) in validation.items():
        indices = unique_assignment(logits.astype(np.float64))
        mask = np.isin(targets, DEV_PICTURES)
        values = cache.score(indices[mask], targets[mask])
        rows[key] = {
            "rows": int(mask.sum()),
            "assigned_ocf": float(values.mean()),
            "assigned_top1": float((indices[mask] == targets[mask]).mean()),
        }
    return rows


def build_summary(folds):
    """Average OCF over mice and blocks per arm, then apply the frozen gate."""
    per_fold = {}
    for fold, arms in folds.items():
        per_fold[fold] = {
            arm: float(np.mean([row["assigned_ocf"] for mouse_rows in arms[arm].values()
                                for row in mouse_rows.values()]))
            for arm in ARMS
        }
    macro = {arm: float(np.mean([per_fold[fold][arm] for fold in per_fold])) for arm in ARMS}
    fold_delta = {fold: per_fold[fold]["real"] - per_fold[fold]["shuffled"] for fold in per_fold}
    mouse_delta = {}
    for mouse in ("mice1", "mice2"):
        deltas = [np.mean([row["assigned_ocf"] for row in folds[fold]["real"][mouse].values()])
                  - np.mean([row["assigned_ocf"] for row in folds[fold]["shuffled"][mouse].values()])
                  for fold in folds]
        mouse_delta[mouse] = float(np.mean(deltas))
    gain = float(np.mean(list(fold_delta.values())))
    return {
        "per_arm_macro": macro,
        "real_minus_shuffled": gain,
        "real_minus_counts": float(macro["real"] - macro["counts"]),
        "shuffled_minus_counts": float(macro["shuffled"] - macro["counts"]),
        "fold_deltas": {fold: float(value) for fold, value in fold_delta.items()},
        "mouse_deltas": mouse_delta,
        "pass": bool(gain >= 2.0 and all(value >= 0 for value in fold_delta.values())
                     and all(value >= 0 for value in mouse_delta.values())),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "development_results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} exists; pass --overwrite.")
    started = time.perf_counter()
    device = torch.device(args.device)
    images = (load_images(DATA) >= 128).astype(np.float32)
    cache = OcfCache(images, device, args.ocf_batch_size)
    raw_blocks = load_blocks(DATA, "train")

    results = {
        "experiment": "66_sub_bin_timing",
        "status": "development_paired_test",
        "scope": "public train blocks only; task3 single-trial folds; development pictures",
        "bin_ms": BIN_MS,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "development_pictures": DEV_PICTURES.tolist(),
        "gate": {"macro_minimum_absolute_gain": 2.0, "both_mice_nonnegative": True,
                 "required_nonnegative_folds": 2},
        "folds": {},
    }

    splits = {seed: make_split([View(block, np.empty((len(block.trial_ptr) - 1, 0), np.float32))
                                for block in raw_blocks], "task3", seed=seed)
              for seed in SEEDS}

    for mouse in ("mice1", "mice2"):
        mouse_blocks = [block for block in raw_blocks if block.mouse == mouse]
        views = {}
        for block in mouse_blocks:
            key = f"{block.mouse}/{block.session}"
            counts, timing = build_features(block)
            _, timing_shuffled = build_features(block, shuffle=True, seed=7)
            views[key] = {
                "counts": View(block, counts),
                "real": View(block, np.concatenate([counts, timing], axis=1)),
                "shuffled": View(block, np.concatenate([counts, timing_shuffled], axis=1)),
            }
            print(f"features {key}: width={counts.shape[1]}", flush=True)
            del counts, timing, timing_shuffled
        for seed in SEEDS:
            for arm in ARMS:
                blocks = [views[f"{block.mouse}/{block.session}"][arm] for block in mouse_blocks]
                x, y, validation = prepare(blocks, splits[seed], mouse)
                print(f"task3/{seed}/{mouse}/{arm}: train={len(y)} width={x.shape[1]}", flush=True)
                model = fit_model(x, y, mouse, device)
                outputs = {key: (predict_logits(model, features, device=device).astype(np.float64), targets)
                           for key, (features, targets) in validation.items()}
                rows = evaluate(outputs, cache)
                results["folds"].setdefault(f"task3/{seed}", {}).setdefault(arm, {})[mouse] = rows
                print(f"task3/{seed}/{mouse}/{arm}: "
                      f"ocf={np.mean([row['assigned_ocf'] for row in rows.values()]):.4f}", flush=True)
                del model, x, y, validation, outputs, rows
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del views

    results["summary"] = build_summary(results["folds"])
    results["elapsed_seconds"] = time.perf_counter() - started
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
