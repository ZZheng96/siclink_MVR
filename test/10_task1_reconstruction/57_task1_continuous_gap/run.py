"""Exploratory: measure continuous-reconstruction OCF on train vs held-out pictures.

Uses the frozen v5 Task1 regressors and the fixed public Task1 protocol.
Writes test/10_task1_reconstruction/57_task1_continuous_gap/exploratory_results.json. No training, no Git commit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.neural import load_regressor, predict_logits
from src.preprocessing import load_cached_blocks
from src.ridge import dct_decode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
MODELS = ROOT / "test/.cache/evaluate_final_task1_v2/models/task1"
OUT = ROOT / "test/10_task1_reconstruction/57_task1_continuous_gap"
BIN_MS = 25
DCT_CUTOFF = 32
N_PER_BLOCK = 211
SEED = 42
PICTURE_SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=BIN_MS)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt_all = (np.asarray(pictures) >= 128).astype(np.float32)
    split = make_split(blocks, "task1", seed=SEED)
    shuffled = np.random.default_rng(PICTURE_SEED).permutation(split.validation_pictures)
    heldout = np.sort(shuffled[64:])
    train_pictures = np.array(sorted(set(np.unique(np.concatenate([b.targets for b in blocks]))).difference(
        set(split.validation_pictures.tolist())
    )), dtype=np.int64)
    sampled_train = np.sort(np.random.default_rng(SEED + 7).permutation(train_pictures)[:N_PER_BLOCK])
    print(f"heldout={len(heldout)} train_pictures={len(train_pictures)} sampled_train={len(sampled_train)}", flush=True)

    models = {mouse: load_regressor(MODELS / mouse / "regressor.pt", device=DEVICE) for mouse in ("mice1", "mice2")}
    records = {"train": {"continuous": [], "binary": []}, "heldout": {"continuous": [], "binary": []}}
    for block in blocks:
        part = split.blocks[block.key]
        model = models[block.mouse]
        for role, trial_indices, wanted in (
            ("train", part.train_trials, sampled_train),
            ("heldout", part.validation_trials, heldout),
        ):
            targets, counts = aggregate_by_target(
                np.asarray(block.features[trial_indices], dtype=np.float32),
                block.trial_targets[trial_indices],
            )
            keep = np.isin(targets, wanted)
            targets, counts = targets[keep], counts[keep]
            order = np.argsort(targets)
            targets, counts = targets[order], counts[order]
            coefficients = predict_logits(model, counts, batch_size=256, device=DEVICE)
            continuous = dct_decode(coefficients, DCT_CUTOFF)
            continuous = np.asarray(continuous, dtype=np.float32)
            clipped = np.clip(continuous, 0.0, 1.0)
            binary = (clipped >= 0.5).astype(np.float32)
            truth = gt_all[targets]
            cont_scores = score_predictions(clipped, truth, device=DEVICE, batch_size=16)
            bin_scores = score_predictions(binary, truth, device=DEVICE, batch_size=16)
            records[role]["continuous"].extend(cont_scores.tolist())
            records[role]["binary"].extend(bin_scores.tolist())
        print(f"done {block.key}", flush=True)

    output = {
        "scope": "exploratory; frozen v5 Task1 regressors; fixed public Task1 protocol; not a Kaggle score",
        "protocol": {
            "split_seed": SEED, "picture_seed": PICTURE_SEED, "bin_ms": BIN_MS,
            "dct_cutoff": DCT_CUTOFF, "heldout_pictures": heldout.tolist(),
            "sampled_train_pictures": sampled_train.tolist(),
            "model_dir": str(MODELS),
        },
        "train_continuous_ocf": summarize(records["train"]["continuous"]),
        "train_binary_ocf": summarize(records["train"]["binary"]),
        "heldout_continuous_ocf": summarize(records["heldout"]["continuous"]),
        "heldout_binary_ocf": summarize(records["heldout"]["binary"]),
        "v5_heldout_final_ocf": 18.43706,
        "v5_heldout_oracle_at_32": 33.01849,
        "font_bank_ceiling_ocf": 79.42,
    }
    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
