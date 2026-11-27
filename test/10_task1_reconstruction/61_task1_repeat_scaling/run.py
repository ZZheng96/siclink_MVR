"""Exploratory: Task1 retrieval OCF as a function of repeat count used at inference."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.neural import load_regressor, predict_logits
from src.preprocessing import load_cached_blocks
from src.retrieval import retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
MODELS = ROOT / "test/.cache/evaluate_final_task1_v2/models/task1"
OUT = ROOT / "test/10_task1_reconstruction/61_task1_repeat_scaling"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPEATS = (1, 2, 3, 4)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    heldout = np.sort(shuffled[64:])
    bank = gt[~np.isin(np.arange(len(gt)), split.validation_pictures)]
    models = {m: load_regressor(MODELS / m / "regressor.pt", device=DEVICE) for m in ("mice1", "mice2")}

    output = {"scope": "exploratory repeat-scaling diagnostic; frozen v5 Task1 regressors", "repeats": list(REPEATS)}
    for n_repeat in REPEATS:
        coeffs, truths = [], []
        for block in blocks:
            part = split.blocks[block.key]
            trials = part.validation_trials
            labels = block.trial_targets[trials]
            order = np.argsort(labels, kind="stable")
            trials, labels = trials[order], labels[order]
            counts = np.asarray(block.features[trials], dtype=np.float32)
            unique = np.unique(labels)
            rows = []
            for target in unique:
                rows.append(counts[labels == target][:n_repeat].mean(axis=0))
            keep = np.isin(unique, heldout)
            rows = np.stack(rows)[keep]
            coeffs.append(np.asarray(predict_logits(models[block.mouse], rows.astype(np.float32),
                                                   batch_size=256, device=DEVICE), dtype=np.float32))
            truths.append(gt[unique[keep]])
        coefficients = np.concatenate(coeffs)
        truth = np.concatenate(truths)
        continuous = np.clip(np.asarray(dct_decode(coefficients, 32), dtype=np.float32), 0.0, 1.0)
        retrieved, _ = retrieve_glyphs(continuous, bank, metric="directional", blur_sigma=0.0, device=DEVICE)
        scores = score_predictions(retrieved, truth, device=str(DEVICE), batch_size=16)
        output[str(n_repeat)] = {"retrieval_ocf": summarize(scores), "rows": int(len(scores))}
        print(n_repeat, "repeats -> retrieval OCF", float(scores.mean()), flush=True)

    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote exploratory_results.json", flush=True)


if __name__ == "__main__":
    main()
