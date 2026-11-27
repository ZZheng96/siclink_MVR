"""Exploratory gain scan: does DCT coefficient rescaling recover OCF?"""

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAINS = (1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3, 2.6)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def main():
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=25)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    heldout = np.sort(shuffled[64:])[:64]
    models = {m: load_regressor(MODELS / m / "regressor.pt", device=DEVICE) for m in ("mice1", "mice2")}

    coeffs, truths = [], []
    for block in blocks:
        part = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            np.asarray(block.features[part.validation_trials], dtype=np.float32),
            block.trial_targets[part.validation_trials],
        )
        keep = np.isin(targets, heldout)
        order = np.argsort(targets[keep])
        targets, counts = targets[keep][order], counts[keep][order]
        coeffs.append(np.asarray(predict_logits(models[block.mouse], counts, batch_size=256, device=DEVICE),
                                 dtype=np.float32))
        truths.append(gt[targets])
    coeffs = np.concatenate(coeffs)
    truths = np.concatenate(truths)

    output = {"scope": "exploratory gain scan on 64 held-out pictures x 8 blocks; not a selection result",
              "rows": int(len(coeffs)), "gains": list(GAINS), "results": {}}
    for gain in GAINS:
        continuous = np.asarray(dct_decode(coeffs * gain, 32), dtype=np.float32)
        clipped = np.clip(continuous, 0.0, 1.0)
        binary = (clipped >= 0.5).astype(np.float32)
        cont = score_predictions(clipped, truths, device=DEVICE, batch_size=16)
        bina = score_predictions(binary, truths, device=DEVICE, batch_size=16)
        output["results"][str(gain)] = {
            "continuous": summarize(cont),
            "binary": summarize(bina),
            "image_std": float(clipped.std()),
            "foreground_fraction": float(binary.mean()),
            "row_norm_mean": float(np.linalg.norm(coeffs * gain, axis=1).mean()),
        }
        print(gain, output["results"][str(gain)]["continuous"]["mean"],
              output["results"][str(gain)]["binary"]["mean"], flush=True)

    output["gt_reference"] = {
        "image_std": float((np.asarray(dct_decode(np.asarray(
            __import__("src.ridge", fromlist=["dct_encode"]).dct_encode(truths[:64], 32)), 32))).std()),
        "foreground_fraction": float(truths.mean()),
    }
    (OUT / "gain_scan.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote gain_scan.json", flush=True)


if __name__ == "__main__":
    main()
