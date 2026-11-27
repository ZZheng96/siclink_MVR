"""Diagnose whether the v5 Task1 DCT regression reconstructs usable images."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.features import aggregate_by_target
from src.neural import load_regressor, predict_logits
from src.preprocessing import load_cached_blocks
from src.ridge import dct_decode, dct_encode
from src.splits import make_split

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
MODELS = ROOT / "test/.cache/evaluate_final_task1_v2/models/task1"
OUT = ROOT / "test/01_task1_reconstruction/57_task1_continuous_gap"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stats(name, array):
    array = np.asarray(array, dtype=np.float64)
    flat = array.reshape(len(array), -1) if array.ndim > 1 else array.reshape(-1, 1)
    row_norm = np.linalg.norm(flat, axis=1)
    return {
        "name": name,
        "shape": list(np.shape(array)),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "std_overall": float(array.std()),
        "row_norm_mean": float(row_norm.mean()),
        "row_norm_p10": float(np.percentile(row_norm, 10)),
        "row_norm_p90": float(np.percentile(row_norm, 90)),
    }


def main():
    blocks = load_cached_blocks(CACHE, split="train", bin_ms=25)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)
    split = make_split(blocks, "task1", seed=42)
    heldout = np.sort(np.random.default_rng(123).permutation(split.validation_pictures)[64:])
    models = {m: load_regressor(MODELS / m / "regressor.pt", device=DEVICE) for m in ("mice1", "mice2")}

    target_pictures = heldout[:64]
    gt_coeff = dct_encode(gt[target_pictures], 32)
    gt_recon = dct_decode(gt_coeff, 32)

    records = {"coeff_gt": gt_coeff, "coeff_pred": [], "recon_pred": [], "recon_gt": gt_recon}
    for block in blocks:
        part = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            np.asarray(block.features[part.validation_trials], dtype=np.float32),
            block.trial_targets[part.validation_trials],
        )
        keep = np.isin(targets, target_pictures)
        counts = counts[keep]
        coefficients = predict_logits(models[block.mouse], counts, batch_size=256, device=DEVICE)
        records["coeff_pred"].append(np.asarray(coefficients, dtype=np.float32))
        records["recon_pred"].append(np.asarray(dct_decode(coefficients, 32), dtype=np.float32))
    pred_coeff = np.concatenate(records["coeff_pred"])
    pred_recon = np.concatenate(records["recon_pred"])
    gt_coeff_tiled = np.tile(gt_coeff, (len(pred_coeff) // len(gt_coeff), 1))
    gt_recon_tiled = np.tile(gt_recon, (len(pred_recon) // len(gt_recon), 1, 1))

    output = {
        "scope": "exploratory; v5 Task1 frozen regressors; 64 held-out pictures x 8 blocks",
        "gt_dct32_coefficients": stats("gt_coeff", gt_coeff),
        "pred_dct32_coefficients": stats("pred_coeff", pred_coeff),
        "gt_dct32_roundtrip_image": stats("gt_recon", gt_recon),
        "pred_dct32_image": stats("pred_recon", pred_recon),
        "coefficient_mse": float(np.mean((pred_coeff - gt_coeff_tiled) ** 2)),
        "coefficient_gt_variance": float(gt_coeff.var()),
        "image_mse": float(np.mean((pred_recon - gt_recon_tiled) ** 2)),
        "pred_image_nonnegative_fraction": float((pred_recon >= 0).mean()),
        "pred_image_gt_half_fraction": float((pred_recon >= 0.5).mean()),
        "gt_image_foreground_fraction": float((gt_recon_tiled >= 0.5).mean()),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    (OUT / "dct_scale_check.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
