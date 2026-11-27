"""P0: calibrate the official OCF metric and binary PNG output path."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def main():
    rng = np.random.default_rng(42)
    bank = load_images(DATA)
    indices = rng.choice(len(bank), size=16, replace=False)
    raw = bank.astype(np.float32) / 255
    targets = raw >= 0.5
    selected = raw[indices]
    selected_targets = targets[indices]
    random_indices = (indices + 1 + rng.integers(0, len(bank) - 1, len(indices))) % len(bank)
    mean_glyph = np.repeat(raw.mean(axis=0, keepdims=True), len(indices), axis=0)
    cases = {
        "oracle_binary": (selected_targets.astype(np.float32), selected_targets, None),
        "oracle_raw_antialiased": (selected, selected_targets, None),
        "blank": (np.zeros_like(selected), selected_targets, None),
        "bank_mean_blur": (mean_glyph, selected_targets, None),
        "random_crisp_glyph": (targets[random_indices].astype(np.float32), selected_targets, None),
        "raw_threshold_128": (selected, selected_targets, 128 / 255),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, (predictions, references, threshold) in cases.items():
        scores = score_predictions(
            predictions, references, threshold=threshold, device=device, batch_size=16
        )
        results[name] = {"n": len(scores), "ocf_mean": float(scores.mean()), "ocf_std": float(scores.std())}
        print(name, results[name], flush=True)
    assert abs(results["oracle_binary"]["ocf_mean"] - 100) < 1e-8
    assert abs(results["blank"]["ocf_mean"]) < 1e-8
    assert abs(results["raw_threshold_128"]["ocf_mean"] - 100) < 1e-8
    output = {
        "experiment": "00_sanity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": ".venv/Scripts/python.exe -B test/00_infra/00_sanity/run.py",
        "device": device,
        "results": results,
        "status": "PASS",
    }
    (ROOT / "test/00_infra/00_sanity/results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
