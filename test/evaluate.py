"""Evaluate a local predictions.npz with the official OCF implementation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions, summarize_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="NPZ with predictions, targets, and optional group_ids.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        if not {"predictions", "targets"} <= set(data.files):
            raise ValueError("NPZ must contain predictions and targets.")
        predictions = data["predictions"]
        targets = data["targets"]
        group_ids = data["group_ids"] if "group_ids" in data else None
    scores = score_predictions(
        predictions, targets, threshold=args.threshold, device=args.device, batch_size=args.batch_size
    )
    result = summarize_scores(scores, group_ids)
    result.update({
        "source": str(args.predictions),
        "threshold": args.threshold,
        "device": args.device,
        "metric": "official_ocf_png_parity",
    })
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
