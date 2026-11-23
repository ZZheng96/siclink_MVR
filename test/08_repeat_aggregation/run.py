"""P2: mean versus median repeat aggregation using frozen best checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import load_classifier, load_regressor, predict_logits
from src.retrieval import retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task1", "task2"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    bin_ms = 25 if args.task == "task1" else 50
    blocks = load_feature_blocks(DATA, "train", bin_ms=bin_ms)
    split = make_split(blocks, args.task, seed=42)
    if args.task == "task1":
        shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
        evaluation_pictures = np.sort(shuffled[64:])
        model_root = ROOT / "test/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
    else:
        selected = np.random.default_rng(123).permutation(len(images))
        evaluation_pictures = np.sort(selected[256:768])
        model_root = ROOT / "test/04_neural_seen/artifacts/task2_mlp_bin50_sessionnorm_assign"
    output_dir = ROOT / "test/08_repeat_aggregation"

    for method in ("mean", "median"):
        prediction_parts = []
        target_parts = []
        group_parts = []
        for block in blocks:
            block_split = split.blocks[block.key]
            if not len(block_split.validation_trials):
                continue
            raw = block.features[block_split.validation_trials]
            raw_targets = block.trial_targets[block_split.validation_trials]
            targets, counts = aggregate_by_target(raw, raw_targets, method=method)
            mask = np.isin(targets, evaluation_pictures)
            if args.task == "task1":
                model = load_regressor(model_root / block.mouse / "regressor.pt", device)
                coefficients = predict_logits(model, counts, device=device)
                continuous = dct_decode(coefficients, 32)
                bank_indices = np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))
                predictions, _ = retrieve_glyphs(continuous, images[bank_indices], metric="directional", device=device)
            else:
                normalizer = fit_session_normalizer(raw)
                counts = apply_session_normalizer(counts, normalizer)
                model = load_classifier(model_root / block.mouse / "classifier.pt", device)
                predictions = images[predict_logits(model, counts, device=device).argmax(axis=1)]
            prediction_parts.append(predictions[mask])
            target_parts.append(images[targets[mask]])
            group_parts.extend([block.key] * int(mask.sum()))
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        result = {
            "experiment": "08_repeat_aggregation",
            "name": f"{args.task}_repeat_{method}",
            "config": {"task": args.task, "aggregation": method, "bin_ms": bin_ms},
            "protocol": {"evaluation_pictures": evaluation_pictures.tolist(), "frozen_models": str(model_root)},
            "evaluation": summary,
            "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        }
        path = output_dir / f"results_{args.task}_repeat_{method}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"FINAL {path}: {summary['ocf_mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
