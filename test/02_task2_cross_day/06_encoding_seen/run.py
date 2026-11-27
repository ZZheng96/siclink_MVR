"""P1 Tasks 2/3: forward image-to-spike encoding and candidate-response matching."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.retrieval import unique_assignment
from src.ridge import RidgePath, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--tune-pictures", type=int, default=256)
    parser.add_argument("--evaluation-pictures", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or f"{args.task}_encoding_dct{args.cutoff}"
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, args.cutoff)
    blocks = load_feature_blocks(DATA, "train", bin_ms=args.bin_ms, mouse=args.mouse)
    split = make_split(blocks, args.task, seed=42)
    rng = np.random.default_rng(123)
    selected = rng.permutation(len(images))
    tune_pictures = np.sort(selected[:args.tune_pictures])
    evaluation_pictures = np.sort(selected[args.tune_pictures:args.tune_pictures + args.evaluation_pictures])
    grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    rows = []
    alphas = (0.01, 0.1, 1.0, 10.0)

    for group, group_blocks in grouped.items():
        train_neural_parts = []
        train_target_parts = []
        validation_rows = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            if len(block_split.train_trials):
                targets, neural = aggregate_by_target(
                    block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
                )
                train_neural_parts.append(np.log1p(neural))
                train_target_parts.append(targets)
            if len(block_split.validation_trials):
                if args.task == "task2":
                    targets, neural = aggregate_by_target(
                        block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
                    )
                else:
                    neural = block.features[block_split.validation_trials]
                    targets = block.trial_targets[block_split.validation_trials]
                validation_rows.append((block.key, np.log1p(neural), targets))
        train_neural = np.concatenate(train_neural_parts)
        train_targets = np.concatenate(train_target_parts)
        path = RidgePath().fit(bank_coefficients[train_targets], train_neural)
        for alpha in alphas:
            prototypes = path.predict(bank_coefficients, alpha)
            for key, validation_x, validation_targets in validation_rows:
                for metric in ("mse", "cosine"):
                    scores = response_scores(validation_x, prototypes, metric=metric, device=device)
                    rows.append({
                        "group": group,
                        "block": key,
                        "alpha": alpha,
                        "metric": metric,
                        "predicted": scores.argmax(axis=1),
                        "assigned": unique_assignment(scores),
                        "targets": validation_targets,
                    })
        print(f"fit {group}: train={len(train_neural)}", flush=True)

    def tuning(field):
        results = []
        for alpha in alphas:
            for metric in ("mse", "cosine"):
                correct = 0
                count = 0
                for row in rows:
                    if row["alpha"] != alpha or row["metric"] != metric:
                        continue
                    mask = np.isin(row["targets"], tune_pictures)
                    correct += int((row[field][mask] == row["targets"][mask]).sum())
                    count += int(mask.sum())
                results.append({"alpha": alpha, "metric": metric, "top1_accuracy": correct / count})
        return results

    independent_tuning = tuning("predicted")
    assignment_tuning = tuning("assigned")
    best_independent = max(independent_tuning, key=lambda row: row["top1_accuracy"])
    best_assignment = max(assignment_tuning, key=lambda row: row["top1_accuracy"])

    def evaluate(config, field):
        predicted_parts = []
        target_parts = []
        group_parts = []
        for row in rows:
            if row["alpha"] != config["alpha"] or row["metric"] != config["metric"]:
                continue
            mask = np.isin(row["targets"], evaluation_pictures)
            predicted_parts.append(row[field][mask])
            target_parts.append(row["targets"][mask])
            group_parts.extend([row["block"]] * int(mask.sum()))
        predicted = np.concatenate(predicted_parts)
        targets = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        predictions = images[predicted]
        references = images[targets]
        scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
        summary = summarize_scores(scores, groups)
        summary["top1_accuracy"] = float((predicted == targets).mean())
        summary["group_top1_accuracy"] = {
            group: float((predicted[groups == group] == targets[groups == group]).mean())
            for group in sorted(set(groups.tolist()))
        }
        return summary, predictions, references, groups

    independent, predictions, references, groups = evaluate(best_independent, "predicted")
    assignment, assigned_predictions, _, _ = evaluate(best_assignment, "assigned")
    output_dir = ROOT / "test/02_task2_cross_day/06_encoding_seen"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    np.savez_compressed(
        artifacts / "assignment_predictions.npz",
        predictions=assigned_predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    result = {
        "experiment": "06_encoding_seen",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "pooling": "Per mouse across training sessions",
        },
        "independent_tuning": independent_tuning,
        "assignment_tuning": assignment_tuning,
        "selected_independent": best_independent,
        "selected_assignment": best_assignment,
        "evaluation": independent,
        "evaluation_assignment": assignment,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {result_path}: independent={independent['ocf_mean']:.6f}/"
        f"{independent['top1_accuracy']:.6f}; assignment={assignment['ocf_mean']:.6f}/"
        f"{assignment['top1_accuracy']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
