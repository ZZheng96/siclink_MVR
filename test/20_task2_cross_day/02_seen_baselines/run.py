"""P0 Tasks 2/3: seen-character neural centroid classification and crisp retrieval."""

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
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.retrieval import CentroidClassifier, unique_assignment
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--transform", choices=("raw", "log"), default="log")
    parser.add_argument("--session-norm", action="store_true")
    parser.add_argument("--unique-assignment", action="store_true")
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--tune-pictures", type=int, default=256)
    parser.add_argument("--evaluation-pictures", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def normalize(features):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    return (features - mean) / np.maximum(scale, 1e-6)


def main():
    args = parse_args()
    name = args.name or (f"{args.task}_centroid_{args.transform}" + ("_sessionnorm" if args.session_norm else "")
                         + ("_assign" if args.unique_assignment else ""))
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=args.bin_ms, mouse=args.mouse)
    split = make_split(blocks, args.task, seed=42)
    rng = np.random.default_rng(123)
    selected = rng.permutation(len(images))
    tune_pictures = np.sort(selected[:args.tune_pictures])
    evaluation_pictures = np.sort(selected[args.tune_pictures:args.tune_pictures + args.evaluation_pictures])
    if len(evaluation_pictures) < args.evaluation_pictures:
        raise ValueError("Not enough pictures for tuning and evaluation.")

    if args.task == "task2":
        grouped = {mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
    else:
        grouped = {block.key: [block] for block in blocks}

    rows = []
    for group, group_blocks in grouped.items():
        train_x_parts = []
        train_y_parts = []
        validation_rows = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            if args.task == "task2":
                if len(block_split.validation_trials):
                    validation_targets, validation_x = aggregate_by_target(
                        block.features[block_split.validation_trials],
                        block.trial_targets[block_split.validation_trials],
                    )
                else:
                    validation_x = np.empty((0, block.n_features), dtype=np.float32)
                    validation_targets = np.empty(0, dtype=block.targets.dtype)
                if len(block_split.train_trials):
                    train_targets, train_x = aggregate_by_target(
                        block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
                    )
                    if args.transform == "log":
                        train_x = np.log1p(train_x)
                    if args.session_norm:
                        train_x = normalize(train_x)
                    train_x_parts.append(train_x)
                    train_y_parts.append(train_targets)
            else:
                validation_x = block.features[block_split.validation_trials]
                validation_targets = block.trial_targets[block_split.validation_trials]
                train_x = block.features[block_split.train_trials]
                train_targets = block.trial_targets[block_split.train_trials]
                if args.transform == "log":
                    train_x = np.log1p(train_x)
                train_x_parts.append(train_x)
                train_y_parts.append(train_targets)
            if len(validation_targets):
                if args.transform == "log":
                    validation_x = np.log1p(validation_x)
                if args.session_norm:
                    validation_x = normalize(validation_x)
                validation_rows.append((block.key, validation_x, validation_targets))
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_y_parts)
        for metric in ("cosine", "euclidean"):
            classifier = CentroidClassifier(metric=metric).fit(train_x, train_targets)
            for key, validation_x, validation_targets in validation_rows:
                predictions = classifier.predict(validation_x)
                assigned = predictions.copy()
                if args.unique_assignment:
                    if len(validation_targets) != len(images) or len(classifier.classes) != len(images):
                        raise ValueError("Unique assignment requires every character and the full image bank.")
                    assigned = classifier.classes[unique_assignment(classifier.decision_function(validation_x))]
                rows.append({
                    "group": group,
                    "block": key,
                    "metric": metric,
                    "predicted_targets": predictions,
                    "assigned_targets": assigned,
                    "targets": validation_targets,
                })
        print(f"fit {group}: train={len(train_x)} validation={sum(len(r[2]) for r in validation_rows)}", flush=True)

    def collect(metric, pictures, field="predicted_targets"):
        predicted_targets = []
        targets = []
        groups = []
        for row in rows:
            if row["metric"] != metric:
                continue
            mask = np.isin(row["targets"], pictures)
            predicted_targets.append(row[field][mask])
            targets.append(row["targets"][mask])
            groups.extend([row["block"]] * int(mask.sum()))
        predicted_targets = np.concatenate(predicted_targets)
        targets = np.concatenate(targets)
        accuracy = float((predicted_targets == targets).mean())
        return predicted_targets, targets, np.asarray(groups), accuracy

    tuning = {}
    for metric in ("cosine", "euclidean"):
        _, _, _, accuracy = collect(metric, tune_pictures)
        tuning[metric] = {"top1_accuracy": accuracy}
    selected_metric = max(tuning, key=lambda metric: tuning[metric]["top1_accuracy"])
    predicted_targets, targets, groups, accuracy = collect(selected_metric, evaluation_pictures)
    predictions = images[predicted_targets]
    references = images[targets]
    scores = score_predictions(predictions, references, device=device, batch_size=args.batch_size)
    summary = summarize_scores(scores, groups)
    summary["top1_accuracy"] = accuracy
    group_accuracies = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_accuracies[group] = float((predicted_targets[mask] == targets[mask]).mean())
    summary["group_top1_accuracy"] = group_accuracies

    assignment_tuning = None
    selected_assignment_metric = None
    assignment_summary = None
    assignment_predictions = None
    if args.unique_assignment:
        assignment_tuning = {}
        for metric in ("cosine", "euclidean"):
            _, _, _, assignment_accuracy = collect(metric, tune_pictures, "assigned_targets")
            assignment_tuning[metric] = {"top1_accuracy": assignment_accuracy}
        selected_assignment_metric = max(
            assignment_tuning, key=lambda metric: assignment_tuning[metric]["top1_accuracy"]
        )
        assigned_targets, assigned_true, assigned_groups, assignment_accuracy = collect(
            selected_assignment_metric, evaluation_pictures, "assigned_targets"
        )
        assert np.array_equal(assigned_true, targets) and np.array_equal(assigned_groups, groups)
        assignment_predictions = images[assigned_targets]
        assignment_scores = score_predictions(
            assignment_predictions, references, device=device, batch_size=args.batch_size
        )
        assignment_summary = summarize_scores(assignment_scores, groups)
        assignment_summary["top1_accuracy"] = assignment_accuracy
        assignment_summary["group_top1_accuracy"] = {
            group: float((assigned_targets[groups == group] == targets[groups == group]).mean())
            for group in sorted(set(groups.tolist()))
        }

    output_dir = ROOT / "test/20_task2_cross_day/02_seen_baselines"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=groups,
    )
    if assignment_summary is not None:
        np.savez_compressed(
            artifacts / "assignment_predictions.npz",
            predictions=assignment_predictions.astype(np.uint8) * 255,
            targets=references.astype(np.uint8) * 255,
            group_ids=groups,
        )
    result = {
        "experiment": "02_seen_baselines",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "note": "Task 2 uses whole-session holdout and four-repeat means; Task 3 uses one held-out trial per sample.",
        },
        "device": device,
        "tuning": tuning,
        "selected_metric": selected_metric,
        "evaluation": summary,
        "assignment_tuning": assignment_tuning,
        "selected_assignment_metric": selected_assignment_metric,
        "evaluation_assignment": assignment_summary,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assignment_text = (f"; assignment_OCF={assignment_summary['ocf_mean']:.6f}; "
                       f"assignment_top1={assignment_summary['top1_accuracy']:.6f}") if assignment_summary else ""
    print(f"FINAL {result_path}: {summary['ocf_mean']:.6f} OCF; top1={accuracy:.6f}; metric={selected_metric}{assignment_text}", flush=True)


if __name__ == "__main__":
    main()
