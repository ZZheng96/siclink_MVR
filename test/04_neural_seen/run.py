"""P1 Tasks 2/3: nonlinear 2,748-way spike classifier with crisp glyph output."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import SpikeClassifier, fit_classifier, load_classifier, predict_logits
from src.retrieval import unique_assignment
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def fit_session_normalizer(features):
    transformed = np.log1p(np.maximum(features, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-5)


def apply_session_normalizer(features, normalizer):
    mean, scale = normalizer
    return (np.log1p(np.maximum(features, 0)) - mean) / scale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task2", "task3"))
    parser.add_argument("--name")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--mouse", choices=("mice1", "mice2"))
    parser.add_argument("--pool-sessions", action="store_true", help="Pool Task 3 sessions per mouse.")
    parser.add_argument("--session-norm", action="store_true", help="Use unlabeled per-session log-count statistics.")
    parser.add_argument("--unique-assignment", action="store_true", help="Also evaluate one-to-one assignment per block.")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--tune-pictures", type=int, default=256)
    parser.add_argument("--evaluation-pictures", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or (f"{args.task}_mlp_h{args.hidden}_e{args.epochs}"
                         + ("_pooled" if args.pool_sessions else "")
                         + ("_sessionnorm" if args.session_norm else "")
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
    output_dir = ROOT / "test/04_neural_seen"
    artifacts = output_dir / "artifacts" / name
    artifacts.mkdir(parents=True, exist_ok=True)

    grouped = ({mouse: [block for block in blocks if block.mouse == mouse] for mouse in {b.mouse for b in blocks}}
               if args.task == "task2" or args.pool_sessions else {block.key: [block] for block in blocks})
    prediction_rows = []
    training_rows = []
    for group, group_blocks in grouped.items():
        train_x_parts = []
        train_y_parts = []
        validation_x_parts = []
        validation_y_parts = []
        validation_keys = []
        for block in group_blocks:
            block_split = split.blocks[block.key]
            if args.task == "task2":
                if len(block_split.train_trials):
                    single_x = block.features[block_split.train_trials]
                    single_targets = block.trial_targets[block_split.train_trials]
                    mean_targets, mean_x = aggregate_by_target(single_x, single_targets)
                    if args.session_norm:
                        normalizer = fit_session_normalizer(single_x)
                        single_x = apply_session_normalizer(single_x, normalizer)
                        mean_x = apply_session_normalizer(mean_x, normalizer)
                    train_x_parts.extend((single_x, mean_x))
                    train_y_parts.extend((single_targets, mean_targets))
                if len(block_split.validation_trials):
                    validation_single_x = block.features[block_split.validation_trials]
                    validation_single_targets = block.trial_targets[block_split.validation_trials]
                    validation_targets, validation_x = aggregate_by_target(
                        validation_single_x, validation_single_targets,
                    )
                    if args.session_norm:
                        normalizer = fit_session_normalizer(validation_single_x)
                        validation_x = apply_session_normalizer(validation_x, normalizer)
                    validation_x_parts.append(validation_x)
                    validation_y_parts.append(validation_targets)
                    validation_keys.extend([block.key] * len(validation_targets))
            else:
                train_block_x = block.features[block_split.train_trials]
                validation_block_x = block.features[block_split.validation_trials]
                if args.session_norm:
                    normalizer = fit_session_normalizer(train_block_x)
                    train_block_x = apply_session_normalizer(train_block_x, normalizer)
                    validation_block_x = apply_session_normalizer(validation_block_x, normalizer)
                train_x_parts.append(train_block_x)
                train_y_parts.append(block.trial_targets[block_split.train_trials])
                validation_x_parts.append(validation_block_x)
                validation_targets = block.trial_targets[block_split.validation_trials]
                validation_y_parts.append(validation_targets)
                validation_keys.extend([block.key] * len(validation_targets))
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_y_parts)
        validation_x = np.concatenate(validation_x_parts)
        validation_targets = np.concatenate(validation_y_parts)
        tune_mask = np.isin(validation_targets, tune_pictures)
        model = SpikeClassifier(
            n_features=train_x.shape[1], n_classes=len(images), hidden=args.hidden, dropout=args.dropout,
            apply_log=not args.session_norm,
        )
        group_output = artifacts / group
        history = fit_classifier(
            model,
            (train_x, train_targets),
            (validation_x[tune_mask], validation_targets[tune_mask]),
            group_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=42,
        )
        best = load_classifier(group_output / "classifier.pt", device)
        logits = predict_logits(best, validation_x, batch_size=args.batch_size, device=device)
        predicted_targets = logits.argmax(axis=1)
        assigned_targets = predicted_targets.copy()
        validation_keys_array = np.asarray(validation_keys)
        if args.unique_assignment:
            for key in dict.fromkeys(validation_keys):
                indices = np.flatnonzero(validation_keys_array == key)
                if len(indices) != len(images):
                    raise ValueError(f"Unique assignment requires all {len(images)} characters in {key}.")
                assigned_targets[indices] = unique_assignment(logits[indices])
        prediction_rows.append({
            "predicted_targets": predicted_targets,
            "assigned_targets": assigned_targets,
            "targets": validation_targets,
            "groups": validation_keys_array,
        })
        best_row = max(history, key=lambda row: row["validation_accuracy"])
        training_rows.append({"group": group, "best": best_row, "epochs_run": len(history)})
        print(f"trained {group}: best={best_row}", flush=True)
        del model, best, logits
        if device == "cuda":
            torch.cuda.empty_cache()

    predicted_targets = np.concatenate([row["predicted_targets"] for row in prediction_rows])
    targets = np.concatenate([row["targets"] for row in prediction_rows])
    groups = np.concatenate([row["groups"] for row in prediction_rows])
    tune_mask = np.isin(targets, tune_pictures)
    evaluation_mask = np.isin(targets, evaluation_pictures)
    tune_accuracy = float((predicted_targets[tune_mask] == targets[tune_mask]).mean())
    evaluation_accuracy = float((predicted_targets[evaluation_mask] == targets[evaluation_mask]).mean())
    predictions = images[predicted_targets[evaluation_mask]]
    references = images[targets[evaluation_mask]]
    evaluation_groups = groups[evaluation_mask]
    scores = score_predictions(predictions, references, device=device, batch_size=args.ocf_batch_size)
    summary = summarize_scores(scores, evaluation_groups)
    summary["top1_accuracy"] = evaluation_accuracy
    summary["tune_top1_accuracy"] = tune_accuracy
    summary["group_top1_accuracy"] = {
        group: float((predicted_targets[evaluation_mask & (groups == group)] == targets[evaluation_mask & (groups == group)]).mean())
        for group in sorted(set(evaluation_groups.tolist()))
    }
    assignment_summary = None
    if args.unique_assignment:
        assigned_targets = np.concatenate([row["assigned_targets"] for row in prediction_rows])
        assignment_predictions = images[assigned_targets[evaluation_mask]]
        assignment_scores = score_predictions(
            assignment_predictions, references, device=device, batch_size=args.ocf_batch_size
        )
        assignment_summary = summarize_scores(assignment_scores, evaluation_groups)
        assignment_summary["top1_accuracy"] = float(
            (assigned_targets[evaluation_mask] == targets[evaluation_mask]).mean()
        )
        assignment_summary["tune_top1_accuracy"] = float(
            (assigned_targets[tune_mask] == targets[tune_mask]).mean()
        )
        assignment_summary["group_top1_accuracy"] = {
            group: float((assigned_targets[evaluation_mask & (groups == group)] == targets[evaluation_mask & (groups == group)]).mean())
            for group in sorted(set(evaluation_groups.tolist()))
        }
        np.savez_compressed(
            artifacts / "assignment_predictions.npz",
            predictions=assignment_predictions.astype(np.uint8) * 255,
            targets=references.astype(np.uint8) * 255,
            group_ids=evaluation_groups,
        )
    np.savez_compressed(
        artifacts / "evaluation_predictions.npz",
        predictions=predictions.astype(np.uint8) * 255,
        targets=references.astype(np.uint8) * 255,
        group_ids=evaluation_groups,
    )
    result = {
        "experiment": "04_neural_seen",
        "name": name,
        "config": vars(args),
        "protocol": {
            **split.metadata(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "task2_training": "single trials plus within-session four-repeat means",
            "task2_validation": "four-repeat mean counts",
            "session_normalization": "log-count mean/std per session; validation day uses unlabeled spikes only" if args.session_norm else None,
        },
        "device": device,
        "training": training_rows,
        "evaluation": summary,
        "evaluation_assignment": assignment_summary,
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / f"results_{name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assignment_text = (f"; assignment_OCF={assignment_summary['ocf_mean']:.6f}; "
                       f"assignment_top1={assignment_summary['top1_accuracy']:.6f}") if assignment_summary else ""
    print(f"FINAL {result_path}: OCF={summary['ocf_mean']:.6f}; top1={evaluation_accuracy:.6f}{assignment_text}", flush=True)


if __name__ == "__main__":
    main()
