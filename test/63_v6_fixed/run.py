"""Fixed public-protocol score of v6: v5 Task1 plus experiment-53 multiresolution Task2/3.

Task1 is unchanged from v5 and is re-scored from the frozen v5 heldout predictions.
Task2/3 replace the v5 classifier branches with the frozen 10/25/50 ms equal-weight
row-z-score ensemble, at the experiment-53 selected per-view epochs (7/5/7 and 7/12/9).
No weight, epoch or candidate search is performed here.
"""

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
from src.final import DATA, file_sha256
from src.preprocessing import load_cached_blocks
from src.splits import make_split
from src.v6 import VIEWS, predict_block, train_task

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
V5_HELDOUT = {
    "task1": ROOT / "test/.cache/evaluate_final_task1_v2/task1/heldout_final.npz",
    "task2": ROOT / "test/.cache/evaluate_final_task23_v1/task2/heldout_final.npz",
    "task3": ROOT / "test/.cache/evaluate_final_task23_v1/resumed/task3/heldout_final.npz",
}
V5_RECORDED = {
    "task1": 18.43706,
    "task2": 23.824141005605483,
    "task3": 16.7598316835003,
}
SPLIT_SEED = 42
PICTURE_SEED = 123


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/63_v6_fixed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Skip a task's training only when all of its fixed-split checkpoints already exist.",
    )
    return parser.parse_args()


def fixed_partitions(n_images):
    order = np.random.default_rng(PICTURE_SEED).permutation(n_images)
    return np.sort(order[:256]), np.sort(order[256:768])


def score_saved_heldout(path, device, batch_size):
    with np.load(path, allow_pickle=False) as data:
        predictions = data["predictions"].astype(np.float32) / 255
        targets = data["targets"].astype(np.float32) / 255
        groups = data["group_ids"]
    scores = score_predictions(predictions, targets, device=device, batch_size=batch_size)
    return summarize_scores(scores, groups)


def evaluate_multires(task, models, blocks_by_view, split, images, heldout, device, batch_size,
                      reuse_models=False):
    started = time.perf_counter()
    expected = [
        Path(models) / "v6" / task / mouse / f"{view}ms" / "classifier.pt"
        for view in VIEWS
        for mouse in sorted({b.mouse for b in blocks_by_view[view]})
    ]
    if reuse_models and all(path.exists() for path in expected):
        training = {"task": task, "reused_checkpoints": [str(p) for p in expected]}
        print(f"{task}: reusing {len(expected)} fixed-split checkpoints", flush=True)
    else:
        training = train_task(task, models, blocks_by_view, split=split, device=device)
    keys = sorted(split.blocks)
    indices_parts, target_parts, group_parts = [], [], []
    n_blocks = 0
    for key in keys:
        blocks = {view: next(b for b in blocks_by_view[view] if b.key == key) for view in VIEWS}
        trials = np.asarray(split.blocks[key].validation_trials, dtype=np.int64)
        if not len(trials):
            continue
        n_blocks += 1
        if task == "task2":
            labels = np.asarray(blocks[25].targets, dtype=np.int64)[
                blocks[25].trial_sample_indices[trials]
            ]
            row_targets, group_ids = np.unique(labels, return_inverse=True)
            if len(row_targets) != 2748 or not np.all(np.bincount(group_ids) == 4):
                raise RuntimeError(f"Unexpected repeat grouping for {key}.")
            out = predict_block(
                task, models, blocks, trial_indices=trials, group_ids=group_ids, device=device
            )
        else:
            row_targets = np.asarray(blocks[25].trial_targets[trials], dtype=np.int64)
            out = predict_block(task, models, blocks, trial_indices=trials, device=device)
        if len(out["indices"]) != len(row_targets):
            raise RuntimeError(f"Prediction/target row mismatch for {key}.")
        indices_parts.append(out["indices"])
        target_parts.append(row_targets)
        group_parts.append(np.asarray([key] * len(row_targets)))
        print(f"{task} predicted {key}: {len(row_targets)} rows", flush=True)
    indices = np.concatenate(indices_parts)
    target_indices = np.concatenate(target_parts)
    groups = np.concatenate(group_parts)
    mask = np.isin(target_indices, heldout)
    if mask.sum() != len(heldout) * n_blocks:
        raise RuntimeError(f"Unexpected heldout row count for {task}: {mask.sum()}.")
    predictions = images[indices[mask]]
    targets = images[target_indices[mask]]
    scores = score_predictions(predictions, targets, device=device, batch_size=batch_size)
    summary = summarize_scores(scores, groups[mask])
    summary["rows_scored"] = int(mask.sum())
    summary["assignment_scope"] = "full block before picture subsetting"
    summary["heldout_pictures"] = heldout.tolist()
    summary["elapsed_seconds"] = time.perf_counter() - started
    return training, summary, {
        "indices": indices[mask],
        "targets": target_indices[mask],
        "groups": groups[mask],
    }


def main():
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"{results_path} already exists; pass --overwrite to replace it.")
    started = time.perf_counter()
    images = (load_images(DATA) >= 128).astype(np.float32)
    tune, heldout = fixed_partitions(len(images))
    device = args.device
    results = {
        "name": "v6_multires_task23_v5_task1",
        "scope": "fixed public split; no hyperparameter, epoch, weight or candidate search",
        "protocol": {
            "split_seed": SPLIT_SEED,
            "picture_seed": PICTURE_SEED,
            "tune_pictures": tune.tolist(),
            "heldout_pictures": heldout.tolist(),
            "task1": "v5 frozen heldout predictions re-scored; unchanged",
            "task2": "leave session4 out; four-public-repeat mean rows; full-block assignment",
            "task3": "make_split seed42 single-trial validation; full-block assignment",
            "heldout_historically_reused": True,
            "heldout_is_pristine_confirmatory": False,
            "v6_sources": [
                "src/v6.py VIEWS=(10,25,50), EPOCHS task2 7/5/7 task3 7/12/9",
                "experiment 48/53 production recipe, equal row-z-score fusion",
                "experiment 53 selection.json multires_ensemble",
            ],
        },
        "v5_heldout_rescored": {},
        "v6_heldout": {},
    }
    for task, path in V5_HELDOUT.items():
        summary = score_saved_heldout(path, device, args.ocf_batch_size)
        results["v5_heldout_rescored"][task] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "ocf_mean": summary["ocf_mean"],
            "n": summary["n"],
            "groups": summary["groups"],
            "recorded_ocf_mean": V5_RECORDED[task],
            "reproduction_abs_diff": abs(summary["ocf_mean"] - V5_RECORDED[task]),
        }
        print(f"v5 {task}: {summary['ocf_mean']:.6f} (n={summary['n']})", flush=True)

    blocks_by_view = {
        view: load_cached_blocks(CACHE, split="train", bin_ms=view) for view in VIEWS
    }
    reference = blocks_by_view[25]
    split2 = make_split(reference, "task2", seed=SPLIT_SEED)
    split3 = make_split(reference, "task3", seed=SPLIT_SEED)
    models = output / "models"
    for task, split in (("task2", split2), ("task3", split3)):
        training, summary, audit = evaluate_multires(
            task, models, blocks_by_view, split, images, heldout, device, args.ocf_batch_size,
            reuse_models=args.reuse_models,
        )
        np.savez_compressed(
            output / f"{task}_heldout_v6.npz",
            indices=audit["indices"],
            targets=audit["targets"],
            group_ids=audit["groups"],
        )
        results["v6_heldout"][task] = {
            "ocf_mean": summary["ocf_mean"],
            "n": summary["n"],
            "groups": summary["groups"],
            "assignment_scope": summary["assignment_scope"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "training": training,
            "v5_delta": summary["ocf_mean"] - results["v5_heldout_rescored"][task]["ocf_mean"],
        }
        print(f"v6 {task}: {summary['ocf_mean']:.6f}", flush=True)

    for task, row in results["v5_heldout_rescored"].items():
        if row["reproduction_abs_diff"] > 1e-4:
            raise RuntimeError(f"v5 {task} did not reproduce its recorded OCF.")
    v5_mean = float(np.mean([results["v5_heldout_rescored"][t]["ocf_mean"]
                             for t in ("task1", "task2", "task3")]))
    v6_scores = {
        "task1": results["v5_heldout_rescored"]["task1"]["ocf_mean"],
        "task2": results["v6_heldout"]["task2"]["ocf_mean"],
        "task3": results["v6_heldout"]["task3"]["ocf_mean"],
    }
    v6_mean = float(np.mean(list(v6_scores.values())))
    results["summary"] = {
        "v5_heldout": {t: results["v5_heldout_rescored"][t]["ocf_mean"]
                       for t in ("task1", "task2", "task3")},
        "v5_three_task_mean": v5_mean,
        "v6_heldout": v6_scores,
        "v6_three_task_mean": v6_mean,
        "v6_minus_v5_mean": v6_mean - v5_mean,
        "task2_delta": v6_scores["task2"] - results["v5_heldout_rescored"]["task2"]["ocf_mean"],
        "task3_delta": v6_scores["task3"] - results["v5_heldout_rescored"]["task3"]["ocf_mean"],
        "acceptance_threshold_40_reached": v6_mean >= 40.0,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
        },
    }
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
