"""End-to-end public-split evaluation of the frozen production pipeline with official OCF."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import load_feature_blocks
from src.final import DATA, file_sha256, predict_task, train_task
from src.splits import make_split

SPLIT_SEED = 42
PICTURE_SEED = 123
TASKS = ("task1", "task2", "task3")


def public_protocol(task, blocks, n_images):
    """Recreate the established public train/tune/heldout partition without test data."""
    split = make_split(blocks, task, seed=SPLIT_SEED)
    if task == "task1":
        shuffled = np.random.default_rng(PICTURE_SEED).permutation(split.validation_pictures)
        tune = np.sort(shuffled[:16])
        heldout = np.sort(shuffled[64:])
        unused = np.sort(shuffled[16:64])
        rule = (
            "make_split task1 seed42 character holdout; RNG123 validation-picture permutation: "
            "first16 tune, next48 unused, remaining211 heldout"
        )
    else:
        order = np.random.default_rng(PICTURE_SEED).permutation(n_images)
        tune = np.sort(order[:256])
        heldout = np.sort(order[256:768])
        unused = np.empty(0, dtype=np.int64)
        rule = (
            "make_split seed42; RNG123 all-picture permutation: first256 tune, next512 heldout; "
            "Task2 holds out session4 and Task3 holds out one trial per picture"
        )
    if not len(tune) or not len(heldout):
        raise ValueError(f"Incomplete fixed public partition for {task}.")
    if np.intersect1d(tune, heldout).size or np.intersect1d(tune, unused).size:
        raise RuntimeError("Tune, heldout, and unused picture partitions overlap.")
    if np.intersect1d(heldout, unused).size:
        raise RuntimeError("Heldout and unused picture partitions overlap.")
    return split, {"tune": tune, "heldout": heldout}, {
        "data_scope": "public train blocks only; hidden test blocks and targets are never read",
        "split_seed": SPLIT_SEED,
        "picture_seed": PICTURE_SEED,
        "rule": rule,
        "tune_pictures": tune.tolist(),
        "heldout_pictures": heldout.tolist(),
        "unused_validation_pictures": unused.tolist(),
        "alignment": "direct public FeatureBlock row order; no sample-ID join or hidden-GT alignment",
        "assignment_scope": "production full validation block before picture subsetting",
        "heldout_used_for_selection_in_this_script": False,
        "heldout_historically_reused": True,
        "heldout_used_for_prior_selection": True,
        "heldout_is_pristine_confirmatory": False,
        "heldout_disclosure": (
            "No selection occurs in this script: src/final_config.json is frozen before tuning or "
            "heldout scoring. The same public heldout IDs were evaluated in earlier experiments and "
            "informed candidate design, so this is a historical reuse evaluation, not pristine confirmation."
        ),
    }


def check_training_scope(task, config, blocks, split):
    records = {}
    for block in blocks:
        part = split.blocks[block.key]
        records[block.key] = {
            "train_trials": len(part.train_trials),
            "validation_trials": len(part.validation_trials),
        }
        if np.intersect1d(part.train_trials, part.validation_trials).size:
            raise RuntimeError(f"Train/validation trial leakage in {block.key}.")
    if task == "task2" and config[task].get("reranker", {}).get("enabled", False):
        train_sessions = {
            block.session for block in blocks if len(split.blocks[block.key].train_trials)
        }
        oof_sessions = set(config[task]["reranker"]["oof_sessions"])
        if not oof_sessions <= train_sessions:
            raise ValueError("Task2 reranker OOF sessions exceed the training-only sessions.")
    return records


def official_evaluate(predictions, targets, groups, device, batch_size):
    scores = score_predictions(predictions, targets, device=device, batch_size=batch_size)
    result = summarize_scores(scores, groups)
    result["metric"] = "official_ocf_png_parity"
    return result


def oracle_predictions(prediction, mask, images, device, batch_size):
    """Choose the best public-GT OCF candidate independently of the untouched final arm."""
    candidates = prediction["candidate_indices"][mask]
    target_indices = prediction["targets"][mask]
    targets = images[target_indices]
    bank = prediction["candidate_bank"]
    width = candidates.shape[1]
    output = []
    for start in range(0, len(candidates), batch_size):
        candidate_part = candidates[start:start + batch_size]
        target_part = targets[start:start + batch_size]
        flat_candidates = candidate_part.reshape(-1)
        candidate_images = bank[flat_candidates]
        repeated_targets = np.repeat(target_part, width, axis=0)
        scores = score_predictions(
            candidate_images, repeated_targets, device=device, batch_size=batch_size
        ).reshape(len(candidate_part), width)
        best = scores.argmax(axis=1)
        selected = candidate_part[np.arange(len(candidate_part)), best]
        output.append(bank[selected].copy())
    return np.concatenate(output).astype(np.float32)


def save_public_npz(path, predictions, targets, groups, target_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=np.rint(predictions).astype(np.uint8) * 255,
        targets=np.rint(targets).astype(np.uint8) * 255,
        group_ids=groups,
        target_indices=target_indices,
    )


def evaluate_task(task, config, output, *, device, ocf_batch_size=16, oracle_top_k=32,
                  images=None, blocks=None):
    started = time.perf_counter()
    if images is None:
        images = (load_images(DATA) >= 128).astype(np.float32)
    if blocks is None:
        blocks = load_feature_blocks(DATA, "train", bin_ms=config[task]["bin_ms"])
    split, partitions, protocol = public_protocol(task, blocks, len(images))
    protocol["blocks"] = check_training_scope(task, config, blocks, split)
    models = Path(output) / "models"
    task_output = Path(output) / task
    task_output.mkdir(parents=True, exist_ok=True)

    training = train_task(
        config, task, models, images=images, blocks=blocks, split=split, device=device
    )
    (task_output / "training_summary.json").write_text(
        json.dumps(training, indent=2, allow_nan=False), encoding="utf-8"
    )
    reranker_checkpoint = None
    if task == "task2" and config[task].get("reranker", {}).get("enabled", False):
        reranker_checkpoint = models / "listwise_cnn.pt"
    prediction = predict_task(
        config,
        task,
        models,
        images=images,
        blocks=blocks,
        split=split,
        device=device,
        reranker_checkpoint=reranker_checkpoint,
        candidate_top_n=oracle_top_k,
        manifest_output=task_output / "reranker_load_manifest.json",
    )
    result = {
        "task": task,
        "production_entrypoints": ["src.final.train_task", "src.final.predict_task"],
        "protocol": protocol,
        "training": training,
        "prediction": {
            "rows": len(prediction["predictions"]),
            "reranker": prediction["reranker"],
            "oracle_top_k": oracle_top_k,
            "oracle_definition": (
                "per-row best official OCF among production pre-assignment ranked candidates; "
                "public-label diagnostic only, never used to alter final predictions or select configuration"
            ),
        },
        "tune": {},
        "heldout": {},
    }
    for partition_name, pictures in partitions.items():
        mask = np.isin(prediction["targets"], pictures)
        if task == "task3":
            expected = len(pictures) * len(blocks)
        else:
            expected = sum(
                len(np.intersect1d(
                    np.unique(block.trial_targets[split.blocks[block.key].validation_trials]),
                    pictures,
                ))
                for block in blocks
            )
        if mask.sum() != expected or set(prediction["targets"][mask]) != set(pictures.tolist()):
            raise RuntimeError(f"Unexpected {partition_name} rows for {task}.")
        final = prediction["predictions"][mask].copy()
        targets = images[prediction["targets"][mask]]
        groups = prediction["groups"][mask]
        target_indices = prediction["targets"][mask]
        final_metrics = official_evaluate(final, targets, groups, device, ocf_batch_size)
        save_public_npz(
            task_output / f"{partition_name}_final.npz",
            final, targets, groups, target_indices,
        )
        result[partition_name]["final"] = final_metrics

        oracle = oracle_predictions(prediction, mask, images, device, ocf_batch_size)
        oracle_metrics = official_evaluate(oracle, targets, groups, device, ocf_batch_size)
        save_public_npz(
            task_output / f"{partition_name}_oracle.npz",
            oracle, targets, groups, target_indices,
        )
        result[partition_name]["oracle"] = oracle_metrics
        print(
            f"{task} {partition_name}: final OCF={final_metrics['ocf_mean']:.6f}; "
            f"oracle@{oracle_top_k} OCF={oracle_metrics['ocf_mean']:.6f}",
            flush=True,
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/final_config.json")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=ROOT / "test/.cache/evaluate_final")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ocf-batch-size", type=int, default=16)
    parser.add_argument("--oracle-top-k", type=int, default=32)
    args = parser.parse_args()
    if args.oracle_top_k < 1:
        parser.error("--oracle-top-k must be positive; final and oracle are reported separately.")
    if args.ocf_batch_size < 1:
        parser.error("--ocf-batch-size must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty so models and metrics from different runs cannot mix.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = {
        "name": config["name"],
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "scope": "fixed public split; end-to-end fixed-epoch training, production prediction, official OCF",
        "selection": "none; the production configuration is frozen before this evaluation",
        "tasks": {},
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
        },
    }
    for task in args.tasks:
        result["tasks"][task] = evaluate_task(
            task,
            config,
            args.output,
            device=args.device,
            ocf_batch_size=args.ocf_batch_size,
            oracle_top_k=args.oracle_top_k,
        )
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
