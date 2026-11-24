"""Seed-only three-model prefix ensembles under the existing local protocols."""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.final import apply_session_normalizer, fit_session_normalizer
from src.neural import (
    SpikeClassifier,
    SpikeRegressor,
    fit_classifier,
    fit_regressor,
    load_classifier,
    load_regressor,
    predict_logits,
)
from src.retrieval import render_gb2312_bank, retrieve_glyphs, unique_assignment
from src.ridge import RidgePath, dct_decode, dct_encode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUTPUT_DIR = ROOT / "test/18_seed_ensemble"
ARTIFACTS = OUTPUT_DIR / "artifacts"
MODELS = ARTIFACTS / "models"
TASK1_SEED42_ROOT = ROOT / "test/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
SEEN_SEED42_ROOTS = {
    "task2": ROOT / "test/04_neural_seen/artifacts/task2_mlp_bin50_sessionnorm_assign",
    "task3": ROOT / "test/04_neural_seen/artifacts/task3_mlp_bin50_pooled_sessionnorm_assign",
}
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
SEEDS = (42, 202, 777)
PREFIX_SIZES = (1, 2, 3)
BASELINE_OCF = {
    "task1": 18.624308986308705,
    "task2": 23.329312169146995,
    "task3": 16.57463190021421,
}
BASELINE_ATOL = 1e-5
EPOCHS = {
    "task1": {"mice1": 8, "mice2": 11},
    "task2": {"mice1": 7, "mice2": 4},
    "task3": {"mice1": 7, "mice2": 7},
}
LEADERBOARD_REFERENCE = {"rank_1": 54.01486, "comparable": False}
KAGGLE_BASELINE = 21.09590


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def baseline_record(task, evaluation):
    error = abs(evaluation["ocf_mean"] - BASELINE_OCF[task])
    print(
        f"BASELINE {task} n=1: actual={evaluation['ocf_mean']:.10f}; "
        f"expected={BASELINE_OCF[task]:.10f}; abs_error={error:.3g}",
        flush=True,
    )
    if error > BASELINE_ATOL:
        raise RuntimeError(
            f"Baseline mismatch for {task}: actual={evaluation['ocf_mean']}, "
            f"expected={BASELINE_OCF[task]}, abs_error={error}, tolerance={BASELINE_ATOL}. "
            "Stopping this task before new-seed training or prefix tuning."
        )
    return {
        "expected_ocf": BASELINE_OCF[task],
        "actual_ocf": evaluation["ocf_mean"],
        "absolute_error": error,
        "absolute_tolerance": BASELINE_ATOL,
        "passed": True,
        "evaluation": evaluation,
    }


def prefix_record(prefix_n, summary):
    record = {
        "prefix_n": prefix_n,
        "seeds": list(SEEDS[:prefix_n]),
        "n": summary["n"],
        "ocf_mean": summary["ocf_mean"],
        "ocf_std": summary["ocf_std"],
    }
    if "top1_accuracy" in summary:
        record["top1_accuracy"] = summary["top1_accuracy"]
    return record


def result_envelope(task, protocol, config, training, baseline, tuning, selected, evaluation, selected_npz):
    result = {
        "experiment": "18_seed_ensemble",
        "name": f"{task}_seed_only_prefix_ensemble",
        "score_scope": "fixed local validation protocol only; not a Kaggle score",
        "config": {
            "task": task,
            "seeds": list(SEEDS),
            "prefix_sizes": list(PREFIX_SIZES),
            "seed42_checkpoint_reused_without_retraining": True,
            "selection_metric": "tune official OCF mean only",
            **config,
        },
        "protocol": protocol,
        "training": training,
        "baseline": baseline,
        "tuning": tuning,
        "selected": selected,
        "evaluation": evaluation,
        "evaluation_delta_ocf": evaluation["ocf_mean"] - baseline["actual_ocf"],
        "selected_npz": str(selected_npz.relative_to(ROOT)),
        "leaderboard_reference": LEADERBOARD_REFERENCE,
        "Kaggle_baseline": KAGGLE_BASELINE,
    }
    if task != "task1":
        result["evaluation_delta_top1_accuracy"] = (
            evaluation["top1_accuracy"] - baseline["evaluation"]["top1_accuracy"]
        )
    return result


def prepare_task1(images):
    blocks = load_feature_blocks(DATA, "train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    model_validation_pictures = np.sort(shuffled[:64])
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    rows = []
    training = {}
    for mouse in sorted({block.mouse for block in blocks}):
        train_x_parts = []
        train_target_parts = []
        validation_rows = []
        for block in [block for block in blocks if block.mouse == mouse]:
            block_split = split.blocks[block.key]
            train_targets, train_means = aggregate_by_target(
                block.features[block_split.train_trials], block.trial_targets[block_split.train_trials]
            )
            validation_targets, validation_x = aggregate_by_target(
                block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
            )
            train_x_parts.extend((train_means, block.features[block_split.train_trials]))
            train_target_parts.extend((train_targets, block.trial_targets[block_split.train_trials]))
            validation_rows.append((validation_x, validation_targets))
            rows.append({
                "block": block.key,
                "mouse": mouse,
                "targets": validation_targets,
                "counts": validation_x,
                "continuous": {},
            })
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_target_parts)
        tune_x = []
        tune_y = []
        for validation_x, validation_targets in validation_rows:
            mask = np.isin(validation_targets, model_validation_pictures)
            tune_x.append(validation_x[mask])
            tune_y.append(dct_encode(images[validation_targets[mask]], 32))
        training[mouse] = {
            "train": (train_x, dct_encode(images[train_targets], 32)),
            "validation": (np.concatenate(tune_x), np.concatenate(tune_y)),
        }
    return {
        "images": images,
        "rows": rows,
        "training": training,
        "model_validation_pictures": model_validation_pictures,
        "tune_pictures": tune_pictures,
        "evaluation_pictures": evaluation_pictures,
        "validation_picture_count": len(split.validation_pictures),
    }


def task1_checkpoint(seed, mouse):
    if seed == 42:
        return TASK1_SEED42_ROOT / mouse / "regressor.pt"
    return MODELS / "task1" / f"seed_{seed}" / mouse / "regressor.pt"


def predict_task1_seed(data, seed, device):
    for mouse in sorted(data["training"]):
        model = load_regressor(task1_checkpoint(seed, mouse), device)
        for row in data["rows"]:
            if row["mouse"] == mouse:
                coefficients = predict_logits(model, row["counts"], device=device)
                row["continuous"][seed] = dct_decode(coefficients, 32)
        del model
        release_memory()
    print(f"TASK1 PREDICTED seed={seed}", flush=True)


def train_task1_seed(data, seed):
    records = []
    for mouse in sorted(data["training"]):
        output = MODELS / "task1" / f"seed_{seed}" / mouse
        train = data["training"][mouse]["train"]
        validation = data["training"][mouse]["validation"]
        print(
            f"TASK1 TRAIN seed={seed} mouse={mouse} epochs={EPOCHS['task1'][mouse]} "
            f"train={len(train[0])} validation={len(validation[0])}",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SpikeRegressor(train[0].shape[1], train[1].shape[1], hidden=512, dropout=0.4)
        history = fit_regressor(
            model,
            train,
            validation,
            output,
            epochs=EPOCHS["task1"][mouse],
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=0.01,
            device=data["device"],
            seed=seed,
        )
        best = min(history, key=lambda row: row["validation_loss"])
        records.append({
            "seed": seed,
            "mouse": mouse,
            "epochs_requested": EPOCHS["task1"][mouse],
            "epochs_run": len(history),
            "checkpoint_epoch": best["epoch"],
            "best_validation_loss": best["validation_loss"],
            "checkpoint": str(task1_checkpoint(seed, mouse).relative_to(ROOT)),
        })
        del model
        release_memory()
    return records


def evaluate_task1(data, prefix_n, pictures, return_details=False):
    prediction_parts = []
    target_parts = []
    group_parts = []
    picture_parts = []
    candidate_parts = []
    for row in data["rows"]:
        mask = np.isin(row["targets"], pictures)
        continuous = np.mean(
            np.stack([row["continuous"][seed] for seed in SEEDS[:prefix_n]]),
            axis=0,
            dtype=np.float32,
        )
        predictions, candidate_indices = retrieve_glyphs(
            continuous[mask], data["bank"], metric="directional2", blur_sigma=1.0, device=data["device"]
        )
        prediction_parts.append(predictions)
        target_parts.append(data["images"][row["targets"][mask]])
        group_parts.extend([row["block"]] * int(mask.sum()))
        picture_parts.append(row["targets"][mask])
        candidate_parts.append(candidate_indices)
    predictions = np.concatenate(prediction_parts)
    references = np.concatenate(target_parts)
    groups = np.asarray(group_parts)
    scores = score_predictions(predictions, references, device=data["device"], batch_size=16)
    summary = summarize_scores(scores, groups)
    if not return_details:
        return summary, None
    return summary, {
        "predictions": predictions,
        "references": references,
        "groups": groups,
        "pictures": np.concatenate(picture_parts),
        "candidate_indices": np.concatenate(candidate_parts),
    }


def run_task1(images, device):
    started = time.perf_counter()
    print("=== TASK1 START ===", flush=True)
    data = prepare_task1(images)
    data["device"] = device
    data["bank"] = render_gb2312_bank(FONT_PATH, font_size=52, variation="Regular")
    if data["bank"].shape != (6763, 64, 64):
        raise RuntimeError(f"Unexpected GB2312 bank shape: {data['bank'].shape}")
    predict_task1_seed(data, 42, device)
    baseline_evaluation, baseline_details = evaluate_task1(
        data, 1, data["evaluation_pictures"], return_details=True
    )
    baseline = baseline_record("task1", baseline_evaluation)

    training = []
    for seed in SEEDS[1:]:
        training.extend(train_task1_seed(data, seed))
        predict_task1_seed(data, seed, device)

    tuning = []
    for prefix_n in PREFIX_SIZES:
        summary, _ = evaluate_task1(data, prefix_n, data["tune_pictures"])
        record = prefix_record(prefix_n, summary)
        tuning.append(record)
        print(f"TASK1 TUNE {record}", flush=True)
    selected = max(tuning, key=lambda row: row["ocf_mean"])
    if selected["prefix_n"] == 1:
        evaluation = baseline_evaluation
        details = baseline_details
    else:
        evaluation, details = evaluate_task1(
            data, selected["prefix_n"], data["evaluation_pictures"], return_details=True
        )

    selected_npz = ARTIFACTS / "task1_seed_ensemble_selected.npz"
    np.savez_compressed(
        selected_npz,
        predictions=details["predictions"].astype(np.uint8) * 255,
        targets=details["references"].astype(np.uint8) * 255,
        group_ids=details["groups"],
        picture_indices=details["pictures"],
        candidate_indices=details["candidate_indices"],
        prefix_n=np.asarray(selected["prefix_n"]),
        seeds=np.asarray(selected["seeds"], dtype=np.int64),
    )
    protocol = {
        "split_seed": 42,
        "picture_seed": 123,
        "model_validation_pictures": data["model_validation_pictures"].tolist(),
        "model_validation_picture_count": len(data["model_validation_pictures"]),
        "tune_pictures": data["tune_pictures"].tolist(),
        "tune_picture_count": len(data["tune_pictures"]),
        "evaluation_pictures": data["evaluation_pictures"].tolist(),
        "evaluation_picture_count": len(data["evaluation_pictures"]),
        "held_out_rule": "n=1 baseline reproduction plus only the tune-selected prefix; no other held-out prefix evaluation",
    }
    config = {
        "bin_ms": 25,
        "dct_cutoff": 32,
        "hidden": 512,
        "dropout": 0.4,
        "weight_decay": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "epochs": EPOCHS["task1"],
        "training_rows": "per-session four-repeat means followed by single trials, pooled by mouse as in test05/test14/test15",
        "model_validation": "shuffled[:64], exactly as the seed42 regressor training protocol",
        "continuous_ensemble": "equal-weight mean of seed-prefix DCT-decoded continuous images",
        "retrieval_bank": "fixed complete strict-GB2312 bank, NotoSansSC Regular size 52, 6763 glyphs",
        "font_path": str(FONT_PATH),
        "retrieval_metric": "directional2",
        "retrieval_blur_sigma": 1.0,
        "seed42_checkpoint_root": str(TASK1_SEED42_ROOT.relative_to(ROOT)),
        "new_checkpoint_root": str((MODELS / "task1").relative_to(ROOT)),
    }
    result = result_envelope(
        "task1", protocol, config, training, baseline, tuning, selected, evaluation, selected_npz
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    path = OUTPUT_DIR / "results_task1_seed_ensemble.json"
    write_json(path, result)
    print(
        f"TASK1 FINAL selected_n={selected['prefix_n']} OCF={evaluation['ocf_mean']:.10f} "
        f"baseline={baseline['actual_ocf']:.10f} delta={result['evaluation_delta_ocf']:.10f}",
        flush=True,
    )
    return result


def prepare_seen(task, images, device):
    bank_coefficients = dct_encode(images, 16)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, task, seed=42)
    selected_pictures = np.random.default_rng(123).permutation(len(images))
    tune_pictures = np.sort(selected_pictures[:256])
    evaluation_pictures = np.sort(selected_pictures[256:768])
    encoding_alpha = 1.0 if task == "task2" else 0.01
    rows = []
    training = {}
    for mouse in sorted({block.mouse for block in blocks}):
        train_x_parts = []
        train_y_parts = []
        validation_x_parts = []
        validation_y_parts = []
        encoding_neural_parts = []
        encoding_target_parts = []
        mouse_rows = []
        for block in [block for block in blocks if block.mouse == mouse]:
            block_split = split.blocks[block.key]
            if task == "task2":
                if len(block_split.train_trials):
                    single_x = block.features[block_split.train_trials]
                    single_targets = block.trial_targets[block_split.train_trials]
                    mean_targets, mean_x = aggregate_by_target(single_x, single_targets)
                    encoding_neural_parts.append(np.log1p(mean_x))
                    encoding_target_parts.append(mean_targets)
                    normalizer = fit_session_normalizer(single_x)
                    train_x_parts.extend((
                        apply_session_normalizer(single_x, normalizer),
                        apply_session_normalizer(mean_x, normalizer),
                    ))
                    train_y_parts.extend((single_targets, mean_targets))
                if len(block_split.validation_trials):
                    raw = block.features[block_split.validation_trials]
                    raw_targets = block.trial_targets[block_split.validation_trials]
                    targets, counts = aggregate_by_target(raw, raw_targets)
                    normalizer = fit_session_normalizer(raw)
                    mlp_counts = apply_session_normalizer(counts, normalizer)
                    validation_x_parts.append(mlp_counts)
                    validation_y_parts.append(targets)
            else:
                train_raw = block.features[block_split.train_trials]
                train_targets = block.trial_targets[block_split.train_trials]
                mean_targets, mean_x = aggregate_by_target(train_raw, train_targets)
                encoding_neural_parts.append(np.log1p(mean_x))
                encoding_target_parts.append(mean_targets)
                normalizer = fit_session_normalizer(train_raw)
                train_x_parts.append(apply_session_normalizer(train_raw, normalizer))
                train_y_parts.append(train_targets)
                raw = block.features[block_split.validation_trials]
                targets = block.trial_targets[block_split.validation_trials]
                counts = raw
                mlp_counts = apply_session_normalizer(counts, normalizer)
                validation_x_parts.append(mlp_counts)
                validation_y_parts.append(targets)
            if len(block_split.validation_trials):
                mouse_rows.append({
                    "block": block.key,
                    "mouse": mouse,
                    "targets": targets,
                    "counts": counts,
                    "mlp_counts": mlp_counts,
                    "logits": {},
                    "assignments": {},
                })

        encoding_path = RidgePath().fit(
            bank_coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_neural_parts)
        )
        prototypes = encoding_path.predict(bank_coefficients, encoding_alpha)
        for row in mouse_rows:
            encoding_scores = response_scores(
                np.log1p(row["counts"]), prototypes, metric="mse", device=device
            )
            row["encoding_z"] = row_zscore(encoding_scores)
            row.pop("counts")
        rows.extend(mouse_rows)

        train_x = np.concatenate(train_x_parts)
        train_y = np.concatenate(train_y_parts)
        validation_x = np.concatenate(validation_x_parts)
        validation_y = np.concatenate(validation_y_parts)
        tune_mask = np.isin(validation_y, tune_pictures)
        training[mouse] = {
            "train": (train_x, train_y),
            "validation": (validation_x[tune_mask], validation_y[tune_mask]),
        }
        print(
            f"{task.upper()} PREPARED {mouse}: train={len(train_x)} "
            f"model_validation={int(tune_mask.sum())} score_rows={len(mouse_rows)}",
            flush=True,
        )
        del prototypes, encoding_path
        release_memory()
    return {
        "images": images,
        "rows": rows,
        "training": training,
        "tune_pictures": tune_pictures,
        "evaluation_pictures": evaluation_pictures,
        "encoding_alpha": encoding_alpha,
        "device": device,
    }


def seen_checkpoint(task, seed, mouse):
    if seed == 42:
        return SEEN_SEED42_ROOTS[task] / mouse / "classifier.pt"
    return MODELS / task / f"seed_{seed}" / mouse / "classifier.pt"


def predict_seen_seed(task, data, seed):
    for mouse in sorted(data["training"]):
        model = load_classifier(seen_checkpoint(task, seed, mouse), data["device"])
        for row in data["rows"]:
            if row["mouse"] == mouse:
                row["logits"][seed] = predict_logits(model, row["mlp_counts"], device=data["device"])
        del model
        release_memory()
    print(f"{task.upper()} PREDICTED seed={seed}", flush=True)


def train_seen_seed(task, data, seed):
    records = []
    for mouse in sorted(data["training"]):
        output = MODELS / task / f"seed_{seed}" / mouse
        train = data["training"][mouse]["train"]
        validation = data["training"][mouse]["validation"]
        print(
            f"{task.upper()} TRAIN seed={seed} mouse={mouse} epochs={EPOCHS[task][mouse]} "
            f"train={len(train[0])} validation={len(validation[0])}",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SpikeClassifier(
            n_features=train[0].shape[1], n_classes=len(data["images"]),
            hidden=512, dropout=0.2, apply_log=False,
        )
        history = fit_classifier(
            model,
            train,
            validation,
            output,
            epochs=EPOCHS[task][mouse],
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=0.001,
            device=data["device"],
            seed=seed,
        )
        best = max(history, key=lambda row: row["validation_accuracy"])
        records.append({
            "seed": seed,
            "mouse": mouse,
            "epochs_requested": EPOCHS[task][mouse],
            "epochs_run": len(history),
            "checkpoint_epoch": best["epoch"],
            "best_validation_accuracy": best["validation_accuracy"],
            "checkpoint": str(seen_checkpoint(task, seed, mouse).relative_to(ROOT)),
        })
        del model
        release_memory()
    return records


def assign_seen_prefix(data, prefix_n):
    for row in data["rows"]:
        mean_logits = np.mean(
            np.stack([row["logits"][seed] for seed in SEEDS[:prefix_n]]),
            axis=0,
            dtype=np.float32,
        )
        combined = 0.5 * row_zscore(mean_logits) + 0.5 * row["encoding_z"]
        row["assignments"][prefix_n] = unique_assignment(combined)
    print(f"ASSIGNED prefix_n={prefix_n} rows={len(data['rows'])}", flush=True)


def evaluate_seen(data, prefix_n, pictures, return_details=False):
    prediction_parts = []
    target_parts = []
    group_parts = []
    predicted_index_parts = []
    true_index_parts = []
    picture_parts = []
    for row in data["rows"]:
        mask = np.isin(row["targets"], pictures)
        assigned = row["assignments"][prefix_n]
        prediction_parts.append(data["images"][assigned[mask]])
        target_parts.append(data["images"][row["targets"][mask]])
        group_parts.extend([row["block"]] * int(mask.sum()))
        predicted_index_parts.append(assigned[mask])
        true_index_parts.append(row["targets"][mask])
        picture_parts.append(row["targets"][mask])
    predictions = np.concatenate(prediction_parts)
    references = np.concatenate(target_parts)
    groups = np.asarray(group_parts)
    predicted_indices = np.concatenate(predicted_index_parts)
    true_indices = np.concatenate(true_index_parts)
    scores = score_predictions(predictions, references, device=data["device"], batch_size=16)
    summary = summarize_scores(scores, groups)
    summary["top1_accuracy"] = float((predicted_indices == true_indices).mean())
    if not return_details:
        return summary, None
    return summary, {
        "predictions": predictions,
        "references": references,
        "groups": groups,
        "predicted_indices": predicted_indices,
        "target_indices": true_indices,
        "pictures": np.concatenate(picture_parts),
    }


def run_seen(task, images, device):
    started = time.perf_counter()
    print(f"=== {task.upper()} START ===", flush=True)
    data = prepare_seen(task, images, device)
    predict_seen_seed(task, data, 42)
    assign_seen_prefix(data, 1)
    baseline_evaluation, baseline_details = evaluate_seen(
        data, 1, data["evaluation_pictures"], return_details=True
    )
    baseline = baseline_record(task, baseline_evaluation)

    training = []
    for seed in SEEDS[1:]:
        training.extend(train_seen_seed(task, data, seed))
        predict_seen_seed(task, data, seed)
    for prefix_n in PREFIX_SIZES[1:]:
        assign_seen_prefix(data, prefix_n)

    tuning = []
    for prefix_n in PREFIX_SIZES:
        summary, _ = evaluate_seen(data, prefix_n, data["tune_pictures"])
        record = prefix_record(prefix_n, summary)
        tuning.append(record)
        print(f"{task.upper()} TUNE {record}", flush=True)
    selected = max(tuning, key=lambda row: row["ocf_mean"])
    if selected["prefix_n"] == 1:
        evaluation = baseline_evaluation
        details = baseline_details
    else:
        evaluation, details = evaluate_seen(
            data, selected["prefix_n"], data["evaluation_pictures"], return_details=True
        )

    selected_npz = ARTIFACTS / f"{task}_seed_ensemble_selected.npz"
    np.savez_compressed(
        selected_npz,
        predictions=details["predictions"].astype(np.uint8) * 255,
        targets=details["references"].astype(np.uint8) * 255,
        group_ids=details["groups"],
        picture_indices=details["pictures"],
        predicted_indices=details["predicted_indices"],
        target_indices=details["target_indices"],
        prefix_n=np.asarray(selected["prefix_n"]),
        seeds=np.asarray(selected["seeds"], dtype=np.int64),
    )
    protocol = {
        "split_seed": 42,
        "picture_seed": 123,
        "tune_pictures": data["tune_pictures"].tolist(),
        "tune_picture_count": len(data["tune_pictures"]),
        "evaluation_pictures": data["evaluation_pictures"].tolist(),
        "evaluation_picture_count": len(data["evaluation_pictures"]),
        "assignment_scope": "unique_assignment over every complete 2748-picture validation block before tune/evaluation subsetting",
        "held_out_rule": "n=1 baseline reproduction plus only the tune-selected prefix; no other held-out prefix evaluation",
    }
    config = {
        "bin_ms": 50,
        "hidden": 512,
        "dropout": 0.2,
        "weight_decay": 0.001,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "epochs": EPOCHS[task],
        "session_normalization": "same test09 session-norm statistics and apply_log=False classifiers",
        "training_rows": "Task2 train singles plus means; Task3 pooled per-session singles with training-derived session normalization",
        "encoding_alpha": data["encoding_alpha"],
        "encoding_prototypes": "test09 per-mouse RidgePath prototypes constructed once and fixed for every seed and prefix",
        "logits_ensemble": "equal-weight mean of raw seed-prefix logits, then row_zscore",
        "fusion": "0.5 * ensemble logits row_zscore + 0.5 * fixed encoding response_scores row_zscore",
        "assignment": "unique_assignment",
        "seed42_checkpoint_root": str(SEEN_SEED42_ROOTS[task].relative_to(ROOT)),
        "new_checkpoint_root": str((MODELS / task).relative_to(ROOT)),
    }
    result = result_envelope(
        task, protocol, config, training, baseline, tuning, selected, evaluation, selected_npz
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    path = OUTPUT_DIR / f"results_{task}_seed_ensemble.json"
    write_json(path, result)
    print(
        f"{task.upper()} FINAL selected_n={selected['prefix_n']} OCF={evaluation['ocf_mean']:.10f} "
        f"top1={evaluation['top1_accuracy']:.10f} baseline={baseline['actual_ocf']:.10f} "
        f"delta={result['evaluation_delta_ocf']:.10f}",
        flush=True,
    )
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE {device}; seeds={SEEDS}; serial order=task1,task2,task3", flush=True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    if len(images) != 2748:
        raise ValueError(f"Expected 2748 training pictures, found {len(images)}.")
    results = []
    results.append(run_task1(images, device))
    release_memory()
    results.append(run_seen("task2", images, device))
    release_memory()
    results.append(run_seen("task3", images, device))
    release_memory()
    print("ALL TASKS COMPLETE", flush=True)
    for result in results:
        print(
            f"SUMMARY {result['config']['task']}: tune={result['tuning']} "
            f"selected_n={result['selected']['prefix_n']} held_out={result['evaluation']['ocf_mean']:.10f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
