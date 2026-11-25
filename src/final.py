"""Train the selected full-data models and build a complete Kaggle submission."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataloader import load_images
from src.encoding import response_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.kaggle_submission import convert
from src.neural import (
    SpikeClassifier,
    SpikeRegressor,
    SpikeTemporalClassifier,
    fit_classifier_fixed_epochs,
    load_classifier,
    load_regressor,
    predict_logits,
)
from src.reranker import (
    apply_listwise_reranker,
    build_candidate_features,
    fit_listwise_reranker,
    load_listwise_reranker,
)
from src.retrieval import render_gb2312_bank, retrieve_glyphs, softmax_dct_similarity, unique_assignment
from src.ridge import RidgePath, dct_decode, dct_encode
from src.submission import pack_submission, save_images

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"


def fit_session_normalizer(features):
    transformed = np.log1p(np.maximum(features, 0))
    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-5)


def apply_session_normalizer(features, normalizer):
    mean, scale = normalizer
    return (np.log1p(np.maximum(features, 0)) - mean) / scale


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def release_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def build_classifier(config, n_features, n_units, n_bins):
    common = dict(
        n_classes=2748,
        hidden=config["hidden"],
        dropout=config["dropout"],
        apply_log=not config.get("session_norm", False),
    )
    if config.get("architecture", "mlp") == "temporal":
        return SpikeTemporalClassifier(n_features, n_units, n_bins, **common)
    return SpikeClassifier(n_features, **common)


def checkpoint_path(output, task, mouse, seed):
    stem = "regressor" if task == "task1" else "classifier"
    suffix = "" if seed == 42 else f"_seed{seed}"
    return output / task / mouse / f"{stem}{suffix}.pt"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reranker_manifest(checkpoint, operation):
    checkpoint = Path(checkpoint).resolve()
    return {
        "task": "task2/reranker",
        "operation": operation,
        "source_checkpoint": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }


def write_reranker_manifest(path, checkpoint, operation, **details):
    manifest = {**reranker_manifest(checkpoint, operation), **details}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fit_full(model, features, targets, output, *, epochs, regression, learning_rate=3e-4,
             weight_decay=1e-3, batch_size=256, device="cuda", seed=42):
    """Fit all permitted data for a fixed CV-selected number of epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.set_normalization(features)
    model.to(device)
    target_array = np.asarray(targets, dtype=np.float32 if regression else np.int64)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(target_array),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=device == "cuda"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                prediction = model(batch_x)
                if regression:
                    loss = torch.nn.functional.mse_loss(prediction.float(), batch_y)
                else:
                    loss = torch.nn.functional.cross_entropy(prediction.float(), batch_y, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        row = {"epoch": epoch, "train_loss": total_loss / len(features)}
        history.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = {
        "n_features": model.n_features,
        "hidden": model.hidden,
        "dropout": model.dropout,
        "apply_log": model.apply_log,
        "state_dict": model.state_dict(),
        "epoch": epochs,
    }
    if regression:
        checkpoint["n_outputs"] = model.n_outputs
    else:
        checkpoint.update({
            "n_classes": model.n_classes,
            "architecture": getattr(model, "architecture", "mlp"),
            "n_units": getattr(model, "n_units", None),
            "n_bins": getattr(model, "n_bins", None),
        })
    temporary = output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    history_name = "history.json" if seed == 42 else f"history_seed{seed}.json"
    (output.parent / history_name).write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def train_task2_oof_reranker(blocks, config, images, output, device, split=None):
    """Train from permitted public session1-3 OOF rows; never use test or session4 labels."""
    reranker_config = config["task2"]["reranker"]
    sessions = tuple(reranker_config["oof_sessions"])
    if len(sessions) < 2:
        raise ValueError("OOF reranker training requires at least two public sessions.")
    coefficients = dct_encode(images, 16)
    rows = []
    records = []
    for mouse in sorted({block.mouse for block in blocks}):
        group = [block for block in blocks if block.mouse == mouse]
        if not set(sessions).issubset({block.session for block in group}):
            raise ValueError(f"Missing public OOF sessions for {mouse}.")
        for score_session in sessions:
            train_sessions = tuple(session for session in sessions if session != score_session)
            train_x_parts, train_y_parts = [], []
            encoding_x_parts, encoding_target_parts = [], []
            target_counts = target_x = target_ids = None
            for block in group:
                if block.session not in train_sessions and block.session != score_session:
                    continue
                if split is None:
                    raw = block.features
                    labels = block.trial_targets
                else:
                    train_trials = split.blocks[block.key].train_trials
                    if not len(train_trials):
                        raise ValueError(f"No permitted OOF trials for {block.key}.")
                    raw = block.features[train_trials]
                    labels = block.trial_targets[train_trials]
                targets, counts = aggregate_by_target(raw, labels)
                if split is None:
                    np.testing.assert_array_equal(targets, np.arange(len(images)))
                normalizer = fit_session_normalizer(raw)
                if block.session in train_sessions:
                    encoding_x_parts.append(np.log1p(counts))
                    encoding_target_parts.append(targets)
                    train_x_parts.extend((
                        apply_session_normalizer(raw, normalizer),
                        apply_session_normalizer(counts, normalizer),
                    ))
                    train_y_parts.extend((labels, targets))
                if block.session == score_session:
                    target_counts = counts
                    target_ids = targets
                    target_x = apply_session_normalizer(counts, normalizer)
            if target_counts is None or target_ids is None or len(encoding_x_parts) != len(train_sessions):
                raise ValueError(f"Invalid OOF fold for {mouse}/{score_session}.")

            ridge = RidgePath().fit(
                coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_x_parts)
            ).predict(coefficients, config["task2"]["encoding_alpha"])
            encoding_z = row_zscore(
                response_scores(np.log1p(target_counts), ridge, metric="mse", device=device)
            ).astype(np.float32)
            del ridge
            train_x = np.concatenate(train_x_parts).astype(np.float32, copy=False)
            train_y = np.concatenate(train_y_parts).astype(np.int64, copy=False)
            logits = []
            final_losses = {}
            for seed in config["task2"].get("seeds", [42]):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = SpikeClassifier(
                    train_x.shape[1], len(images), hidden=config["task2"]["hidden"],
                    dropout=config["task2"]["dropout"], apply_log=False
                )
                model, history = fit_classifier_fixed_epochs(
                    model,
                    (train_x, train_y),
                    epochs=config["task2"]["epochs"][mouse],
                    weight_decay=config["task2"]["weight_decay"],
                    device=device,
                    seed=seed,
                )
                logits.append(predict_logits(model, target_x, device=device))
                final_losses[str(seed)] = history[-1]
                del model
                release_memory(device)
            mlp_z = row_zscore(np.mean(np.stack(logits), axis=0, dtype=np.float32)).astype(np.float32)
            weight = config["task2"].get("mlp_weight", 1.0)
            base = (weight * mlp_z + (1 - weight) * encoding_z).astype(np.float32)
            top, numeric = build_candidate_features(
                base, mlp_z, encoding_z, mouse, top_n=reranker_config["top_n"]
            )
            targets = target_ids
            inside = np.any(top == targets[:, None], axis=1)
            rows.append({
                "block": f"{mouse}/{score_session}",
                "targets": targets,
                "top": top,
                "numeric": numeric,
            })
            records.append({
                "mouse": mouse,
                "score_session": score_session,
                "train_sessions": list(train_sessions),
                "train_rows_including_trial_and_mean_rows": len(train_x),
                "mlp_epochs_per_seed": config["task2"]["epochs"][mouse],
                "mlp_final_train_loss": final_losses,
                "target_rows": len(targets),
                "oof_raw_top1": float(np.mean(base.argmax(axis=1) == targets)),
                "oof_base_recall32": float(inside.mean()),
            })
            print(f"OOF RERANKER FEATURES {mouse}/{score_session}: eligible={inside.sum()}", flush=True)
            del train_x, train_y, logits, mlp_z, encoding_z, base, target_counts, target_x
            release_memory(device)
    checkpoint = output / "listwise_cnn.pt"
    model, training = fit_listwise_reranker(
        rows,
        images,
        checkpoint,
        seed=reranker_config["seed"],
        epochs=reranker_config["epochs"],
        batch_size=reranker_config["batch_size"],
        learning_rate=reranker_config["learning_rate"],
        weight_decay=reranker_config["weight_decay"],
        top_n=reranker_config["top_n"],
        device=device,
    )
    del model, rows
    release_memory(device)
    label_source = "public train session1-3 OOF only"
    if split is not None:
        label_source += "; fixed-split train trials only"
    manifest = write_reranker_manifest(
        output / "reranker_provenance.json",
        checkpoint,
        "trained",
        label_source=label_source,
        oof_sessions=list(sessions),
        oof_base_training=records,
    )
    return {
        "status": "trained",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["sha256"],
        "manifest": str(output / "reranker_provenance.json"),
        "label_source": label_source,
        "oof_base_training": records,
        **training,
    }


def train_task(config, task, output, *, images=None, blocks=None, split=None,
               skip_existing=False, device=None):
    if task not in ("task1", "task2", "task3"):
        raise ValueError("Unknown task.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if images is None:
        images = (load_images(DATA) >= 128).astype(np.float32)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    task_config = config[task]
    if blocks is None:
        blocks = load_feature_blocks(DATA, "train", bin_ms=task_config["bin_ms"])
    bank_coefficients = dct_encode(images, 16)
    summary = {"name": config["name"], "task": task, "models": {}}
    for mouse in sorted({block.mouse for block in blocks}):
        mouse_blocks = [block for block in blocks if block.mouse == mouse]
        train_x_parts = []
        train_target_parts = []
        encoding_neural_parts = []
        encoding_target_parts = []
        normalizers = {}
        for block in mouse_blocks:
            trial_indices = (
                split.blocks[block.key].train_trials if split is not None
                else np.arange(block.n_trials, dtype=np.int64)
            )
            if not len(trial_indices):
                continue
            single_x = block.features[trial_indices]
            single_targets = block.trial_targets[trial_indices]
            mean_targets, mean_x = aggregate_by_target(single_x, single_targets)
            if task != "task1":
                encoding_neural_parts.append(np.log1p(mean_x))
                encoding_target_parts.append(mean_targets)
            if task == "task1":
                train_x_parts.extend((mean_x, single_x))
                train_target_parts.extend((mean_targets, single_targets))
            elif task == "task2":
                normalizer = fit_session_normalizer(single_x)
                train_x_parts.extend((
                    apply_session_normalizer(single_x, normalizer),
                    apply_session_normalizer(mean_x, normalizer),
                ))
                train_target_parts.extend((single_targets, mean_targets))
            else:
                normalizer = fit_session_normalizer(single_x)
                normalizers[block.session] = normalizer
                train_x_parts.append(apply_session_normalizer(single_x, normalizer))
                train_target_parts.append(single_targets)
        if not train_x_parts:
            raise ValueError(f"No training trials for {task}/{mouse}.")
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_target_parts)
        key = f"{task}/{mouse}"
        mouse_output = output / task / mouse
        if task == "task1":
            train_y = dct_encode(images[train_targets], 32)
        else:
            train_y = train_targets
            mouse_output.mkdir(parents=True, exist_ok=True)
            for session, normalizer in normalizers.items():
                np.savez(mouse_output / f"{session}_normalizer.npz", mean=normalizer[0], scale=normalizer[1])
            encoding_path = RidgePath().fit(
                bank_coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_neural_parts)
            )
            prototypes = encoding_path.predict(bank_coefficients, task_config["encoding_alpha"])
            np.save(mouse_output / "encoding_prototypes.npy", prototypes)
        for seed in task_config.get("seeds", [42]):
            checkpoint = checkpoint_path(output, task, mouse, seed)
            model_key = key if seed == 42 else f"{key}/seed{seed}"
            if skip_existing and checkpoint.exists():
                try:
                    load = load_regressor if task == "task1" else load_classifier
                    model = load(checkpoint, "cpu")
                except Exception as error:
                    print(f"retraining {model_key}: checkpoint could not be loaded: {error}", flush=True)
                else:
                    summary["models"][model_key] = {
                        "samples": len(train_x),
                        "epochs": task_config["epochs"][mouse],
                        "seed": seed,
                        "status": "skipped",
                        "checkpoint": str(checkpoint),
                    }
                    del model
                    print(f"skipped {model_key}: existing checkpoint loaded", flush=True)
                    continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            if task == "task1":
                model = SpikeRegressor(
                    train_x.shape[1], train_y.shape[1], task_config["hidden"], task_config["dropout"]
                )
            else:
                model = build_classifier(
                    task_config, train_x.shape[1], mouse_blocks[0].n_units, round(250 / task_config["bin_ms"])
                )
            history = fit_full(
                model,
                train_x,
                train_y,
                checkpoint,
                epochs=task_config["epochs"][mouse],
                regression=task == "task1",
                weight_decay=task_config["weight_decay"],
                device=device,
                seed=seed,
            )
            summary["models"][model_key] = {
                "samples": len(train_x),
                "epochs": task_config["epochs"][mouse],
                "seed": seed,
                "status": "trained",
                "final_train_loss": history[-1]["train_loss"],
                "checkpoint": str(checkpoint),
            }
            del model
        del train_x, train_targets, train_y
        release_memory(device)
    if task == "task2" and task_config.get("reranker", {}).get("enabled", False):
        checkpoint = output / "listwise_cnn.pt"
        if skip_existing and checkpoint.is_file():
            try:
                reranker = load_listwise_reranker(checkpoint, "cpu")
            except Exception as error:
                print(f"retraining task2/reranker: checkpoint could not be loaded: {error}", flush=True)
            else:
                del reranker
                manifest = write_reranker_manifest(
                    output / "reranker_provenance.json", checkpoint, "loaded_existing_during_training"
                )
                summary["models"]["task2/reranker"] = {
                    "status": "skipped",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": manifest["sha256"],
                    "manifest": str(output / "reranker_provenance.json"),
                }
                print("skipped task2/reranker: existing checkpoint loaded", flush=True)
                checkpoint = None
        if checkpoint is not None:
            oof_blocks = blocks
            if split is not None:
                oof_blocks = [
                    block for block in blocks if len(split.blocks[block.key].train_trials)
                ]
            summary["models"]["task2/reranker"] = train_task2_oof_reranker(
                oof_blocks, config, images, output, device, split=split
            )
    return summary


def train_final(config, output, skip_existing=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"name": config["name"], "models": {}}
    for task in ("task1", "task2", "task3"):
        task_summary = train_task(
            config, task, output, images=images, skip_existing=skip_existing, device=device
        )
        summary["models"].update(task_summary["models"])
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def retrieval_candidate_indices(continuous, bank, metric, device, top_n):
    """Expose ranked directional-retrieval candidates for public-label oracle diagnostics."""
    from src.retrieval import _retrieval_embedding

    directional_metrics = {"directional2", "directional", "directional8", "directional_multi"}
    if metric not in directional_metrics:
        raise ValueError("Candidate diagnostics require a directional retrieval metric.")
    if not 1 <= top_n <= len(bank):
        raise ValueError("candidate_top_n is outside the retrieval bank.")
    bank_tensor = torch.from_numpy(np.asarray(bank, dtype=np.float32)[:, None]).to(device)
    bank_embedding = torch.nn.functional.normalize(
        _retrieval_embedding(bank_tensor, metric), dim=1
    )
    output = []
    with torch.inference_mode():
        for start in range(0, len(continuous), 64):
            batch = torch.from_numpy(
                np.asarray(continuous[start:start + 64], dtype=np.float32)[:, None]
            ).to(device)
            query = torch.nn.functional.normalize(_retrieval_embedding(batch, metric), dim=1)
            scores = query @ bank_embedding.T
            output.append(torch.topk(scores, top_n, dim=1).indices.cpu().numpy())
    return np.concatenate(output).astype(np.int64)


def predict_task(config, task, models, *, images=None, blocks=None, split=None, device=None,
                 reranker_checkpoint=None, candidate_top_n=0, manifest_output=None):
    if task not in ("task1", "task2", "task3"):
        raise ValueError("Unknown task.")
    if images is None:
        images = (load_images(DATA) >= 128).astype(np.float32)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = Path(models)
    task_config = config[task]
    seeds = task_config.get("seeds", [42])
    if blocks is None:
        blocks = load_feature_blocks(DATA, "test", task=task, bin_ms=task_config["bin_ms"])
    task_predictions = []
    task_ids = []
    task_groups = []
    task_targets = []
    candidate_parts = []
    models_by_mouse = {}
    reranker = None
    loaded_reranker_manifest = None
    reranker_config = task_config.get("reranker", {})
    if task == "task2" and reranker_config.get("enabled", False):
        if reranker_checkpoint is None:
            raise ValueError("An explicit Task2 reranker checkpoint is required.")
        reranker = load_listwise_reranker(reranker_checkpoint, device)
        loaded_reranker_manifest = reranker_manifest(reranker_checkpoint, "loaded_for_prediction")
        if manifest_output is not None:
            manifest_output = Path(manifest_output)
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(
                json.dumps(loaded_reranker_manifest, indent=2), encoding="utf-8"
            )
    centered_coefficients = None
    if task == "task3" and task_config.get("prototype_continuous", {}).get("enabled", False):
        if task_config["prototype_continuous"].get("temperature", 1.0) != 1.0:
            raise ValueError("The verified Task3 continuous branch requires temperature=1.")
        coefficients = dct_encode(images, 16)
        centered_coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    retrieval_bank = images
    retrieval_metric = task_config.get("retrieval")
    if task == "task1" and retrieval_metric == "font_directional2":
        font_path = Path(r"C:\Windows\Fonts") / task_config["font_file"]
        retrieval_bank = render_gb2312_bank(
            font_path,
            font_size=task_config["font_size"],
            variation=task_config["font_variation"],
        )
        retrieval_metric = "directional2"
    for block in blocks:
        trial_indices = selected_ids = None
        if split is not None:
            if block.targets is None:
                raise ValueError("Public evaluation requires labelled training blocks, never hidden test targets.")
            block_split = split.blocks[block.key]
            trial_indices = block_split.validation_trials
            if not len(trial_indices):
                continue
            if task == "task3":
                selected_ids = block.sample_ids[block.trial_sample_indices[trial_indices]]
        if block.mouse not in models_by_mouse:
            load = load_regressor if task == "task1" else load_classifier
            models_by_mouse[block.mouse] = [
                load(checkpoint_path(models, task, block.mouse, seed), device) for seed in seeds
            ]
        mouse_models = models_by_mouse[block.mouse]
        if task == "task1":
            if split is not None:
                targets, counts = aggregate_by_target(
                    block.features[trial_indices], block.trial_targets[trial_indices]
                )
                selected_ids = np.asarray([
                    f"{block.key}/glyph_{int(target):04d}" for target in targets
                ])
            else:
                counts = block.aggregate()
            coefficients = np.mean(
                np.stack([predict_logits(model, counts, device=device) for model in mouse_models]),
                axis=0,
                dtype=np.float32,
            )
            continuous = dct_decode(coefficients, 32)
            predictions, _ = retrieve_glyphs(
                continuous, retrieval_bank, metric=retrieval_metric, device=device
            )
            if candidate_top_n:
                candidate_parts.append(retrieval_candidate_indices(
                    continuous, retrieval_bank, retrieval_metric, device, candidate_top_n
                ))
        else:
            if task == "task2":
                if split is not None:
                    raw_validation = block.features[trial_indices]
                    targets, raw_counts = aggregate_by_target(
                        raw_validation, block.trial_targets[trial_indices]
                    )
                    selected_ids = np.asarray([
                        f"{block.key}/glyph_{int(target):04d}" for target in targets
                    ])
                else:
                    raw_validation = block.features
                    raw_counts = block.aggregate()
                normalizer = fit_session_normalizer(raw_validation)
            else:
                raw_counts = block.features[trial_indices] if split is not None else block.features
                with np.load(models / task / block.mouse / f"{block.session}_normalizer.npz") as data:
                    normalizer = (data["mean"], data["scale"])
            counts = apply_session_normalizer(raw_counts, normalizer)
            mlp_scores = np.mean(
                np.stack([predict_logits(model, counts, device=device) for model in mouse_models]),
                axis=0,
                dtype=np.float32,
            )
            prototypes = np.load(models / task / block.mouse / "encoding_prototypes.npy", allow_pickle=False)
            encoding_scores = response_scores(np.log1p(raw_counts), prototypes, metric="mse", device=device)
            mlp_z = row_zscore(mlp_scores)
            encoding_z = row_zscore(encoding_scores)
            if task == "task3" and centered_coefficients is not None:
                continuous_scores = softmax_dct_similarity(mlp_scores, centered_coefficients, device=device)
                scores = (mlp_z + encoding_z + row_zscore(continuous_scores)) / 3.0
            else:
                weight = task_config.get("mlp_weight", 1.0)
                scores = weight * mlp_z + (1 - weight) * encoding_z
            if reranker is not None:
                scores = apply_listwise_reranker(
                    reranker,
                    scores,
                    mlp_z,
                    encoding_z,
                    block.mouse,
                    images,
                    device,
                    top_n=reranker_config["top_n"],
                    k=reranker_config["k"],
                    batch_size=reranker_config["inference_batch_size"],
                )
            if candidate_top_n:
                if candidate_top_n > scores.shape[1]:
                    raise ValueError("candidate_top_n exceeds the glyph count.")
                candidate_parts.append(
                    np.argsort(-scores, axis=1, kind="stable")[:, :candidate_top_n].astype(np.int64)
                )
            indices = unique_assignment(scores) if task_config["assignment"] else scores.argmax(axis=1)
            predictions = images[indices]
            if split is not None and task == "task3":
                targets = block.trial_targets[trial_indices]
        task_predictions.append(predictions.astype(np.float32))
        task_ids.extend((selected_ids if split is not None else block.sample_ids).tolist())
        task_groups.extend([block.key] * len(predictions))
        if split is not None:
            task_targets.append(np.asarray(targets, dtype=np.int64))
        print(f"predicted {task}/{block.key}: {len(predictions)}", flush=True)
    if not task_predictions:
        raise ValueError(f"No prediction rows for {task}.")
    result = {
        "predictions": np.concatenate(task_predictions).astype(np.float32),
        "sample_ids": np.asarray(task_ids),
        "groups": np.asarray(task_groups),
        "targets": np.concatenate(task_targets) if split is not None else None,
        "candidate_indices": np.concatenate(candidate_parts) if candidate_top_n else None,
        "candidate_bank": retrieval_bank if candidate_top_n else None,
        "reranker": loaded_reranker_manifest,
    }
    del models_by_mouse, reranker
    release_memory(device)
    return result


def predict_final(config, models, output, *, reranker_checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions_by_task = {}
    ids_by_task = {}
    loaded_reranker_manifest = None
    reranker_enabled = config["task2"].get("reranker", {}).get("enabled", False)
    if reranker_enabled and reranker_checkpoint is None:
        raise ValueError("An explicit Task2 reranker checkpoint is required.")
    for task in ("task1", "task2", "task3"):
        prediction = predict_task(
            config,
            task,
            models,
            images=images,
            device=device,
            reranker_checkpoint=reranker_checkpoint if task == "task2" else None,
            manifest_output=output / "artifact_manifest.json" if task == "task2" and reranker_enabled else None,
        )
        predictions_by_task[task] = prediction["predictions"]
        ids_by_task[task] = prediction["sample_ids"].tolist()
        if prediction["reranker"] is not None:
            loaded_reranker_manifest = prediction["reranker"]
    if sum(len(ids) for ids in ids_by_task.values()) != 30200:
        raise ValueError("Prediction IDs do not total 30,200.")
    png_folder = output / "png"
    if png_folder.exists():
        raise FileExistsError(f"Refuse to mix predictions with an existing folder: {png_folder}")
    for task in ("task1", "task2", "task3"):
        task_folder = png_folder / task
        task_folder.mkdir(parents=True, exist_ok=False)
        task_array = predictions_by_task[task][:, None]
        save_images(
            torch.from_numpy(task_array), ids_by_task[task], task_folder, image_size=(64, 64)
        )
    png_archive = output / "submission_png.zip"
    pack_submission(png_folder, png_archive, ids_by_task)
    kaggle_archive = output / "kaggle_submission.zip"
    convert(png_archive, kaggle_archive)
    return {
        "png_archive": str(png_archive),
        "kaggle_archive": str(kaggle_archive),
        "kaggle_sha256": file_sha256(kaggle_archive),
        "samples": 30200,
        "task2_reranker": loaded_reranker_manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/final_config.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--models-run-name", help="Reuse base models from an earlier submission run.")
    parser.add_argument(
        "--reranker-run-name",
        help="Explicit submission run containing models/listwise_cnn.pt; requires --skip-train.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip all model training.")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Train only missing or unloadable checkpoints."
    )
    args = parser.parse_args()
    if args.skip_train and args.skip_existing:
        parser.error("--skip-existing cannot be used with --skip-train.")
    if args.models_run_name and not args.skip_train:
        parser.error("--models-run-name requires --skip-train.")
    if args.reranker_run_name and not args.skip_train:
        parser.error("--reranker-run-name requires --skip-train; training writes the current run's reranker.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reranker_enabled = config["task2"].get("reranker", {}).get("enabled", False)
    if args.reranker_run_name and not reranker_enabled:
        parser.error("--reranker-run-name was provided but the Task2 reranker is disabled.")
    run_output = ROOT / "test/90_submission/artifacts" / args.run_name
    models_run_name = args.models_run_name or args.run_name
    models = ROOT / "test/90_submission/artifacts" / models_run_name / "models"
    reranker_run_name = None
    reranker_checkpoint = None
    if reranker_enabled:
        reranker_run_name = args.reranker_run_name or models_run_name
        if (
            args.skip_train and args.reranker_run_name is None
            and args.models_run_name is not None and args.models_run_name != args.run_name
        ):
            parser.error(
                "--reranker-run-name is required when --models-run-name differs from --run-name; "
                "the reranker is never inferred from the prediction output directory."
            )
        if not args.skip_train:
            reranker_run_name = args.run_name
        reranker_checkpoint = (
            ROOT / "test/90_submission/artifacts" / reranker_run_name / "models/listwise_cnn.pt"
        )
    if not args.skip_train:
        train_final(config, models, skip_existing=args.skip_existing)
    result = predict_final(config, models, run_output, reranker_checkpoint=reranker_checkpoint)
    result.update({
        "name": config["name"],
        "models_run_name": models_run_name,
        "task2_reranker_run_name": reranker_run_name,
        "task2_reranker_checkpoint": str(reranker_checkpoint) if reranker_checkpoint else None,
        "task2_reranker_sha256": result["task2_reranker"]["sha256"]
            if result["task2_reranker"] else None,
    })
    (run_output / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
