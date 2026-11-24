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


def train_task2_oof_reranker(blocks, config, images, output, device):
    """Train from public session1-3 OOF predictions; no test or session4 labels are used."""
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
            target_counts = target_x = None
            for block in group:
                if block.session not in train_sessions and block.session != score_session:
                    continue
                raw = block.features
                labels = block.trial_targets
                targets, counts = aggregate_by_target(raw, labels)
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
                    target_x = apply_session_normalizer(counts, normalizer)
            if target_counts is None or len(encoding_x_parts) != len(train_sessions):
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
            targets = np.arange(len(images), dtype=np.int64)
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
                "oof_raw_top1_all_glyphs": float(np.mean(base.argmax(axis=1) == targets)),
                "oof_base_recall32_all_glyphs": float(inside.mean()),
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
    return {
        "status": "trained",
        "checkpoint": str(checkpoint),
        "label_source": "public train session1-3 OOF only",
        "oof_base_training": records,
        **training,
    }


def train_final(config, output, skip_existing=False):
    output.mkdir(parents=True, exist_ok=True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    bank_coefficients = dct_encode(images, 16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"name": config["name"], "models": {}}
    for task in ("task1", "task2", "task3"):
        task_config = config[task]
        blocks = load_feature_blocks(DATA, "train", bin_ms=task_config["bin_ms"])
        for mouse in {block.mouse for block in blocks}:
            mouse_blocks = [block for block in blocks if block.mouse == mouse]
            train_x_parts = []
            train_target_parts = []
            encoding_neural_parts = []
            encoding_target_parts = []
            normalizers = {}
            for block in mouse_blocks:
                single_x = block.features
                single_targets = block.trial_targets
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
                    summary["models"]["task2/reranker"] = {
                        "status": "skipped", "checkpoint": str(checkpoint)
                    }
                    print("skipped task2/reranker: existing checkpoint loaded", flush=True)
                    checkpoint = None
            if checkpoint is not None:
                summary["models"]["task2/reranker"] = train_task2_oof_reranker(
                    blocks, config, images, output, device
                )
        del blocks
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_final(config, models, output):
    images = (load_images(DATA) >= 128).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions_by_task = {}
    ids_by_task = {}
    for task in ("task1", "task2", "task3"):
        task_config = config[task]
        seeds = task_config.get("seeds", [42])
        blocks = load_feature_blocks(DATA, "test", task=task, bin_ms=task_config["bin_ms"])
        task_predictions = []
        task_ids = []
        models_by_mouse = {}
        reranker = None
        reranker_config = task_config.get("reranker", {})
        if task == "task2" and reranker_config.get("enabled", False):
            reranker = load_listwise_reranker(output / "models/listwise_cnn.pt", device)
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
            if block.mouse not in models_by_mouse:
                load = load_regressor if task == "task1" else load_classifier
                models_by_mouse[block.mouse] = [
                    load(checkpoint_path(models, task, block.mouse, seed), device) for seed in seeds
                ]
            mouse_models = models_by_mouse[block.mouse]
            if task == "task1":
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
            else:
                if task == "task2":
                    raw_counts = block.aggregate()
                    normalizer = fit_session_normalizer(block.features)
                else:
                    raw_counts = block.features
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
                indices = unique_assignment(scores) if task_config["assignment"] else scores.argmax(axis=1)
                predictions = images[indices]
            task_predictions.append(predictions.astype(np.float32))
            task_ids.extend(block.sample_ids.tolist())
            print(f"predicted {task}/{block.key}: {len(predictions)}", flush=True)
        predictions_by_task[task] = task_predictions
        ids_by_task[task] = task_ids
        del models_by_mouse, blocks, reranker
        release_memory(device)
    if sum(len(ids) for ids in ids_by_task.values()) != 30200:
        raise ValueError("Prediction IDs do not total 30,200.")
    png_folder = output / "png"
    if png_folder.exists():
        raise FileExistsError(f"Refuse to mix predictions with an existing folder: {png_folder}")
    for task in ("task1", "task2", "task3"):
        task_folder = png_folder / task
        task_folder.mkdir(parents=True, exist_ok=False)
        task_array = np.concatenate(predictions_by_task[task]).astype(np.float32)[:, None]
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
        "kaggle_sha256": hashlib.sha256(kaggle_archive.read_bytes()).hexdigest(),
        "samples": 30200,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/final_config.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--models-run-name", help="Reuse models from an earlier submission run.")
    parser.add_argument("--skip-train", action="store_true", help="Skip all model training.")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Train only missing or unloadable checkpoints."
    )
    args = parser.parse_args()
    if args.skip_train and args.skip_existing:
        parser.error("--skip-existing cannot be used with --skip-train.")
    if args.models_run_name and not args.skip_train:
        parser.error("--models-run-name requires --skip-train.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_output = ROOT / "test/90_submission/artifacts" / args.run_name
    models_run_name = args.models_run_name or args.run_name
    models = ROOT / "test/90_submission/artifacts" / models_run_name / "models"
    if not args.skip_train:
        train_final(config, models, skip_existing=args.skip_existing)
    result = predict_final(config, models, run_output)
    result.update({
        "name": config["name"],
        "models_run_name": models_run_name,
        "task2_reranker_checkpoint": str(run_output / "models/listwise_cnn.pt")
            if config["task2"].get("reranker", {}).get("enabled", False) else None,
    })
    (run_output / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
