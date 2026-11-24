"""Train the selected full-data models and build a complete Kaggle submission."""

import argparse
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
    load_classifier,
    load_regressor,
    predict_logits,
)
from src.retrieval import render_gb2312_bank, retrieve_glyphs, unique_assignment
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


def fit_full(model, features, targets, output, *, epochs, regression, learning_rate=3e-4,
             weight_decay=1e-3, batch_size=256, device="cuda"):
    """Fit all permitted data for a fixed CV-selected number of epochs."""
    torch.manual_seed(42)
    np.random.seed(42)
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
    (output.parent / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def train_final(config, output):
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
            if task == "task1":
                train_y = dct_encode(images[train_targets], 32)
                model = SpikeRegressor(
                    train_x.shape[1], train_y.shape[1], task_config["hidden"], task_config["dropout"]
                )
                checkpoint = output / task / mouse / "regressor.pt"
            else:
                train_y = train_targets
                model = build_classifier(
                    task_config, train_x.shape[1], mouse_blocks[0].n_units, round(250 / task_config["bin_ms"])
                )
                checkpoint = output / task / mouse / "classifier.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                for session, normalizer in normalizers.items():
                    np.savez(output / task / mouse / f"{session}_normalizer.npz", mean=normalizer[0], scale=normalizer[1])
                encoding_path = RidgePath().fit(
                    bank_coefficients[np.concatenate(encoding_target_parts)], np.concatenate(encoding_neural_parts)
                )
                prototypes = encoding_path.predict(bank_coefficients, task_config["encoding_alpha"])
                np.save(output / task / mouse / "encoding_prototypes.npy", prototypes)
            history = fit_full(
                model,
                train_x,
                train_y,
                checkpoint,
                epochs=task_config["epochs"][mouse],
                regression=task == "task1",
                weight_decay=task_config["weight_decay"],
                device=device,
            )
            summary["models"][key] = {
                "samples": len(train_x),
                "epochs": task_config["epochs"][mouse],
                "final_train_loss": history[-1]["train_loss"],
                "checkpoint": str(checkpoint),
            }
            del model, train_x, train_targets, train_y
            if device == "cuda":
                torch.cuda.empty_cache()
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
        blocks = load_feature_blocks(DATA, "test", task=task, bin_ms=task_config["bin_ms"])
        task_predictions = []
        task_ids = []
        models_by_mouse = {}
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
                if task == "task1":
                    models_by_mouse[block.mouse] = load_regressor(models / task / block.mouse / "regressor.pt", device)
                else:
                    models_by_mouse[block.mouse] = load_classifier(models / task / block.mouse / "classifier.pt", device)
            model = models_by_mouse[block.mouse]
            if task == "task1":
                counts = block.aggregate()
                coefficients = predict_logits(model, counts, device=device)
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
                mlp_scores = predict_logits(model, counts, device=device)
                prototypes = np.load(models / task / block.mouse / "encoding_prototypes.npy", allow_pickle=False)
                encoding_scores = response_scores(np.log1p(raw_counts), prototypes, metric="mse", device=device)
                weight = task_config.get("mlp_weight", 1.0)
                scores = weight * row_zscore(mlp_scores) + (1 - weight) * row_zscore(encoding_scores)
                indices = unique_assignment(scores) if task_config["assignment"] else scores.argmax(axis=1)
                predictions = images[indices]
            task_predictions.append(predictions.astype(np.float32))
            task_ids.extend(block.sample_ids.tolist())
            print(f"predicted {task}/{block.key}: {len(predictions)}", flush=True)
        predictions_by_task[task] = task_predictions
        ids_by_task[task] = task_ids
        del models_by_mouse, blocks
        if device == "cuda":
            torch.cuda.empty_cache()
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
    return {"png_archive": str(png_archive), "kaggle_archive": str(kaggle_archive), "samples": 30200}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/final_config.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--models-run-name", help="Reuse models from an earlier submission run.")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_output = ROOT / "test/90_submission/artifacts" / args.run_name
    models_run_name = args.models_run_name or args.run_name
    models = ROOT / "test/90_submission/artifacts" / models_run_name / "models"
    if not args.skip_train:
        if args.models_run_name:
            parser.error("--models-run-name requires --skip-train.")
        train_final(config, models)
    result = predict_final(config, models, run_output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
