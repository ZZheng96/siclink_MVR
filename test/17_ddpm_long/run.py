"""Extend test07's official DDPM to 100 epochs with tune-only checkpoint selection."""

from copy import deepcopy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.ddpm import build_model, diffusion_scheduler, sample_images, training_loss
from src.evaluation import score_predictions, summarize_scores, sweep_thresholds
from src.features import aggregate_by_target, load_feature_blocks
from src.retrieval import retrieve_glyphs
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUTPUT_DIR = ROOT / "test/17_ddpm_long"
ARTIFACTS = OUTPUT_DIR / "artifacts"
CHECKPOINT_EPOCHS = (25, 50, 100)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BASELINE_OCF = 18.4831207873
CONFIG = {
    "task": "task1",
    "reference_implementation": "test/07_ddpm/run.py",
    "model_implementation": "src/ddpm.py",
    "bin_ms": 50,
    "mice": ["mice1", "mice2"],
    "pooling": "separate per-mouse models with permitted training sessions pooled",
    "epochs": 100,
    "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
    "batch_size": 64,
    "sample_batch_size": 64,
    "ocf_batch_size": 16,
    "image_binarization_threshold": 128,
    "split_seed": 42,
    "tune_split_seed": 123,
    "tune_picture_count": 16,
    "evaluation_picture_count": 211,
    "noise_seed": 742,
    "diffusion_steps": 64,
    "beta_schedule": "squaredcos_cap_v2",
    "prediction_type": "v_prediction",
    "min_snr_gamma": 5,
    "condition_dropout": 0.1,
    "classifier_free_guidance_scale": 1.5,
    "ema_decay": 0.999,
    "optimizer": "AdamW",
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "direct_thresholds": list(THRESHOLDS),
    "retrieval_metric": "directional2",
    "retrieval_blur_sigma": 1.0,
    "selection_metric": "tuning retrieval OCF",
}


def capture_rng():
    """Capture every RNG state needed to continue test07-style training."""
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "bit_generator": numpy_state[0],
            "key": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "python": random.getstate(),
    }


def restore_rng(state):
    """Restore RNGs after model initialization or read-only checkpoint sampling."""
    torch.set_rng_state(state["torch_cpu"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["key"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    random.setstate(state["python"])
    if torch.cuda.is_available():
        for index, cuda_state in enumerate(state["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=index)


def write_json(path, payload):
    """Atomically retain small history and result JSON files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_path(mouse, epoch):
    return ARTIFACTS / "checkpoints" / mouse / f"epoch_{epoch:03d}.pt"


def history_path(mouse):
    return ARTIFACTS / "history" / f"{mouse}.json"


def write_history(mouse, history):
    write_json(history_path(mouse), {
        "experiment": "17_ddpm_long",
        "mouse": mouse,
        "config": CONFIG,
        "history": history,
    })


def read_checkpoint(mouse, epoch, inputs):
    """Load only a checkpoint matching this experiment and expected split configuration."""
    path = checkpoint_path(mouse, epoch)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "mouse", "inputs", "epoch", "config", "model_state_dict",
        "ema_state_dict", "optimizer_state_dict", "history", "rng",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing checkpoint fields {sorted(missing)}")
    if payload["format"] != "17_ddpm_long_v1":
        raise ValueError(f"{path}: unsupported checkpoint format")
    if payload["mouse"] != mouse or payload["inputs"] != inputs:
        raise ValueError(f"{path}: mouse or feature dimension mismatch")
    if payload["epoch"] != epoch or epoch not in CHECKPOINT_EPOCHS:
        raise ValueError(f"{path}: unexpected checkpoint epoch")
    if payload["config"] != CONFIG:
        raise ValueError(f"{path}: configuration mismatch")
    history = payload["history"]
    if [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError(f"{path}: incomplete or inconsistent history")
    return payload


def save_checkpoint(mouse, inputs, epoch, model, ema, optimizer, history):
    """Save an independently resumable model/EMA/optimizer/RNG checkpoint."""
    path = checkpoint_path(mouse, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "17_ddpm_long_v1",
        "mouse": mouse,
        "inputs": inputs,
        "epoch": epoch,
        "config": CONFIG,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "rng": capture_rng(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"CHECKPOINT {mouse} epoch={epoch}: {path}", flush=True)


def prepare_mouse(group_blocks, split, images):
    """Reproduce test07's pooled training rows and per-block validation aggregation."""
    train_x_parts = []
    train_y_parts = []
    validation_x_parts = []
    validation_y_parts = []
    validation_keys = []
    for block in group_blocks:
        block_split = split.blocks[block.key]
        if len(block_split.train_trials):
            train_x_parts.append(block.features[block_split.train_trials])
            train_y_parts.append(block.trial_targets[block_split.train_trials])
        if len(block_split.validation_trials):
            validation_targets, validation_x = aggregate_by_target(
                block.features[block_split.validation_trials],
                block.trial_targets[block_split.validation_trials],
            )
            validation_x_parts.append(validation_x)
            validation_y_parts.append(validation_targets)
            validation_keys.extend([block.key] * len(validation_targets))
    train_x = np.concatenate(train_x_parts)
    train_targets = np.concatenate(train_y_parts)
    validation_x = np.concatenate(validation_x_parts)
    validation_targets = np.concatenate(validation_y_parts)
    train_images = images[train_targets][:, None]
    return train_x, train_images, validation_x, validation_targets, np.asarray(validation_keys)


def build_training_state(inputs, device, payload=None):
    """Build exactly test07's denoiser, EMA, scheduler, and optimizer."""
    model = build_model(inputs).to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    scheduler = diffusion_scheduler()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=0.01, fused=device.type == "cuda"
    )
    if payload is not None:
        model.load_state_dict(payload["model_state_dict"])
        ema.load_state_dict(payload["ema_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, ema, scheduler, optimizer


def train_mouse(mouse, train_x, train_images, device):
    """Train to epoch 100, resuming the newest valid checkpoint when available."""
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_images))
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    initial_rng = capture_rng()
    state = None
    history = []
    start_epoch = 0
    for epoch in reversed(CHECKPOINT_EPOCHS):
        path = checkpoint_path(mouse, epoch)
        if not path.is_file():
            continue
        try:
            payload = read_checkpoint(mouse, epoch, train_x.shape[1])
            candidate = build_training_state(train_x.shape[1], device, payload)
            restore_rng(payload["rng"])
        except Exception as error:
            restore_rng(initial_rng)
            print(f"RESUME rejected {path}: {error}", flush=True)
            continue
        state = candidate
        history = payload["history"]
        start_epoch = epoch
        print(f"RESUME {mouse} epoch={epoch}: {path}", flush=True)
        break
    if state is None:
        restore_rng(initial_rng)
        state = build_training_state(train_x.shape[1], device)
        print(f"TRAIN {mouse}: initialize from seed 42 protocol", flush=True)

    model, ema, scheduler, optimizer = state
    for epoch in range(start_epoch + 1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for counts, targets in loader:
            counts = counts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).mul_(2).sub_(1)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, scheduler, targets, counts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 0.001)
            total_loss += loss.item() * len(counts)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_x)})
        print(f"TRAIN {mouse} {history[-1]}", flush=True)
        write_history(mouse, history)
        if epoch in CHECKPOINT_EPOCHS:
            save_checkpoint(
                mouse, train_x.shape[1], epoch, model, ema, optimizer, history
            )
    write_history(mouse, history)


def sample_checkpoint(mouse, epoch, inputs, validation_x, device):
    """Sample every validation row with test07's 64-step CFG and fixed noise seed."""
    preserved_rng = capture_rng()
    try:
        payload = read_checkpoint(mouse, epoch, inputs)
        model = build_model(inputs).to(device, memory_format=torch.channels_last)
        model.load_state_dict(payload["ema_state_dict"])
        model.eval().requires_grad_(False)
        scheduler = diffusion_scheduler()
        prediction_parts = []
        generator = torch.Generator(device=device).manual_seed(742)
        with torch.inference_mode():
            for start in range(0, len(validation_x), 64):
                counts = torch.from_numpy(validation_x[start:start + 64]).to(
                    device, dtype=torch.float32
                )
                samples = sample_images(model, scheduler, counts, generator)
                prediction_parts.append(samples[:, 0].float().cpu().numpy())
        return np.concatenate(prediction_parts)
    finally:
        restore_rng(preserved_rng)


def collect(rows, pictures, images):
    """Collect rows exactly as in test07 after all per-mouse sampling is complete."""
    predictions = []
    targets = []
    groups = []
    for row in rows:
        mask = np.isin(row["targets"], pictures)
        predictions.append(row["predictions"][mask])
        targets.append(images[row["targets"][mask]])
        groups.extend(row["groups"][mask].tolist())
    return np.concatenate(predictions), np.concatenate(targets), np.asarray(groups)


def tune_checkpoint(epoch, rows, tune_pictures, images, bank, device):
    """Tune direct binarization and measure retrieval using only the 16 tune pictures."""
    tune_predictions, tune_targets, tune_groups = collect(rows, tune_pictures, images)
    threshold_rows = sweep_thresholds(
        tune_predictions, tune_targets, THRESHOLDS,
        device=str(device), batch_size=16,
    )
    best_threshold = max(threshold_rows, key=lambda row: row["ocf_mean"])
    direct_tune_scores = score_predictions(
        tune_predictions, tune_targets, threshold=best_threshold["threshold"],
        device=str(device), batch_size=16,
    )
    direct_tune_summary = summarize_scores(direct_tune_scores, tune_groups)

    # Match test07's order: retrieval receives raw continuous samples, not thresholded direct output.
    retrieved_tune, _ = retrieve_glyphs(
        tune_predictions, bank, metric="directional2", blur_sigma=1.0, device=str(device)
    )
    retrieval_tune_scores = score_predictions(
        retrieved_tune, tune_targets, device=str(device), batch_size=16
    )
    retrieval_tune_summary = summarize_scores(retrieval_tune_scores, tune_groups)
    tuning = {
        "epoch": epoch,
        "direct": {
            "threshold_sweep": threshold_rows,
            "selected_threshold": best_threshold,
            "summary": direct_tune_summary,
        },
        "retrieval": retrieval_tune_summary,
    }
    print(
        f"TUNE epoch={epoch}: direct={direct_tune_summary['ocf_mean']:.6f}; "
        f"retrieval={retrieval_tune_summary['ocf_mean']:.6f}; "
        f"threshold={best_threshold['threshold']}",
        flush=True,
    )
    return tuning


def evaluate_held_out(rows, evaluation_pictures, images, bank, device, threshold):
    """Evaluate direct and retrieval OCF on eval211 only after checkpoint selection."""
    evaluation_predictions, evaluation_targets, evaluation_groups = collect(
        rows, evaluation_pictures, images
    )
    direct_scores = score_predictions(
        evaluation_predictions, evaluation_targets, threshold=threshold,
        device=str(device), batch_size=16,
    )
    direct_summary = summarize_scores(direct_scores, evaluation_groups)
    retrieved, _ = retrieve_glyphs(
        evaluation_predictions, bank, metric="directional2", blur_sigma=1.0, device=str(device)
    )
    retrieval_scores = score_predictions(
        retrieved, evaluation_targets, device=str(device), batch_size=16
    )
    retrieval_summary = summarize_scores(retrieval_scores, evaluation_groups)
    evaluation = {"direct": direct_summary, "retrieval": retrieval_summary}
    selected_data = {
        "predictions": evaluation_predictions,
        "retrieved": retrieved,
        "targets": evaluation_targets,
        "groups": evaluation_groups,
    }
    return evaluation, selected_data


def main():
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    if len(tune_pictures) != 16 or len(evaluation_pictures) != 211:
        raise RuntimeError("Task1 split does not match test11's 16/211 tune/eval protocol")
    bank_indices = np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))
    bank = images[bank_indices]

    grouped = {
        mouse: [block for block in blocks if block.mouse == mouse]
        for mouse in sorted({block.mouse for block in blocks})
    }
    prediction_rows = {epoch: [] for epoch in CHECKPOINT_EPOCHS}
    row_counts = {}
    for mouse, group_blocks in grouped.items():
        train_x, train_images, validation_x, validation_targets, validation_keys = prepare_mouse(
            group_blocks, split, images
        )
        row_counts[mouse] = {
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
        }
        print(
            f"TRAIN {mouse}: rows={len(train_x)}; validation_rows={len(validation_x)}; "
            f"features={train_x.shape[1]}; device={device}",
            flush=True,
        )
        train_mouse(mouse, train_x, train_images, device)
        for epoch in CHECKPOINT_EPOCHS:
            print(f"SAMPLE {mouse} epoch={epoch}: rows={len(validation_x)}", flush=True)
            predictions = sample_checkpoint(
                mouse, epoch, train_x.shape[1], validation_x, device
            )
            prediction_rows[epoch].append({
                "predictions": predictions,
                "targets": validation_targets,
                "groups": validation_keys,
            })
        del train_x, train_images, validation_x, validation_targets, validation_keys
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tuning = []
    for epoch in CHECKPOINT_EPOCHS:
        tuning.append(tune_checkpoint(
            epoch, prediction_rows[epoch], tune_pictures, images, bank, device
        ))

    # Ties are resolved by earliest checkpoint, never by any evaluation result.
    selected_tuning = max(tuning, key=lambda row: row["retrieval"]["ocf_mean"])
    selected_epoch = selected_tuning["epoch"]
    selected_threshold = selected_tuning["direct"]["selected_threshold"]["threshold"]
    evaluation, chosen = evaluate_held_out(
        prediction_rows[selected_epoch], evaluation_pictures, images, bank, device,
        selected_threshold,
    )
    baseline_delta = evaluation["retrieval"]["ocf_mean"] - BASELINE_OCF
    evaluation = {
        "epoch": selected_epoch,
        **evaluation,
        "retrieval_delta_to_baseline": baseline_delta,
    }
    selected = {
        "epoch": selected_epoch,
        "selection_metric": "tuning.retrieval.ocf_mean",
        "tuning_ocf_mean": selected_tuning["retrieval"]["ocf_mean"],
        "direct_threshold": selected_threshold,
        "tie_break": "earliest checkpoint",
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACTS / "selected_predictions.npz",
        predictions=(chosen["predictions"] >= selected_threshold).astype(np.uint8) * 255,
        retrieval_predictions=chosen["retrieved"].astype(np.uint8) * 255,
        targets=chosen["targets"].astype(np.uint8) * 255,
        group_ids=chosen["groups"],
        epoch=np.asarray(selected_epoch, dtype=np.int64),
    )
    result = {
        "experiment": "17_ddpm_long",
        "name": "task1_ddpm_long",
        "config": CONFIG,
        "protocol": {
            **split.metadata(),
            "tune_pictures": tune_pictures.tolist(),
            "tune_picture_count": len(tune_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "unused_picture_count": len(shuffled[16:64]),
            "pooling": "Per mouse across permitted training sessions",
            "rows_by_mouse": row_counts,
            "training_bank_excludes_validation_pictures": True,
            "noise_protocol": "Reset torch.Generator to 742 for each mouse and checkpoint",
            "direct_protocol": "Threshold selected only on tune16; official OCF reported on eval211",
            "retrieval_protocol": (
                "As in test07, raw continuous samples are retrieved after direct threshold tuning "
                "without applying that threshold; directional2 uses the training-only picture bank"
            ),
            "selection_protocol": (
                "Select epoch only by tune16 retrieval OCF; eval211 is computed only for the "
                "selected checkpoint after selection and is never used for epoch or threshold selection"
            ),
        },
        "tuning": tuning,
        "selected": selected,
        "evaluation": evaluation,
        "baseline_reference": {
            "experiment": "11_retrieval_ablation",
            "metric": "directional2 retrieval OCF",
            "ocf_mean": BASELINE_OCF,
            "note": "Fixed reference from a different model; not an expected DDPM reproduction value",
        },
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "elapsed_seconds": time.time() - started,
    }
    result_path = OUTPUT_DIR / "results_task1_ddpm_long.json"
    write_json(result_path, result)
    print(
        f"FINAL {result_path}: selected_epoch={selected_epoch}; "
        f"tune_retrieval={selected['tuning_ocf_mean']:.6f}; "
        f"direct={evaluation['direct']['ocf_mean']:.6f}; "
        f"retrieval={evaluation['retrieval']['ocf_mean']:.6f}; "
        f"baseline_delta={baseline_delta:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
