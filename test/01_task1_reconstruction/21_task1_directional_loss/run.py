"""Task1 seed42: replace only DCT-MSE with differentiable decoded directional2 cosine loss."""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dct
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import SpikeRegressor, load_regressor, predict_logits
from src.ocf.ocf_cuda import kernel_spectrum
from src.retrieval import _retrieval_embedding, render_gb2312_bank, retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUTPUT_DIR = ROOT / "test/01_task1_reconstruction/21_task1_directional_loss"
ARTIFACTS = OUTPUT_DIR / "artifacts"
MODELS = ARTIFACTS / "models" / "seed_42_directional_cosine"
RESULTS_PATH = OUTPUT_DIR / "results_task1_directional_loss.json"
TUNE_NPZ = ARTIFACTS / "task1_directional_loss_tune_predictions.npz"
HELD_OUT_NPZ = ARTIFACTS / "task1_directional_loss_held_out_predictions.npz"
BASELINE_ROOT = ROOT / "test/01_task1_reconstruction/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
SEED = 42
EPOCHS = {"mice1": 8, "mice2": 11}
BASELINE_HELD_OUT_OCF = 18.624308986308705
BASELINE_TUNE_OCF = 16.811784552406262
BASELINE_ATOL = 1e-5
DCT_ATOL = 1e-4


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_task1(images):
    """Copy experiment18's Task1 rows, replacing stored DCT targets with their picture IDs."""
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
            })
        train_x = np.concatenate(train_x_parts)
        train_targets = np.concatenate(train_target_parts).astype(np.int64, copy=False)
        validation_x_parts = []
        validation_target_parts = []
        for validation_x, validation_targets in validation_rows:
            mask = np.isin(validation_targets, model_validation_pictures)
            validation_x_parts.append(validation_x[mask])
            validation_target_parts.append(validation_targets[mask])
        training[mouse] = {
            "train": (train_x, train_targets),
            "validation": (
                np.concatenate(validation_x_parts),
                np.concatenate(validation_target_parts).astype(np.int64, copy=False),
            ),
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


def make_dct_basis(device):
    """Return the first 32 rows of the orthonormal 64-point DCT-II matrix."""
    basis = dct(np.eye(64, dtype=np.float32), axis=0, norm="ortho")[:32].copy()
    return torch.from_numpy(basis).to(device=device, dtype=torch.float32)


def torch_dct_decode(coefficients, basis):
    """Differentiably decode N x 1024 coefficients as D_low.T @ C @ D_low."""
    square = coefficients.float().reshape(-1, 32, 32)
    return torch.einsum("yu,nuv,vx->nyx", basis.T, square, basis)


def verify_dct(device):
    """Compare the training decode with src.ridge.dct_decode on fixed random coefficients."""
    rng = np.random.default_rng(42)
    coefficients = rng.standard_normal((8, 1024), dtype=np.float32)
    reference = dct_decode(coefficients, 32)
    basis = make_dct_basis(device)
    tensor = torch.from_numpy(coefficients).to(device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        actual = torch_dct_decode(tensor, basis).cpu().numpy()
    max_abs_error = float(np.max(np.abs(actual - reference)))
    record = {
        "random_seed": 42,
        "random_coefficient_shape": list(coefficients.shape),
        "distribution": "standard normal float32",
        "reference": "src.ridge.dct_decode(coefficients, 32), scipy idctn norm=ortho",
        "implementation": "precomputed orthonormal DCT matrix; differentiable float32 torch.einsum",
        "device": str(device),
        "output_dtype": str(actual.dtype),
        "max_abs_error": max_abs_error,
        "absolute_tolerance": DCT_ATOL,
        "passed": max_abs_error <= DCT_ATOL,
    }
    print(f"DCT VERIFICATION {json.dumps(record)}", flush=True)
    if not record["passed"]:
        raise RuntimeError(f"Torch DCT decode mismatch: {record}")
    return record, basis


def directional_embedding(images):
    """Exactly match directional2 retrieval: metric embedding followed by overall normalization."""
    return F.normalize(_retrieval_embedding(images, "directional2"), dim=1)


@torch.no_grad()
def precompute_ground_truth_embeddings(images, device, batch_size=32):
    parts = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start:start + batch_size, None]).to(device=device, dtype=torch.float32)
        parts.append(directional_embedding(batch))
    embeddings = torch.cat(parts, dim=0)
    if embeddings.shape != (2748, 12288):
        raise RuntimeError(f"Unexpected directional2 ground-truth embedding shape: {tuple(embeddings.shape)}")
    print(f"GT EMBEDDINGS shape={tuple(embeddings.shape)} dtype={embeddings.dtype}", flush=True)
    return embeddings


def decoded_directional_loss(coefficients, target_ids, basis, ground_truth_embeddings):
    """Pure cosine loss on unclipped decoded images, with every decoded/FFT operation in float32."""
    with torch.autocast(device_type=coefficients.device.type, enabled=False):
        decoded = torch_dct_decode(coefficients.float(), basis)
        prediction_embedding = directional_embedding(decoded[:, None])
        target_embedding = ground_truth_embeddings[target_ids]
        cosine = (prediction_embedding * target_embedding).sum(dim=1)
        return (1.0 - cosine).mean()


def fit_directional_regressor(model, train, validation, output, *, epochs, basis,
                              ground_truth_embeddings, device, seed=42):
    """Match fit_regressor's protocol while replacing only its training and validation loss."""
    kernel_spectrum.cache_clear()
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_targets = train
    val_x, val_targets = validation
    model.set_normalization(train_x)
    model.to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_x, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_targets, dtype=np.int64)),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_features = torch.from_numpy(np.asarray(val_x, dtype=np.float32))
    val_target_ids = torch.from_numpy(np.asarray(val_targets, dtype=np.int64))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, fused=device.type == "cuda"
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, target_ids in train_loader:
            features = features.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                predictions = model(features)
            loss = decoded_directional_loss(predictions, target_ids, basis, ground_truth_embeddings)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite directional loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(target_ids)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_features), 256):
                features = val_features[start:start + 256].to(device)
                target_ids = val_target_ids[start:start + 256].to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    predictions = model(features)
                loss = decoded_directional_loss(predictions, target_ids, basis, ground_truth_embeddings)
                val_loss += loss.item() * len(target_ids)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_x),
            "validation_loss": val_loss / len(val_x),
        }
        history.append(row)
        print(f"TRAIN {output.parent.name}/{output.name} {json.dumps(row)}", flush=True)
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            stale_epochs = 0
            torch.save({
                "n_features": model.n_features,
                "n_outputs": model.n_outputs,
                "hidden": model.hidden,
                "dropout": model.dropout,
                "apply_log": model.apply_log,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": best_loss,
            }, output / "regressor.pt")
        else:
            stale_epochs += 1
        write_json(output / "history.json", history)
        if stale_epochs >= 10:
            break
    return history


def train_candidate(data, basis, ground_truth_embeddings, device):
    records = []
    for mouse in sorted(data["training"]):
        train = data["training"][mouse]["train"]
        validation = data["training"][mouse]["validation"]
        output = MODELS / mouse
        print(
            f"CANDIDATE TRAIN mouse={mouse} epochs={EPOCHS[mouse]} "
            f"train={len(train[0])} validation={len(validation[0])}",
            flush=True,
        )
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = SpikeRegressor(train[0].shape[1], 1024, hidden=512, dropout=0.4)
        history = fit_directional_regressor(
            model,
            train,
            validation,
            output,
            epochs=EPOCHS[mouse],
            basis=basis,
            ground_truth_embeddings=ground_truth_embeddings,
            device=device,
            seed=SEED,
        )
        best = min(history, key=lambda row: row["validation_loss"])
        records.append({
            "variant": "decoded_directional2_cosine",
            "seed": SEED,
            "mouse": mouse,
            "epochs_requested": EPOCHS[mouse],
            "epochs_run": len(history),
            "train_rows": len(train[0]),
            "validation_rows": len(validation[0]),
            "validation_unique_pictures": 64,
            "checkpoint_epoch": best["epoch"],
            "best_validation_loss": best["validation_loss"],
            "checkpoint": str((output / "regressor.pt").relative_to(ROOT)),
            "history": history,
        })
        del model
        release_memory()
    return records


def evaluate_models(data, checkpoints, pictures, device):
    """Decode without clipping and run the unchanged full-bank directional2 retrieval."""
    prediction_parts = []
    target_parts = []
    group_parts = []
    picture_parts = []
    candidate_parts = []
    continuous_min = float("inf")
    continuous_max = float("-inf")
    for mouse in sorted(checkpoints):
        model = load_regressor(checkpoints[mouse], device)
        for row in data["rows"]:
            if row["mouse"] != mouse:
                continue
            mask = np.isin(row["targets"], pictures)
            coefficients = predict_logits(model, row["counts"], device=device)
            continuous = dct_decode(coefficients, 32)[mask]
            continuous_min = min(continuous_min, float(continuous.min()))
            continuous_max = max(continuous_max, float(continuous.max()))
            predictions, candidate_indices = retrieve_glyphs(
                continuous, data["bank"], metric="directional2", blur_sigma=1.0, device=device
            )
            prediction_parts.append(predictions)
            target_parts.append(data["images"][row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
            picture_parts.append(row["targets"][mask])
            candidate_parts.append(candidate_indices)
        del model
        release_memory()
    predictions = np.concatenate(prediction_parts)
    references = np.concatenate(target_parts)
    groups = np.asarray(group_parts)
    scores = score_predictions(predictions, references, device=device, batch_size=16)
    summary = summarize_scores(scores, groups)
    summary["continuous_prediction_range"] = [continuous_min, continuous_max]
    summary["continuous_predictions_clipped"] = False
    details = {
        "predictions": predictions,
        "references": references,
        "groups": groups,
        "pictures": np.concatenate(picture_parts),
        "candidate_indices": np.concatenate(candidate_parts),
    }
    return summary, details


def verify_baseline(tune_summary, held_out_summary):
    tune_error = abs(tune_summary["ocf_mean"] - BASELINE_TUNE_OCF)
    held_out_error = abs(held_out_summary["ocf_mean"] - BASELINE_HELD_OUT_OCF)
    record = {
        "checkpoint_root": str(BASELINE_ROOT.relative_to(ROOT)),
        "expected_tune_ocf": BASELINE_TUNE_OCF,
        "actual_tune_ocf": tune_summary["ocf_mean"],
        "tune_absolute_error": tune_error,
        "expected_held_out_ocf": BASELINE_HELD_OUT_OCF,
        "actual_held_out_ocf": held_out_summary["ocf_mean"],
        "held_out_absolute_error": held_out_error,
        "absolute_tolerance": BASELINE_ATOL,
        "passed": tune_error <= BASELINE_ATOL and held_out_error <= BASELINE_ATOL,
    }
    print(f"BASELINE VERIFICATION {json.dumps(record)}", flush=True)
    if not record["passed"]:
        raise RuntimeError(f"Seed42 baseline mismatch; stopping before candidate training: {record}")
    return record


def save_prediction_npz(path, baseline_details, candidate_details=None, selected_variant=None):
    arrays = {
        "baseline_predictions": baseline_details["predictions"].astype(np.uint8) * 255,
        "targets": baseline_details["references"].astype(np.uint8) * 255,
        "group_ids": baseline_details["groups"],
        "picture_indices": baseline_details["pictures"],
        "baseline_candidate_indices": baseline_details["candidate_indices"],
        "candidate_evaluated": np.asarray(candidate_details is not None),
    }
    if candidate_details is not None:
        arrays.update({
            "candidate_predictions": candidate_details["predictions"].astype(np.uint8) * 255,
            "candidate_candidate_indices": candidate_details["candidate_indices"],
        })
    if selected_variant is not None:
        arrays["selected_variant"] = np.asarray(selected_variant)
    np.savez_compressed(path, **arrays)


def main():
    started = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE {device}; seed={SEED}; candidate=decoded_directional2_cosine", flush=True)
    dct_verification, basis = verify_dct(device)
    images = (load_images(DATA) >= 128).astype(np.float32)
    if len(images) != 2748:
        raise ValueError(f"Expected 2748 training pictures, found {len(images)}.")
    data = prepare_task1(images)
    data["bank"] = render_gb2312_bank(FONT_PATH, font_size=52, variation="Regular")
    if data["bank"].shape != (6763, 64, 64):
        raise RuntimeError(f"Unexpected GB2312 bank shape: {data['bank'].shape}")
    if (len(data["model_validation_pictures"]), len(data["tune_pictures"]), len(data["evaluation_pictures"])) != (64, 16, 211):
        raise RuntimeError("Task1 picture partition differs from the fixed 64/16/211 protocol.")

    baseline_checkpoints = {mouse: BASELINE_ROOT / mouse / "regressor.pt" for mouse in sorted(data["training"])}
    baseline_held_out, baseline_held_out_details = evaluate_models(
        data, baseline_checkpoints, data["evaluation_pictures"], device
    )
    baseline_tune, baseline_tune_details = evaluate_models(
        data, baseline_checkpoints, data["tune_pictures"], device
    )
    baseline_verification = verify_baseline(baseline_tune, baseline_held_out)

    kernel_spectrum.cache_clear()
    ground_truth_embeddings = precompute_ground_truth_embeddings(images, device)
    training = train_candidate(data, basis, ground_truth_embeddings, device)
    candidate_checkpoints = {mouse: MODELS / mouse / "regressor.pt" for mouse in sorted(data["training"])}
    candidate_tune, candidate_tune_details = evaluate_models(
        data, candidate_checkpoints, data["tune_pictures"], device
    )
    print(
        f"TUNE baseline={baseline_tune['ocf_mean']:.12f} candidate={candidate_tune['ocf_mean']:.12f}",
        flush=True,
    )

    candidate_wins = candidate_tune["ocf_mean"] > BASELINE_TUNE_OCF
    selected_variant = "candidate_decoded_directional2_cosine" if candidate_wins else "baseline_equal_weight_dct_mse"
    tuning = [
        {"variant": "baseline_equal_weight_dct_mse", **baseline_tune},
        {"variant": "candidate_decoded_directional2_cosine", **candidate_tune},
    ]
    selection = {
        "metric": "tune official OCF mean only",
        "candidate_win_rule": f"candidate tune OCF strictly greater than {BASELINE_TUNE_OCF}",
        "candidate_wins": candidate_wins,
        "selected_variant": selected_variant,
        "held_out_used_for_selection": False,
    }
    save_prediction_npz(TUNE_NPZ, baseline_tune_details, candidate_tune_details)

    candidate_held_out = None
    candidate_held_out_details = None
    candidate_not_evaluated_reason = None
    if candidate_wins:
        candidate_held_out, candidate_held_out_details = evaluate_models(
            data, candidate_checkpoints, data["evaluation_pictures"], device
        )
        selected_held_out = candidate_held_out
    else:
        candidate_not_evaluated_reason = (
            f"Candidate tune OCF {candidate_tune['ocf_mean']} <= predefined baseline tune OCF "
            f"{BASELINE_TUNE_OCF}; candidate held-out retrieval and OCF were not computed, as required."
        )
        selected_held_out = baseline_held_out
        print(candidate_not_evaluated_reason, flush=True)
    save_prediction_npz(
        HELD_OUT_NPZ,
        baseline_held_out_details,
        candidate_held_out_details,
        selected_variant=selected_variant,
    )
    held_out = {
        "baseline": baseline_held_out,
        "candidate": candidate_held_out,
        "candidate_ocf_mean": None if candidate_held_out is None else candidate_held_out["ocf_mean"],
        "candidate_not_evaluated_reason": candidate_not_evaluated_reason,
        "selected_variant": selected_variant,
        "selected_evaluation": selected_held_out,
    }
    protocol = {
        "reused_from": "test/06_fusion_transduction/18_seed_ensemble Task1 seed42 protocol",
        "split_seed": 42,
        "picture_seed": 123,
        "model_validation_pictures": data["model_validation_pictures"].tolist(),
        "model_validation_picture_count": len(data["model_validation_pictures"]),
        "tune_pictures": data["tune_pictures"].tolist(),
        "tune_picture_count": len(data["tune_pictures"]),
        "evaluation_pictures": data["evaluation_pictures"].tolist(),
        "evaluation_picture_count": len(data["evaluation_pictures"]),
        "validation_picture_count": data["validation_picture_count"],
        "held_out_rule": (
            "Reproduce the existing baseline held-out OCF before training. Candidate held-out retrieval/OCF is "
            "computed only if its tune OCF strictly beats the predefined baseline; held-out results never select."
        ),
    }
    config = {
        "task": "task1",
        "seed": 42,
        "bin_ms": 25,
        "dct_cutoff": 32,
        "n_outputs": 1024,
        "hidden": 512,
        "dropout": 0.4,
        "weight_decay": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "epochs": EPOCHS,
        "optimizer": "AdamW (fused on CUDA), gradient norm clipped to 5.0",
        "early_stopping_patience": 10,
        "training_rows": "per-session four-repeat means followed by single trials, pooled by mouse exactly as experiment18",
        "model_validation": "the same 64 shuffled[:64] pictures, aggregated over four sessions per mouse",
        "baseline": "existing equal-weight DCT-MSE seed42 checkpoints; no retraining",
        "candidate": "one predefined candidate; pure mean(1-directional2 cosine), no DCT-MSE term and no lambda search",
        "loss_decode": "unclipped differentiable float32 orthonormal DCT32 decode",
        "loss_embedding": "src.retrieval._retrieval_embedding(..., 'directional2') followed by overall F.normalize(dim=1)",
        "autograd_kernel_cache": "inference-created Gaussian spectrum cache is cleared before autograd; identical float32 kernels are recomputed normally",
        "ground_truth_embeddings": "all 2748 GT images precomputed once in float32 and indexed by target picture ID",
        "model_forward": "original CUDA bfloat16 autocast; decoded image, FFT and cosine loss in float32",
        "checkpoint_selection": "minimum directional cosine validation_loss on the model-validation 64 pictures",
        "checkpoint_format": "src.neural.load_regressor compatible",
        "continuous_predictions": "DCT32 decoded without clipping before directional2 retrieval",
        "evaluation_inference": "predict and decode each complete 275-picture validation block before tune/held-out subsetting, exactly as experiment18",
        "retrieval_bank": "complete strict-GB2312 bank, NotoSansSC Regular size 52, 6763 glyphs",
        "font_path": str(FONT_PATH),
        "retrieval_metric": "directional2",
        "retrieval_blur_sigma": 1.0,
        "selection_metric": "tune official OCF mean only",
    }
    result = {
        "experiment": "21_task1_directional_loss",
        "name": "task1_seed42_decoded_directional2_cosine_loss",
        "score_scope": "fixed local validation protocol only; not a Kaggle score and not leaderboard-comparable",
        "config": config,
        "protocol": protocol,
        "dct_verification": dct_verification,
        "baseline_verification": baseline_verification,
        "training": training,
        "tuning": tuning,
        "selection": selection,
        "held_out": held_out,
        "held_out_delta_ocf": selected_held_out["ocf_mean"] - baseline_held_out["ocf_mean"],
        "artifacts": {
            "tune_predictions_npz": str(TUNE_NPZ.relative_to(ROOT)),
            "held_out_predictions_npz": str(HELD_OUT_NPZ.relative_to(ROOT)),
            "candidate_checkpoint_root": str(MODELS.relative_to(ROOT)),
        },
        "leaderboard_reference": {
            "rank_1": 54.01486,
            "comparable": False,
            "statement": "Different local held-out protocol and predictions; scores cannot be compared with Kaggle leaderboard values.",
        },
        "Kaggle_baseline": 21.09590,
        "limitations": [
            "Pure cosine training leaves decoded-image amplitude unconstrained; retrieved binary glyphs determine OCF.",
            "Only the predefined seed42 candidate was tested; no MSE regularizer, lambda sweep, clipping, or other variant was introduced.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RESULTS_PATH, result)
    print(
        f"FINAL selected={selected_variant} tune_candidate={candidate_tune['ocf_mean']:.12f} "
        f"held_out_candidate={held_out['candidate_ocf_mean']} results={RESULTS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
