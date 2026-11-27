"""P2: prediction-only four-session consensus and grayscale retrieval fusion for Task 1."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import load_regressor, predict_logits
from src.retrieval import _retrieval_embedding, retrieve_glyphs
from src.ridge import dct_decode
from src.splits import make_split

DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
MODEL_ROOT = ROOT / "test/01_task1_reconstruction/05_task1_pooled/artifacts/task1_pooled_mlp_bin25_singles"
MODES = ("independent", "consensus")
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
POST_PREDICTION_SHUFFLE_SEED = 2026
BASELINE_OCF = 18.4831207873
BASELINE_ATOL = 1e-5


@torch.inference_mode()
def hungarian_to_anchor(anchor_predictions, session_predictions, device):
    """Match shuffled session rows to anchor rows using continuous predictions only."""
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.float32)
    session_predictions = np.asarray(session_predictions, dtype=np.float32)
    if anchor_predictions.ndim != 3 or session_predictions.shape != anchor_predictions.shape:
        raise ValueError("Hungarian alignment requires equal-shaped (N, H, W) prediction blocks.")
    anchor_tensor = torch.from_numpy(anchor_predictions[:, None]).to(device)
    session_tensor = torch.from_numpy(session_predictions[:, None]).to(device)
    anchor_embedding = F.normalize(_retrieval_embedding(anchor_tensor, "directional2"), dim=1)
    session_embedding = F.normalize(_retrieval_embedding(session_tensor, "directional2"), dim=1)
    similarity = (session_embedding @ anchor_embedding.T).cpu().numpy()
    session_indices, anchor_indices = linear_sum_assignment(similarity, maximize=True)
    session_to_anchor = np.empty(len(session_predictions), dtype=np.int64)
    session_to_anchor[session_indices] = anchor_indices
    assigned_similarity = similarity[session_indices, anchor_indices]
    diagnostics = {
        "hungarian_mean_cosine": float(assigned_similarity.mean()),
        "hungarian_min_cosine": float(assigned_similarity.min()),
        "hungarian_max_cosine": float(assigned_similarity.max()),
    }
    return session_to_anchor, diagnostics


def four_session_consensus(block_predictions, device):
    """Average at anchor positions, then inverse-map consensus to every session position."""
    if len(block_predictions) != 4:
        raise ValueError(f"Consensus requires exactly four prediction blocks; found {len(block_predictions)}.")
    lengths = {len(predictions) for predictions in block_predictions}
    if len(lengths) != 1:
        raise ValueError(f"Consensus requires equal-length prediction blocks; found lengths {sorted(lengths)}.")

    aligned = [block_predictions[0]]
    session_mappings = []
    diagnostics = []
    for predictions in block_predictions[1:]:
        session_to_anchor, session_diagnostics = hungarian_to_anchor(block_predictions[0], predictions, device)
        anchor_to_session = np.empty_like(session_to_anchor)
        anchor_to_session[session_to_anchor] = np.arange(len(session_to_anchor))
        aligned.append(predictions[anchor_to_session])
        session_mappings.append(session_to_anchor)
        diagnostics.append(session_diagnostics)

    consensus_at_anchor = np.mean(np.stack(aligned), axis=0, dtype=np.float32)
    consensus = [consensus_at_anchor.copy()]
    consensus.extend(consensus_at_anchor[mapping] for mapping in session_mappings)
    return consensus, diagnostics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=25)
    split = make_split(blocks, "task1", seed=42)
    shuffled = np.random.default_rng(123).permutation(split.validation_pictures)
    tune_pictures = np.sort(shuffled[:16])
    evaluation_pictures = np.sort(shuffled[64:])
    bank = images[np.flatnonzero(~np.isin(np.arange(len(images)), split.validation_pictures))]
    rows = []

    for block_index, block in enumerate(blocks):
        block_split = split.blocks[block.key]
        targets, counts = aggregate_by_target(
            block.features[block_split.validation_trials], block.trial_targets[block_split.validation_trials]
        )
        model = load_regressor(MODEL_ROOT / block.mouse / "regressor.pt", device)
        predictions = dct_decode(predict_logits(model, counts, device=device), 32)
        permutation = np.random.default_rng(POST_PREDICTION_SHUFFLE_SEED + block_index).permutation(len(targets))
        targets = targets[permutation]
        predictions = predictions[permutation]
        rows.append({
            "block": block.key,
            "mouse": block.mouse,
            "targets": targets,
            "continuous": {"independent": predictions},
        })

    def evaluate(mode, lambda_, pictures):
        prediction_parts = []
        target_parts = []
        group_parts = []
        for row in rows:
            mask = np.isin(row["targets"], pictures)
            continuous = row["continuous"][mode][mask]
            retrieved, _ = retrieve_glyphs(
                continuous, bank, metric="directional2", blur_sigma=1.0, device=device
            )
            fused = np.clip((1 - lambda_) * retrieved + lambda_ * continuous, 0, 1)
            prediction_parts.append(fused)
            target_parts.append(images[row["targets"][mask]])
            group_parts.extend([row["block"]] * int(mask.sum()))
        predictions = np.concatenate(prediction_parts)
        references = np.concatenate(target_parts)
        groups = np.asarray(group_parts)
        scores = score_predictions(predictions, references, device=device, batch_size=16)
        return summarize_scores(scores, groups), predictions, references, groups

    baseline, _, _, _ = evaluate("independent", 0.0, evaluation_pictures)
    baseline_error = abs(baseline["ocf_mean"] - BASELINE_OCF)
    print(
        f"BASELINE independent lambda=0: actual={baseline['ocf_mean']:.10f}; "
        f"expected={BASELINE_OCF:.10f}; abs_error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(
            f"Independent lambda=0 baseline mismatch: actual={baseline['ocf_mean']}, "
            f"expected={BASELINE_OCF}, abs_error={baseline_error}, tolerance={BASELINE_ATOL}."
        )

    alignment_diagnostics = []
    for mouse in sorted({block.mouse for block in blocks}):
        mouse_rows = sorted((row for row in rows if row["mouse"] == mouse), key=lambda row: row["block"])
        if len(mouse_rows) != 4:
            raise ValueError(f"{mouse} must have exactly four session blocks; found {len(mouse_rows)}.")
        lengths = {len(row["continuous"]["independent"]) for row in mouse_rows}
        if len(lengths) != 1:
            raise ValueError(f"{mouse} session blocks must have equal lengths; found {sorted(lengths)}.")
        consensus, diagnostics = four_session_consensus(
            [row["continuous"]["independent"] for row in mouse_rows], device
        )
        for row, consensus_predictions in zip(mouse_rows, consensus):
            row["continuous"]["consensus"] = consensus_predictions
        alignment_diagnostics.append({
            "mouse": mouse,
            "anchor_block": mouse_rows[0]["block"],
            "samples_per_block": len(mouse_rows[0]["continuous"]["independent"]),
            "sessions": [
                {"block": row["block"], **session_diagnostics}
                for row, session_diagnostics in zip(mouse_rows[1:], diagnostics)
            ],
            "hungarian_mean_cosine": float(np.mean([
                session_diagnostics["hungarian_mean_cosine"] for session_diagnostics in diagnostics
            ])),
        })
        print(f"consensus {mouse}: anchor={mouse_rows[0]['block']}", flush=True)

    tuning = []
    for mode in MODES:
        for lambda_ in LAMBDAS:
            summary, _, _, _ = evaluate(mode, lambda_, tune_pictures)
            record = {"mode": mode, "lambda": lambda_, "ocf_mean": summary["ocf_mean"]}
            tuning.append(record)
            print(f"TUNE {record}", flush=True)
    best = max(tuning, key=lambda row: row["ocf_mean"])
    evaluation, predictions, references, groups = evaluate(best["mode"], best["lambda"], evaluation_pictures)

    output_dir = ROOT / "test/01_task1_reconstruction/14_task1_consensus"
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifacts / "task1_consensus_predictions.npz",
        predictions=np.rint(np.clip(predictions, 0, 1) * 255).astype(np.uint8),
        targets=references.astype(np.uint8) * 255,
        groups=groups,
    )
    result = {
        "experiment": "14_task1_consensus",
        "name": "task1_four_session_consensus_soft_fusion",
        "config": {
            "task": "task1",
            "bin_ms": 25,
            "dct_cutoff": 32,
            "retrieval_metric": "directional2",
            "retrieval_blur_sigma": 1.0,
            "modes": list(MODES),
            "lambdas": list(LAMBDAS),
            "selection_metric": "tune_ocf_mean",
            "post_prediction_shuffle_seed": POST_PREDICTION_SHUFFLE_SEED,
            "alignment": "three per-mouse N-by-N Hungarian assignments to the first block key; predictions only",
        },
        "protocol": {
            "split_seed": 42,
            "picture_seed": 123,
            "tune_picture_count": len(tune_pictures),
            "tune_pictures": tune_pictures.tolist(),
            "evaluation_picture_count": len(evaluation_pictures),
            "evaluation_pictures": evaluation_pictures.tolist(),
            "bank_picture_count": len(bank),
            "bank_excludes": "all split.validation_pictures",
            "local_repeats_per_character": 4,
            "competition_test_repeats_per_character": 20,
        },
        "baseline_reproduction": {
            "expected_ocf": BASELINE_OCF,
            "actual_ocf": baseline["ocf_mean"],
            "absolute_error": baseline_error,
            "absolute_tolerance": BASELINE_ATOL,
            "passed": True,
        },
        "alignment_diagnostics": alignment_diagnostics,
        "tuning": tuning,
        "selected": best,
        "evaluation": evaluation,
        "evaluation_baseline": baseline,
        "notes": {
            "repeat_protocol": "Local validation aggregates four repeats per character; the competition test uses 20.",
            "soft_fusion": "Retrieval/continuous soft fusion is an exploratory grayscale experiment; lambda=0 is binary retrieval.",
        },
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
    }
    path = output_dir / "results_task1_consensus.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL {path}: selected={evaluation['ocf_mean']:.10f}; "
        f"baseline={baseline['ocf_mean']:.10f}; variant={best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
