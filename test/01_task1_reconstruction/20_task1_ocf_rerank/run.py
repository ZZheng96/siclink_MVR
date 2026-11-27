"""Task1 使用官方 OCF 对 directional2 的前 16 个候选进行无标签重排。"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.ocf.score import score as official_score
from src.retrieval import _retrieval_embedding, render_gb2312_bank

OUTPUT_DIR = ROOT / "test/01_task1_reconstruction/20_task1_ocf_rerank"
ARTIFACTS = OUTPUT_DIR / "artifacts"
RESULTS_PATH = OUTPUT_DIR / "results_task1_ocf_rerank.json"
SELECTED_NPZ = ARTIFACTS / "task1_ocf_rerank_selected.npz"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
BASELINE_OCF = 18.624308986308705
BASELINE_ATOL = 1e-5
TOP_K = 16


def load_experiment18():
    path = ROOT / "test/06_fusion_transduction/18_seed_ensemble/run.py"
    spec = importlib.util.spec_from_file_location("seed_ensemble_18", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.inference_mode()
def candidate_indices(continuous, bank_embedding, device, batch_size=32):
    """仅根据连续预测与字体字库生成候选，不读取真实标签。"""
    output = []
    for start in range(0, len(continuous), batch_size):
        batch = torch.from_numpy(continuous[start:start + batch_size, None]).to(device)
        embedding = F.normalize(_retrieval_embedding(batch, "directional2"), dim=1)
        scores = embedding @ bank_embedding.T
        output.append(torch.topk(scores, k=TOP_K, dim=1, largest=True, sorted=True).indices.cpu().numpy())
    return np.concatenate(output).astype(np.int64)


def prepare_data(experiment18, images, device):
    data = experiment18.prepare_task1(images)
    data["device"] = device
    data["bank"] = render_gb2312_bank(FONT_PATH, font_size=52, variation="Regular")
    experiment18.predict_task1_seed(data, 42, device)
    bank_tensor = torch.from_numpy(data["bank"][:, None]).to(device)
    bank_embedding = F.normalize(_retrieval_embedding(bank_tensor, "directional2"), dim=1)
    rows = {
        "continuous": [],
        "targets": [],
        "groups": [],
        "pictures": [],
        "candidates": [],
    }
    for row in data["rows"]:
        continuous = row["continuous"][42].astype(np.float32)
        rows["continuous"].append(np.clip(continuous, 0, 1))
        rows["targets"].append(images[row["targets"]])
        rows["groups"].extend([row["block"]] * len(continuous))
        rows["pictures"].append(row["targets"])
        rows["candidates"].append(candidate_indices(continuous, bank_embedding, device))
        print(f"CANDIDATES {row['block']}: n={len(continuous)}, top_k={TOP_K}", flush=True)
    del bank_tensor, bank_embedding
    if device == "cuda":
        torch.cuda.empty_cache()
    return data, {
        "continuous": np.concatenate(rows["continuous"]),
        "targets": np.concatenate(rows["targets"]).astype(np.float32),
        "groups": np.asarray(rows["groups"]),
        "pictures": np.concatenate(rows["pictures"]),
        "candidates": np.concatenate(rows["candidates"]),
    }


def directional_predictions(rows, mask):
    return rows["candidates"][mask, 0]


def evaluate_predictions(predictions, indices, rows, mask, device):
    bank = rows["bank"]
    scores = score_predictions(bank[indices], rows["targets"][mask], device=device, batch_size=16)
    summary = summarize_scores(scores, rows["groups"][mask])
    details = {
        "predictions": bank[indices],
        "targets": rows["targets"][mask],
        "groups": rows["groups"][mask],
        "pictures": rows["pictures"][mask],
        "candidate_indices": np.asarray(indices),
    }
    return summary, details


def ocf_rerank(rows, mask, device):
    """用连续预测和候选字形之间的 OCF 兼容性选择候选，全程不使用 GT。"""
    continuous = rows["continuous"][mask]
    candidates = rows["candidates"][mask]
    n = len(continuous)
    repeated = np.repeat(continuous, TOP_K, axis=0)
    candidate_images = rows["bank"][candidates.reshape(-1)]
    compatibility = official_score(repeated, candidate_images, device=device, batch_size=16)
    compatibility = compatibility.reshape(n, TOP_K)
    best_positions = compatibility.argmax(axis=1)
    selected = candidates[np.arange(n), best_positions]
    diagnostics = {
        "changed_from_directional2_top1_rate": float(np.mean(best_positions != 0)),
        "mean_compatibility_top1": float(compatibility[:, 0].mean()),
        "mean_compatibility_selected": float(compatibility[np.arange(n), best_positions].mean()),
        "mean_selected_candidate_rank": float(np.mean(best_positions + 1)),
    }
    return selected, diagnostics


def main():
    started = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This OCF candidate-reranking experiment requires CUDA.")
    experiment18 = load_experiment18()
    images = (load_images(experiment18.DATA) >= 128).astype(np.float32)
    data, rows = prepare_data(experiment18, images, device)
    rows["bank"] = data["bank"]
    tune_mask = np.isin(rows["pictures"], data["tune_pictures"])
    evaluation_mask = np.isin(rows["pictures"], data["evaluation_pictures"])

    baseline_indices = directional_predictions(rows, evaluation_mask)
    baseline_evaluation, baseline_details = evaluate_predictions(
        baseline_indices, baseline_indices, rows, evaluation_mask, device
    )
    baseline_error = abs(baseline_evaluation["ocf_mean"] - BASELINE_OCF)
    print(
        f"BASELINE actual={baseline_evaluation['ocf_mean']:.10f} expected={BASELINE_OCF:.10f} "
        f"error={baseline_error:.3g}",
        flush=True,
    )
    if baseline_error > BASELINE_ATOL:
        raise RuntimeError(f"Baseline mismatch: absolute error {baseline_error}")

    tune_baseline_indices = directional_predictions(rows, tune_mask)
    tune_baseline, _ = evaluate_predictions(
        tune_baseline_indices, tune_baseline_indices, rows, tune_mask, device
    )
    tuning = [{"variant": "directional2_top1", "top_k": 1, **tune_baseline}]
    rerank_tune_indices, tune_diagnostics = ocf_rerank(rows, tune_mask, device)
    tune_rerank, _ = evaluate_predictions(rerank_tune_indices, rerank_tune_indices, rows, tune_mask, device)
    tuning.append({
        "variant": "official_ocf_rerank",
        "top_k": TOP_K,
        **tune_rerank,
        "candidate_diagnostics": tune_diagnostics,
    })
    print(f"TUNING {json.dumps(tuning, ensure_ascii=False)}", flush=True)

    selected = max(tuning, key=lambda row: row["ocf_mean"])
    if selected["variant"] == "directional2_top1":
        evaluation = baseline_evaluation
        details = baseline_details
    else:
        rerank_evaluation_indices, evaluation_diagnostics = ocf_rerank(rows, evaluation_mask, device)
        evaluation, details = evaluate_predictions(
            rerank_evaluation_indices, rerank_evaluation_indices, rows, evaluation_mask, device
        )
        selected["held_out_candidate_diagnostics"] = evaluation_diagnostics

    np.savez_compressed(
        SELECTED_NPZ,
        predictions=details["predictions"].astype(np.uint8) * 255,
        targets=details["targets"].astype(np.uint8) * 255,
        group_ids=details["groups"],
        picture_indices=details["pictures"],
        candidate_indices=details["candidate_indices"],
        variant=np.asarray(selected["variant"]),
        top_k=np.asarray(selected["top_k"]),
    )
    result = {
        "experiment": "20_task1_ocf_rerank",
        "name": "task1_official_ocf_top16_rerank",
        "score_scope": "fixed local validation protocol only; not a Kaggle score",
        "config": {
            "task": "task1",
            "seed": 42,
            "continuous_predictions": "same seed42 DCT32 MLP as test15/test18",
            "candidate_generator": "directional2 top16 over fixed NotoSansSC Regular size52 strict-GB2312 bank",
            "candidate_reranker": "official OCF between clipped continuous prediction and each binary candidate; no GT used",
            "selection_metric": "tune official OCF mean only",
            "candidate_count": int(data["bank"].shape[0]),
        },
        "protocol": {
            "split_seed": 42,
            "picture_seed": 123,
            "tune_pictures": data["tune_pictures"].tolist(),
            "tune_picture_count": len(data["tune_pictures"]),
            "evaluation_pictures": data["evaluation_pictures"].tolist(),
            "evaluation_picture_count": len(data["evaluation_pictures"]),
            "held_out_rule": "Reproduce directional2 baseline, then evaluate only the tune-selected variant; no held-out variant selection.",
        },
        "baseline": {
            "expected_ocf": BASELINE_OCF,
            "actual_ocf": baseline_evaluation["ocf_mean"],
            "absolute_error": baseline_error,
            "passed": baseline_error <= BASELINE_ATOL,
            "evaluation": baseline_evaluation,
        },
        "tuning": tuning,
        "selected": selected,
        "evaluation": evaluation,
        "evaluation_delta_ocf": evaluation["ocf_mean"] - baseline_evaluation["ocf_mean"],
        "selected_npz": str(SELECTED_NPZ.relative_to(ROOT)),
        "leaderboard_reference": {"rank_1": 54.01486, "comparable": False},
        "Kaggle_baseline": 21.09590,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RESULTS_PATH, result)
    print(
        f"FINAL selected={selected['variant']} OCF={evaluation['ocf_mean']:.10f} "
        f"delta={result['evaluation_delta_ocf']:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
