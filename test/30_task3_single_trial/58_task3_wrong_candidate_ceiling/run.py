"""Is 40+ reachable by structure-aware candidate choice? Measure the best wrong-candidate OCF."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\WORK\siclink")
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.retrieval import glyph_similarity

CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
OUT = ROOT / "test/30_task3_single_trial/58_task3_wrong_candidate_ceiling"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GT = 2748
SAMPLE = 211
TOPK = (32, 100, 500)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
            "max": float(values.max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = np.load(CACHE / "pictures.npy", mmap_mode="r", allow_pickle=False)
    gt = (np.asarray(pictures) >= 128).astype(np.float32)

    similarity = np.asarray(glyph_similarity(gt, metric="directional_multi", device=DEVICE), dtype=np.float32)
    print("similarity", similarity.shape, flush=True)
    np.fill_diagonal(similarity, -np.inf)

    rows = np.sort(np.random.default_rng(20261126).permutation(N_GT)[:SAMPLE])
    order = np.argsort(-similarity[rows], axis=1, kind="stable")

    # Global mean glyph as a generation-style baseline.
    mean_glyph = (gt.mean(axis=0) >= 0.5).astype(np.float32)
    mean_scores = score_predictions(np.repeat(mean_glyph[None], len(rows), axis=0), gt[rows],
                                    device=DEVICE, batch_size=16)
    print("global mean glyph OCF", float(mean_scores.mean()), flush=True)

    output = {
        "scope": "exploratory; GT gallery only; structure-aware wrong-candidate ceiling",
        "rows": int(len(rows)),
        "global_mean_glyph_ocf": summarize(mean_scores),
        "topk": {},
    }
    for k in TOPK:
        candidates = order[:, :k]
        gt_index = np.repeat(rows, k)
        flat = score_predictions(gt[candidates.reshape(-1)], gt[gt_index], device=DEVICE, batch_size=64)
        matrix = flat.reshape(len(rows), k)
        best = matrix.max(axis=1)
        top1 = matrix[:, 0]
        output["topk"][str(k)] = {
            "best_wrong_ocf": summarize(best),
            "top1_wrong_ocf": summarize(top1),
            "fraction_best_at_rank0": float((matrix.argmax(axis=1) == 0).mean()),
        }
        print(k, output["topk"][str(k)]["best_wrong_ocf"]["mean"],
              output["topk"][str(k)]["top1_wrong_ocf"]["mean"], flush=True)

    (OUT / "exploratory_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
