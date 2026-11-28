"""Four-fold cross-fitted evaluation of the Task 3 listwise reranker.

For every held fold k the reranker is trained only on the OOF rows of the
other three folds; the base scores and candidate features are reused
unchanged from experiment 83.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CACHE = OUT / "artifacts" / "cache"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.reranker import fit_listwise_reranker, permute_topk_scores, predict_utilities
from src.ridge import dct_encode


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp83 = _load("exp83_84", ROOT / "test/03_task3_single_trial/83_task3_listwise_reranker/run.py")
probe, exp74 = exp83.probe, exp83.exp74
FOLDS, MICE, C = exp83.FOLDS, exp83.MICE, exp83.C
HYPER = exp83.HYPER
REF83 = ROOT / "test/03_task3_single_trial/83_task3_listwise_reranker/results.json"


def fold_rows(blocks, k, mouse, bank, centered, gt, heldout, device, cache_path):
    """Score one (mouse, fold) and cache everything the reranker needs later."""
    logits, encoding, continuous, labels = exp83.fold_scores(blocks, k, bank, centered, device)
    stats = exp83.bare_stats(logits, labels)
    base_cells, arrays, pool = [], {}, []
    for s, block in enumerate(blocks, start=1):
        sl = slice((s - 1) * C, s * C)
        lab = labels[sl]
        mlp_z = exp83.row_zscore(logits[sl]).astype(np.float32)
        enc_z = exp83.row_zscore(encoding[sl]).astype(np.float32)
        cont_z = exp83.row_zscore(continuous[sl]).astype(np.float32)
        base = ((mlp_z + enc_z + cont_z) / 3.0).astype(np.float32)
        top, numeric = exp83.build_candidate_features(base, mlp_z, enc_z, mouse, HYPER["top_n"])
        arrays[f"s{s}_base"] = base
        arrays[f"s{s}_top"] = top
        arrays[f"s{s}_numeric"] = numeric
        arrays[f"s{s}_labels"] = lab.astype(np.int64)
        pool.append({"block": f"{mouse}/fold{k}/session{s}",
                     "targets": lab.astype(np.int64), "top": top, "numeric": numeric})
        read = exp83.readout(base, lab, gt, heldout, device)
        base_cells.append({"session": s, "labels": lab, "readout": read})
        print(mouse, f"fold{k}", f"session{s}", "base_ocf", round(read["ocf"], 6), flush=True)
    np.savez(cache_path, **arrays)
    del logits, encoding, continuous, arrays
    torch.cuda.empty_cache()
    return stats, base_cells, pool


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    device = "cuda"
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(C)[256:768])
    bank = dct_encode(gt, 16)
    centered = (bank - bank.mean(axis=0, keepdims=True)).astype(np.float32)
    data = {mouse: probe.load_mouse(mouse, units) for mouse, units in MICE.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    print("data loaded", {m: len(b) for m, b in data.items()}, flush=True)

    rows_by_fold, base_cells, mlp_stats = {}, {}, {}
    for mouse, blocks in data.items():
        mlp_stats[mouse] = {}
        for k in FOLDS:
            stats, stored, pool = fold_rows(blocks, k, mouse, bank, centered, gt, heldout,
                                            device, CACHE / f"{mouse}_fold{k}.npz")
            mlp_stats[mouse][str(k)] = stats
            base_cells[(mouse, k)] = stored
            rows_by_fold[(mouse, k)] = pool
            print(mouse, f"fold{k}", stats, flush=True)

    reference = json.loads(REF83.read_text(encoding="utf-8")) if REF83.exists() else None
    consistency = {"reference": str(REF83) if reference else None,
                   "mlp_stats_match": None, "fold4_base_match": None,
                   "max_abs_diff": None}
    if reference is not None:
        diffs = []
        for mouse in MICE:
            for k in FOLDS:
                for key in ("top1", "recall32"):
                    diffs.append(abs(mlp_stats[mouse][str(k)][key]
                                     - reference["mlp_only_arm"][mouse]["per_fold"][str(k)][key]))
            for s in range(1, 5):
                diffs.append(abs(base_cells[(mouse, 4)][s - 1]["readout"]["ocf"]
                                 - reference["base"][mouse]["sessions"][f"session{s}"]["ocf"]))
        consistency["max_abs_diff"] = float(max(diffs))
        consistency["mlp_stats_match"] = bool(max(consistency["max_abs_diff"], 0.0) < 1e-9)
        consistency["fold4_base_match"] = consistency["mlp_stats_match"]
    print("consistency", consistency, flush=True)

    result = {
        "experiment": "84_reranker_cross_fitted",
        "preregistration_sha256": digest,
        "data_scope": "public train only; hidden test data never read",
        "hyperparameters": {**HYPER, "encoding_alpha": exp83.ENCODING_ALPHA},
        "consistency_with_83": consistency,
        "mlp_only_per_fold": mlp_stats,
        "folds": {},
        "aggregated": {},
        "per_session_aggregated": {},
        "paired": {},
        "reranker_training": {},
    }
    if reference is not None and not consistency["mlp_stats_match"]:
        result["status"] = "aborted_base_mismatch_with_83"
        result["pass_rule"] = {"advance": False, "reason": "base scores do not match experiment 83"}
        (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        return

    def evaluate(rows, k):
        model, training = fit_listwise_reranker(
            rows, gt, OUT / f"artifacts/listwise_cnn_fold{k}.pt",
            seed=HYPER["seed"], epochs=HYPER["epochs"], batch_size=HYPER["batch_size"],
            learning_rate=HYPER["learning_rate"], weight_decay=HYPER["weight_decay"],
            top_n=HYPER["top_n"], device=device)
        out = {}
        for mouse, blocks in data.items():
            base_list, rerank_list = base_cells[(mouse, k)], []
            with np.load(CACHE / f"{mouse}_fold{k}.npz") as cached:
                for s in range(1, 5):
                    top = cached[f"s{s}_top"]
                    numeric = cached[f"s{s}_numeric"]
                    base = cached[f"s{s}_base"]
                    labels = cached[f"s{s}_labels"]
                    utilities = predict_utilities(model, top, numeric, gt, device)
                    reranked = permute_topk_scores(base, top, utilities, HYPER["k"])
                    rerank_list.append({"session": s, "readout": exp83.readout(
                        reranked, labels, gt, heldout, device)})
                    print(mouse, f"fold{k}", f"session{s}", "rerank_ocf",
                          round(rerank_list[-1]["readout"]["ocf"], 6), flush=True)
            out[mouse] = {"base": base_list, "rerank": rerank_list}
        del model
        torch.cuda.empty_cache()
        return out, training

    for k in FOLDS:
        pool = [row for (mouse, j), rows in rows_by_fold.items() if j != k for row in rows]
        evaluated, training = evaluate(pool, k)
        result["reranker_training"][str(k)] = training
        for mouse in MICE:
            fold = {"base_sessions": {}, "rerank_sessions": {}}
            for base_cell, rr_cell in zip(evaluated[mouse]["base"], evaluated[mouse]["rerank"]):
                key = f"session{rr_cell['session']}"
                fold["base_sessions"][key] = {
                    key2: base_cell["readout"][key2] for key2 in ("ocf", "top1", "recall32", "top1_assigned")}
                fold["rerank_sessions"][key] = {
                    key2: rr_cell["readout"][key2] for key2 in ("ocf", "top1", "recall32", "top1_assigned")}
            fold["base_fold_mean_ocf"] = float(np.mean(
                [v["ocf"] for v in fold["base_sessions"].values()]))
            fold["rerank_fold_mean_ocf"] = float(np.mean(
                [v["ocf"] for v in fold["rerank_sessions"].values()]))
            fold["base_top1"] = float(np.mean([v["top1"] for v in fold["base_sessions"].values()]))
            fold["rerank_top1"] = float(np.mean([v["top1"] for v in fold["rerank_sessions"].values()]))
            result["folds"].setdefault(mouse, {})[str(k)] = fold
            print(mouse, f"fold{k}", "base", round(fold["base_fold_mean_ocf"], 6),
                  "rerank", round(fold["rerank_fold_mean_ocf"], 6),
                  "gain", round(fold["rerank_fold_mean_ocf"] - fold["base_fold_mean_ocf"], 6),
                  flush=True)

    for mouse in MICE:
        sessions = sorted(result["folds"][mouse]["1"]["base_sessions"])
        per_session = {}
        for key in sessions:
            base_vals = [result["folds"][mouse][str(k)]["base_sessions"][key] for k in FOLDS]
            rerank_vals = [result["folds"][mouse][str(k)]["rerank_sessions"][key] for k in FOLDS]
            entry = {
                "base_ocf": float(np.mean([v["ocf"] for v in base_vals])),
                "rerank_ocf": float(np.mean([v["ocf"] for v in rerank_vals])),
                "base_top1": float(np.mean([v["top1"] for v in base_vals])),
                "rerank_top1": float(np.mean([v["top1"] for v in rerank_vals])),
                "base_recall32": float(np.mean([v["recall32"] for v in base_vals])),
                "rerank_recall32": float(np.mean([v["recall32"] for v in rerank_vals])),
            }
            entry["gain"] = entry["rerank_ocf"] - entry["base_ocf"]
            per_session[key] = entry
        result["per_session_aggregated"][mouse] = per_session
        base_fold_means = [result["folds"][mouse][str(k)]["base_fold_mean_ocf"] for k in FOLDS]
        rerank_fold_means = [result["folds"][mouse][str(k)]["rerank_fold_mean_ocf"] for k in FOLDS]
        base_top1_folds = [result["folds"][mouse][str(k)]["base_top1"] for k in FOLDS]
        rerank_top1_folds = [result["folds"][mouse][str(k)]["rerank_top1"] for k in FOLDS]
        result["aggregated"][mouse] = {
            "base_ocf": float(np.mean(base_fold_means)),
            "rerank_ocf": float(np.mean(rerank_fold_means)),
            "base_top1": float(np.mean(base_top1_folds)),
            "rerank_top1": float(np.mean(rerank_top1_folds)),
            "base_recall32": float(np.mean([v["base_recall32"] for v in per_session.values()])),
            "rerank_recall32": float(np.mean([v["rerank_recall32"] for v in per_session.values()])),
            "fold_mean_base_ocf": {str(k): v for k, v in zip(FOLDS, base_fold_means)},
            "fold_mean_rerank_ocf": {str(k): v for k, v in zip(FOLDS, rerank_fold_means)},
        }
        gains = {key: v["gain"] for key, v in per_session.items()}
        result["paired"][mouse] = {
            "gain": result["aggregated"][mouse]["rerank_ocf"] - result["aggregated"][mouse]["base_ocf"],
            "top1_delta": result["aggregated"][mouse]["rerank_top1"]
                          - result["aggregated"][mouse]["base_top1"],
            "sessions": gains,
            "non_negative_sessions": int(sum(v >= 0 for v in gains.values())),
        }

    gains = {m: result["paired"][m]["gain"] for m in MICE}
    session_gains = [v for m in MICE for v in result["paired"][m]["sessions"].values()]
    non_negative = int(sum(v >= 0 for v in session_gains))
    top1_ok = all(result["paired"][m]["top1_delta"] >= 0 for m in MICE)
    recall_ok = all(abs(result["aggregated"][m]["rerank_recall32"]
                        - result["aggregated"][m]["base_recall32"]) < 1e-12 for m in MICE)
    result["pass_rule"] = {
        "aggregated_gain": gains,
        "gain_at_least_0.30_both": bool(all(v >= 0.30 for v in gains.values())),
        "non_negative_mouse_sessions": non_negative,
        "at_least_5_of_8_non_negative": bool(non_negative >= 5),
        "top1_not_lower_both": bool(top1_ok),
        "recall32_unchanged": bool(recall_ok),
        "advance": bool(all(v >= 0.30 for v in gains.values()) and non_negative >= 5 and top1_ok),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"aggregated": result["aggregated"],
                      "paired": result["paired"],
                      "pass_rule": result["pass_rule"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
