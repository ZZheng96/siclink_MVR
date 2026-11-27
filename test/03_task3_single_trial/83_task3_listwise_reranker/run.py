"""Port the Task 2 listwise glyph reranker to Task 3 under a frozen protocol.

The reranker may only permute scores inside the base top-32 shortlist, so
candidate recall is unchanged and any OCF effect is purely ordering.
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
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.encoding import response_scores
from src.evaluation import score_predictions
from src.features import aggregate_by_target
from src.final import row_zscore
from src.reranker import apply_listwise_reranker, build_candidate_features, fit_listwise_reranker
from src.retrieval import softmax_dct_similarity, unique_assignment
from src.ridge import RidgePath, dct_encode


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("probe83", ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py")
exp74 = _load("exp74_83", ROOT / "test/03_task3_single_trial/74_feature_rerank_verifier/run.py")

FOLDS = probe.FOLDS
MICE = probe.MICE
C = probe.C
ENCODING_ALPHA = 0.01  # src/final_config.json task3
HYPER = {"top_n": 32, "k": 8, "epochs": 12, "batch_size": 32,
         "learning_rate": 1e-3, "weight_decay": 1e-3, "seed": 31}
EXPECTED = {
    "mice1": {"top1_mean": 0.015625, "recall32_mean": 0.179267, "fold4_ocf_mean": 16.126378},
    "mice2": {"top1_mean": 0.007710, "recall32_mean": 0.112923, "fold4_ocf_mean": 14.725502},
}
TOL = 1e-5


def fold_inputs(blocks, k):
    """Per-fold z-scored MLP input plus the untouched log1p counts."""
    x = probe.normalize(blocks, k)
    r = np.concatenate([b["r"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    log_counts = np.concatenate([b["x"] for b in blocks])
    tr, te = r != k, r == k
    return x, log_counts, y, tr, te


def encoding_prototypes(blocks, k, bank, log_counts):
    """Production Task 3 recipe: DCT16 ridge on per-session repeat means."""
    raw = np.expm1(log_counts)
    neural_parts, target_parts, offset = [], [], 0
    for block in blocks:
        n = len(block["y"])
        train_rows = np.flatnonzero(block["r"] != k)
        targets, mean_x = aggregate_by_target(raw[offset + train_rows], block["y"][train_rows])
        neural_parts.append(np.log1p(mean_x))
        target_parts.append(targets)
        offset += n
    path = RidgePath().fit(bank[np.concatenate(target_parts)], np.concatenate(neural_parts))
    return path.predict(bank, ENCODING_ALPHA)


def fold_scores(blocks, k, bank, centered, device):
    x, log_counts, y, tr, te = fold_inputs(blocks, k)
    model = exp74.fit_mlp(x[tr], y[tr])
    logits = exp74.logits_of(model, x[te])
    del model
    torch.cuda.empty_cache()
    prototypes = encoding_prototypes(blocks, k, bank, log_counts)
    encoding = response_scores(log_counts[te], prototypes, metric="mse", device=device)
    del prototypes
    continuous = softmax_dct_similarity(logits, centered, device=device)
    return logits, encoding, continuous, y[te]


def bare_stats(logits, labels):
    top1 = float((logits.argmax(1) == labels).mean())
    recall = float((np.argpartition(-logits, 32, axis=1)[:, :32] == labels[:, None]).any(1).mean())
    return {"top1": top1, "recall32": recall}


def readout(scores, labels, gt, heldout, device="cuda"):
    assigned = unique_assignment(scores)
    mask = np.isin(labels, heldout)
    values = score_predictions(gt[assigned[mask]], gt[labels[mask]],
                               device=device, batch_size=16)
    return {"ocf": float(values.mean()),
            "top1": float((scores.argmax(1) == labels).mean()),
            "recall32": float((np.argpartition(-scores, 32, axis=1)[:, :32]
                               == labels[:, None]).any(1).mean()),
            "top1_assigned": float((assigned == labels).mean())}


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    device = "cuda"
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    bank = dct_encode(gt, 16)
    centered = (bank - bank.mean(axis=0, keepdims=True)).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(C)[256:768])
    data = {mouse: probe.load_mouse(mouse, units) for mouse, units in MICE.items()}
    print("data loaded", {m: len(b) for m, b in data.items()}, flush=True)

    result = {
        "experiment": "83_task3_listwise_reranker",
        "preregistration_sha256": digest,
        "data_scope": "public train only; hidden test data never read",
        "hyperparameters": {**HYPER, "encoding_alpha": ENCODING_ALPHA, "top_n_candidates": 32},
        "mlp_only_arm": {},
        "internal_check": {"expected": EXPECTED},
        "base": {},
        "rerank": {},
        "paired": {},
    }

    rows = []
    fold4 = {}
    for mouse, blocks in data.items():
        fold1to4 = {}
        for k in FOLDS:
            logits, encoding, continuous, labels = fold_scores(blocks, k, bank, centered, device)
            fold1to4[k] = bare_stats(logits, labels)
            if k == FOLDS[-1]:
                stored = []
                for s, block in enumerate(blocks, start=1):
                    sl = slice((s - 1) * C, s * C)
                    lab = labels[sl]
                    mlp_z = row_zscore(logits[sl]).astype(np.float32)
                    enc_z = row_zscore(encoding[sl]).astype(np.float32)
                    cont_z = row_zscore(continuous[sl]).astype(np.float32)
                    base = ((mlp_z + enc_z + cont_z) / 3.0).astype(np.float32)
                    mlp_readout = readout(mlp_z, lab, gt, heldout, device)
                    base_readout = readout(base, lab, gt, heldout, device)
                    base_readout["mlp_only_ocf"] = mlp_readout["ocf"]
                    stored.append({"session": s, "labels": lab, "base": base,
                                   "mlp_z": mlp_z, "encoding_z": enc_z,
                                   "readout": base_readout})
                    print(mouse, f"fold{k}", f"session{s}",
                          "mlp_ocf", round(mlp_readout["ocf"], 6),
                          "base_ocf", round(base_readout["ocf"], 6), flush=True)
                fold4[mouse] = stored
            else:
                for s, block in enumerate(blocks, start=1):
                    sl = slice((s - 1) * C, s * C)
                    mlp_z = row_zscore(logits[sl]).astype(np.float32)
                    enc_z = row_zscore(encoding[sl]).astype(np.float32)
                    cont_z = row_zscore(continuous[sl]).astype(np.float32)
                    base = ((mlp_z + enc_z + cont_z) / 3.0).astype(np.float32)
                    top, numeric = build_candidate_features(
                        base, mlp_z, enc_z, mouse, HYPER["top_n"])
                    rows.append({"block": f"{mouse}/fold{k}/session{s}",
                                 "targets": labels[sl].astype(np.int64),
                                 "top": top, "numeric": numeric})
            print(mouse, f"fold{k}", fold1to4[k], flush=True)
            del logits, encoding, continuous
            torch.cuda.empty_cache()
        result["mlp_only_arm"][mouse] = {
            "per_fold": {str(k): fold1to4[k] for k in FOLDS},
            "top1_mean": float(np.mean([fold1to4[k]["top1"] for k in FOLDS])),
            "recall32_mean": float(np.mean([fold1to4[k]["recall32"] for k in FOLDS])),
            "fold4_ocf_mean": float(np.mean([c["readout"]["mlp_only_ocf"]
                                             for c in fold4[mouse]])),
        }

    check = result["internal_check"]
    check["observed"] = {m: {key: result["mlp_only_arm"][m][key]
                             for key in ("top1_mean", "recall32_mean", "fold4_ocf_mean")}
                         for m in MICE}
    check["delta"] = {m: {key: result["mlp_only_arm"][m][key] - EXPECTED[m][key]
                          for key in EXPECTED[m]} for m in MICE}
    check["passed"] = bool(all(abs(check["delta"][m][key]) < TOL
                               for m in MICE for key in EXPECTED[m]))
    print("internal_check", json.dumps(check, indent=2), flush=True)
    if not check["passed"]:
        result["status"] = "void_internal_check_failed"
        result["pass_rule"] = {"advance": False, "reason": "MLP-only arm did not reproduce experiment 82"}
        (OUT / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    model, training = fit_listwise_reranker(
        rows, gt, OUT / "artifacts/listwise_cnn.pt",
        seed=HYPER["seed"], epochs=HYPER["epochs"], batch_size=HYPER["batch_size"],
        learning_rate=HYPER["learning_rate"], weight_decay=HYPER["weight_decay"],
        top_n=HYPER["top_n"], device=device)
    del rows
    torch.cuda.empty_cache()
    result["reranker_training"] = training

    overall_diffs = []
    for mouse in MICE:
        base_sessions, rerank_sessions, diffs = {}, {}, {}
        for cell in fold4[mouse]:
            s = cell["session"]
            key = f"session{s}"
            reranked = apply_listwise_reranker(
                model, cell["base"], cell["mlp_z"], cell["encoding_z"], mouse, gt, device,
                top_n=HYPER["top_n"], k=HYPER["k"])
            reread = readout(reranked, cell["labels"], gt, heldout, device)
            base_sessions[key] = {k2: cell["readout"][k2] for k2 in ("ocf", "top1", "recall32", "top1_assigned")}
            rerank_sessions[key] = {k2: reread[k2] for k2 in ("ocf", "top1", "recall32", "top1_assigned")}
            diffs[key] = reread["ocf"] - cell["readout"]["ocf"]
            overall_diffs.append(diffs[key])
            print(mouse, key, "base_ocf", round(base_sessions[key]["ocf"], 6),
                  "rerank_ocf", round(rerank_sessions[key]["ocf"], 6),
                  "diff", round(diffs[key], 6), flush=True)
        result["base"][mouse] = {
            "sessions": base_sessions,
            "ocf_mean": float(np.mean([v["ocf"] for v in base_sessions.values()])),
            "top1": float(np.mean([v["top1"] for v in base_sessions.values()])),
            "recall32": float(np.mean([v["recall32"] for v in base_sessions.values()])),
        }
        result["rerank"][mouse] = {
            "sessions": rerank_sessions,
            "ocf_mean": float(np.mean([v["ocf"] for v in rerank_sessions.values()])),
            "top1": float(np.mean([v["top1"] for v in rerank_sessions.values()])),
            "recall32": float(np.mean([v["recall32"] for v in rerank_sessions.values()])),
        }
        result["paired"][mouse] = {
            "sessions": diffs,
            "mean": float(np.mean(list(diffs.values()))),
            "ocf_gain_vs_base": result["rerank"][mouse]["ocf_mean"] - result["base"][mouse]["ocf_mean"],
            "positive_sessions": int(sum(v >= 0 for v in diffs.values())),
            "top1_delta": result["rerank"][mouse]["top1"] - result["base"][mouse]["top1"],
        }

    gains = {m: result["paired"][m]["ocf_gain_vs_base"] for m in MICE}
    positive_groups = int(sum(1 for v in overall_diffs if v >= 0))
    top1_ok = all(result["rerank"][m]["top1"] >= result["base"][m]["top1"] for m in MICE)
    recall_ok = all(abs(result["rerank"][m]["recall32"] - result["base"][m]["recall32"]) < 1e-12
                    for m in MICE)
    result["paired"]["overall"] = {"mean": float(np.mean(overall_diffs)),
                                   "positive_groups": positive_groups}
    result["pass_rule"] = {
        "ocf_gain": gains,
        "ocf_gain_at_least_0.30_both": bool(all(v >= 0.30 for v in gains.values())),
        "non_negative_groups": positive_groups,
        "at_least_5_of_8_non_negative": bool(positive_groups >= 5),
        "top1_not_lower_both": bool(top1_ok),
        "recall32_unchanged": bool(recall_ok),
        "advance": bool(all(v >= 0.30 for v in gains.values()) and positive_groups >= 5 and top1_ok),
    }
    (OUT / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"paired": result["paired"], "pass_rule": result["pass_rule"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
