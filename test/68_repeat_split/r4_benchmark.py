"""R4 repeat-holdout benchmark for previously selected Task3-style recipes.

All candidates train on repeat1-3 only and are scored once on repeat4. Official OCF
is not computed; the primary metrics are Top-1, Recall@32, macro-F1 and NLL.
"""
import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dataloader import load_images
from src.encoding import response_scores
from src.features import FeatureBlock, trial_count_features
from src.final import (
    apply_session_normalizer,
    checkpoint_path,
    row_zscore,
    train_task,
)
from src.neural import load_classifier, predict_logits
from src.retrieval import softmax_dct_similarity, unique_assignment
from src.ridge import dct_encode

SPLIT = ROOT / "data/derived/repeat_split_v1"
DATA = ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset"
OUT = ROOT / "data/derived/repeat_bench_v1"
K = 2748
MICE = ("mice1", "mice2")
SESSIONS = (1, 2, 3, 4)
UNITS = {"mice1": 134, "mice2": 78}
SEEDS = (42, 202, 777)
IMAGES = (load_images(DATA) >= 128).astype(np.float32)
CENTERED = None  # set in main


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_features(mouse, session, repeat, bin_ms):
    path = SPLIT / mouse / f"session{session}_repeat{repeat}.npz"
    with np.load(path, allow_pickle=False) as z:
        y = z["target"].astype(np.int64)
        p = z["trial_ptr"].astype(np.int64)
        t = z["spike_samples"]
        u = z["unit_index"]
        if not (len(y) == K and p.shape == (K + 1,) and len(t) == len(u) and p[0] == 0 and p[-1] == len(t)):
            raise ValueError(f"invalid repeat block: {path}")
        if not np.array_equal(np.sort(y), np.arange(K)) or not np.all(z["repeat_index"] == repeat):
            raise ValueError(f"invalid repeat metadata: {path}")
        dummy = SimpleNamespace(
            sample_rate_hz=30000, window_ms=250, spike_samples=t, unit_index=u,
            trial_ptr=p, n_units=UNITS[mouse],
        )
        counts = trial_count_features(dummy, bin_ms=bin_ms)
    return counts, y


def make_block(mouse, session, bin_ms, split):
    repeats = (1, 2, 3) if split == "train" else (4,)
    parts = [count_features(mouse, session, r, bin_ms) for r in repeats]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    n = len(y)
    return FeatureBlock(
        mouse=mouse, session=f"session{session}", split=split, task="task3",
        sample_ids=np.arange(n).astype(np.str_), group_ptr=np.arange(n + 1),
        targets=y, features=x, n_units=UNITS[mouse], bin_ms=bin_ms,
    )


def config_for(bin_ms, epochs, dropout, name):
    config = json.loads((ROOT / "src/final_config.json").read_text(encoding="utf-8"))
    config["name"] = name
    config["task3"].update({
        "bin_ms": bin_ms,
        "epochs": {"mice1": epochs, "mice2": epochs},
        "dropout": dropout,
        "seeds": list(SEEDS),
        "session_norm": True,
        "assignment": True,
        "encoding_alpha": 0.01,
        "prototype_continuous": {"enabled": True, "temperature": 1.0,
                                 "fusion": "equal_row_zscore_mlp_encoding_continuous"},
    })
    return config


def train(name, bin_ms, epochs, dropouts):
    """dropouts=(mice1,mice2); train one mouse at a time so dropout can differ."""
    out = OUT / name
    for mouse, dropout in zip(MICE, dropouts):
        blocks = [make_block(mouse, s, bin_ms, "train") for s in SESSIONS]
        config = config_for(bin_ms, epochs, dropout, name)
        train_task(config, "task3", out, images=IMAGES, blocks=blocks, split=None,
                   skip_existing=True, device="cuda")
        del blocks
    manifest = {
        "name": name, "bin_ms": bin_ms, "epochs": epochs, "dropouts": dict(zip(MICE, dropouts)),
        "seeds": list(SEEDS), "protocol": "repeat1-3 train, repeat4 evaluate",
        "utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            f"{m}/{seed}": sha(checkpoint_path(out, "task3", m, seed))
            for m in MICE for seed in SEEDS
        },
    }
    (out / "bench_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"trained {name}", flush=True)


def session_scores(name, mouse, session, bin_ms):
    out = OUT / name
    raw, y = count_features(mouse, session, 4, bin_ms)
    with np.load(out / "task3" / mouse / f"session{session}_normalizer.npz", allow_pickle=False) as z:
        normalizer = (z["mean"], z["scale"])
    counts = apply_session_normalizer(raw, normalizer)
    logits = []
    for seed in SEEDS:
        model = load_classifier(checkpoint_path(out, "task3", mouse, seed), device="cuda")
        logits.append(predict_logits(model, counts, device="cuda"))
        del model
    mean_logits = np.mean(logits, axis=0, dtype=np.float32)
    prototypes = np.load(out / "task3" / mouse / "encoding_prototypes.npy")
    encoding = response_scores(np.log1p(raw), prototypes, metric="mse", device="cuda")
    continuous = softmax_dct_similarity(mean_logits, CENTERED, device="cuda")
    v5 = (row_zscore(mean_logits) + row_zscore(encoding) + row_zscore(continuous)) / 3.0
    continuous1 = softmax_dct_similarity(logits[0], CENTERED, device="cuda")
    v5_s1 = (row_zscore(logits[0]) + row_zscore(encoding) + row_zscore(continuous1)) / 3.0
    return {
        "y": y, "logits_s1": logits[0].astype(np.float32), "logits_s3": mean_logits,
        "encoding": encoding.astype(np.float32), "v5_s1": v5_s1.astype(np.float32),
        "v5_s3": v5.astype(np.float32),
    }


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def score_rows(scores, logits, y, assignment):
    idx = unique_assignment(scores) if assignment else scores.argmax(axis=1)
    n = len(y)
    top32 = np.argpartition(-scores, 32, axis=1)[:, :32]
    hit32 = (top32 == y[:, None]).any(axis=1)
    logp = log_softmax(logits)
    return {
        "y": y,
        "idx": idx,
        "hit32": hit32,
        "top1": float((idx == y).mean()),
        "recall32": float(hit32.mean()),
        "nll": float(-logp[np.arange(n), y].mean()),
    }


def macro_f1(y, idx):
    den = np.bincount(y, minlength=K) + np.bincount(idx, minlength=K)
    tp = np.bincount(y[y == idx], minlength=K)
    return float(np.mean(2 * tp / np.maximum(den, 1)))


def per_mouse(result, mouse):
    ys = np.concatenate([result[s]["y"] for s in SESSIONS])
    idx = np.concatenate([result[s]["idx"] for s in SESSIONS])
    return {
        "top1": float((idx == ys).mean()),
        "recall32": float(np.mean([result[s]["recall32"] for s in SESSIONS])),
        "nll": float(np.mean([result[s]["nll"] for s in SESSIONS])),
        "macro_f1": macro_f1(ys, idx),
    }


def char_sums(result, mouse):
    ys = np.concatenate([result[s]["y"] for s in SESSIONS])
    idx = np.concatenate([result[s]["idx"] for s in SESSIONS])
    hit = np.concatenate([result[s]["hit32"] for s in SESSIONS])
    nll = np.concatenate([np.full(len(result[s]["y"]), result[s]["nll"]) for s in SESSIONS])
    count = np.bincount(ys, minlength=K).astype(np.float64)
    return {
        "count": count,
        "top1": np.bincount(ys, weights=(idx == ys), minlength=K),
        "recall32": np.bincount(ys, weights=hit, minlength=K),
        "nll": np.bincount(ys, weights=nll, minlength=K),
    }


def bootstrap_ci(results):
    """Character-level paired bootstrap over both mice."""
    sums = {m: char_sums(results[m], m) for m in MICE}
    rng = np.random.default_rng(31415)
    draws = {k: [] for k in ("top1", "recall32", "nll")}
    for _ in range(2000):
        w = np.bincount(rng.integers(0, K, K), minlength=K)
        mouse_values = {k: [] for k in draws}
        for m in MICE:
            den = float(np.dot(w, sums[m]["count"]))
            for k in draws:
                mouse_values[k].append(float(np.dot(w, sums[m][k]) / den))
        for k in draws:
            draws[k].append(float(np.mean(mouse_values[k])))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in draws.items()}


def evaluate(name, data, assignment):
    results = {
        m: {s: score_rows(data[m][s]["scores"], data[m][s]["logits"], data[m][s]["y"], assignment)
            for s in SESSIONS}
        for m in MICE
    }
    mice = {m: per_mouse(results[m], m) for m in MICE}
    macro = {k: float(np.mean([mice[m][k] for m in MICE])) for k in ("top1", "recall32", "nll", "macro_f1")}
    per_session = {}
    for m in MICE:
        for s in SESSIONS:
            per_session[f"{m}:session{s}"] = {k: results[m][s][k] for k in ("top1", "recall32", "nll")}
    return {
        "assignment": assignment,
        "mouse": mice,
        "macro": macro,
        "macro_ci95_character_bootstrap": bootstrap_ci(results),
        "per_session": per_session,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    global CENTERED
    CENTERED = dct_encode(IMAGES, 16)
    CENTERED = CENTERED - CENTERED.mean(axis=0, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)

    runs = {
        "v5_50e7": dict(bin_ms=50, epochs=7, dropouts=(0.2, 0.2)),
        "e53_10e7": dict(bin_ms=10, epochs=7, dropouts=(0.2, 0.35)),
        "e53_25e12": dict(bin_ms=25, epochs=12, dropouts=(0.2, 0.35)),
        "e53_50e9": dict(bin_ms=50, epochs=9, dropouts=(0.2, 0.35)),
    }
    if args.stage in ("train", "all"):
        for name, kw in runs.items():
            if (OUT / name / "bench_manifest.json").exists():
                print(f"skip {name}", flush=True)
                continue
            train(name, **kw)

    if args.stage in ("eval", "all"):
        loaded = {name: {m: {s: session_scores(name, m, s, kw["bin_ms"]) for s in SESSIONS}
                         for m in MICE} for name, kw in runs.items()}

        candidates = {}

        def add(name, scores_by_mouse_session, nll_logits_by_mouse_session, note):
            for assignment in (False, True):
                data = {m: {s: {"scores": scores_by_mouse_session[m][s],
                                "logits": nll_logits_by_mouse_session[m][s],
                                "y": loaded["v5_50e7"][m][s]["y"]} for s in SESSIONS} for m in MICE}
                candidates[f"{name}{'_assign' if assignment else '_argmax'}"] = {
                    **evaluate(name, data, assignment), "note": note}

        def view_logits(run, seed):
            return {m: {s: loaded[run][m][s][seed] for s in SESSIONS} for m in MICE}

        def mean_logits(view_maps):
            return {m: {s: np.mean([v[m][s] for v in view_maps], axis=0).astype(np.float32)
                        for s in SESSIONS} for m in MICE}

        def fuse(view_maps):
            return {m: {s: np.mean([row_zscore(v[m][s]) for v in view_maps], axis=0).astype(np.float32)
                        for s in SESSIONS} for m in MICE}

        # Experiment 48 Task3: 25 ms, epoch 12, single seed, MLP only.
        add("48_task3_25e12_s1", view_logits("e53_25e12", "logits_s1"),
            view_logits("e53_25e12", "logits_s1"), "48 production recipe, single seed, MLP only")

        # Experiment 53 Task3: 10/25/50 ms, selected epochs, single seed, row-z fusion.
        views1 = [view_logits("e53_10e7", "logits_s1"), view_logits("e53_25e12", "logits_s1"),
                  view_logits("e53_50e9", "logits_s1")]
        add("53_task3_3view_s1", fuse(views1), mean_logits(views1), "53 three-view MLP, single seed")
        views3 = [view_logits("e53_10e7", "logits_s3"), view_logits("e53_25e12", "logits_s3"),
                  view_logits("e53_50e9", "logits_s3")]
        add("53_task3_3view_s3", fuse(views3), mean_logits(views3), "53 three-view MLP, 3 seeds")

        # v5 Task3: 50 ms epoch 7, MLP+encoding+DCT continuous, assignment.
        add("v5_task3_s1", view_logits("v5_50e7", "v5_s1"), view_logits("v5_50e7", "logits_s1"),
            "v5 Task3 single seed")
        add("v5_task3_s3", view_logits("v5_50e7", "v5_s3"), view_logits("v5_50e7", "logits_s3"),
            "v5 Task3 three seeds")
        add("v5_task3_mlp_s3", view_logits("v5_50e7", "logits_s3"), view_logits("v5_50e7", "logits_s3"),
            "v5 architecture, MLP only, three seeds")

        # Hybrid: v5 branch at all three temporal views, three seeds.
        v5_view_maps = [{m: {s: loaded[run][m][s]["v5_s3"] for s in SESSIONS} for m in MICE}
                        for run in ("e53_10e7", "e53_25e12", "e53_50e9")]
        add("v5_task3_3view_s3", fuse(v5_view_maps), mean_logits(views3),
            "v5 branch at 10/25/50 ms, three seeds, row-z fusion")

        payload = {
            "protocol": "repeat-holdout v1",
            "deviation": "official OCF not computed; R4 used as labelled development feedback",
            "utc": datetime.now(timezone.utc).isoformat(),
            "train_files": "repeat1-3 only",
            "test_files": "repeat4 only",
            "runs": runs,
            "data_manifest": {
                "fit": {f"{m}/session{s}_repeat{r}": sha(SPLIT / m / f"session{s}_repeat{r}.npz")
                        for m in MICE for s in SESSIONS for r in (1, 2, 3)},
                "test": {f"{m}/session{s}_repeat4": sha(SPLIT / m / f"session{s}_repeat4.npz")
                         for m in MICE for s in SESSIONS},
                "pictures_sha256": sha(DATA / "train/pictures.npz"),
            },
            "candidates": candidates,
        }
        (OUT / "R4_BENCHMARK.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = [
            "# R4 第 4 次 repeat 方案对比（非官方 OCF）", "",
            "所有方案只用 repeat1-3 训练；repeat4 每鼠 4 个 session、每字一次。"
            "Top-1/Recall@32 来自最终融合分数，macro-F1 按每鼠 4 个 session 合并计算，NLL 统一用 MLP logits。",
            "", "| 方案 | 分配 | Top-1 | 95% CI | Recall@32 | macro-F1 | NLL | 鼠1 Top-1 | 鼠2 Top-1 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        order = [
            "48_task3_25e12_s1_argmax", "48_task3_25e12_s1_assign",
            "53_task3_3view_s1_argmax", "53_task3_3view_s1_assign",
            "53_task3_3view_s3_assign",
            "v5_task3_s1_argmax", "v5_task3_s1_assign",
            "v5_task3_s3_argmax", "v5_task3_s3_assign",
            "v5_task3_mlp_s3_assign",
            "v5_task3_3view_s3_assign",
        ]
        labels = {
            "48_task3_25e12_s1": "48 Task3 25ms e12 s1",
            "53_task3_3view_s1": "53 Task3 3view s1",
            "53_task3_3view_s3": "53 Task3 3view s3",
            "v5_task3_s1": "v5 Task3 50ms s1",
            "v5_task3_s3": "v5 Task3 50ms s3",
            "v5_task3_mlp_s3": "v5 架构 MLP-only s3",
            "v5_task3_3view_s3": "v5 分支 3view s3",
        }
        for key in order:
            if key not in candidates:
                continue
            r = candidates[key]
            label = key.rsplit("_", 1)[0]
            ci = r["macro_ci95_character_bootstrap"]["top1"]
            lines.append(
                f"| {labels.get(label, label)} | {'匈牙利' if r['assignment'] else 'argmax'} | "
                f"{r['macro']['top1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{r['macro']['recall32']:.4f} | {r['macro']['macro_f1']:.4f} | "
                f"{r['macro']['nll']:.4f} | {r['mouse']['mice1']['top1']:.4f} | "
                f"{r['mouse']['mice2']['top1']:.4f} |"
            )
        (OUT / "R4_BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
