"""Time-resolved (temporal-code) linear decoder vs rate-code baseline on repeat data.

Protocol: train on repeat1-3 of the training sessions, test on repeat4 of a held-out
session. No repeat4 trial is used for fitting. The decoder is a single linear layer
(logistic regression), matching the paper's linear readout idea.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SPLIT = ROOT / "data/derived/repeat_split_v1"
OUT = ROOT / "data/derived/repeat_temporal_v1"
K = 2748
MICE = ("mice1", "mice2")
SESSIONS = (1, 2, 3, 4)
UNITS = {"mice1": 134, "mice2": 78}
BINS = {"rate250": 250, "rate50": 50, "temporal25": 25, "temporal10": 10,
        "temporal10_rate250": 10}
CACHE = {}


def load_counts(mouse, session, repeat, bin_ms):
    key = (mouse, session, repeat, bin_ms)
    if key in CACHE:
        return CACHE[key]
    path = SPLIT / mouse / f"session{session}_repeat{repeat}.npz"
    with np.load(path, allow_pickle=False) as z:
        y = z["target"].astype(np.int64)
        p = z["trial_ptr"].astype(np.int64)
        t = z["spike_samples"]
        u = z["unit_index"]
        if not (len(y) == K and p.shape == (K + 1,) and p[0] == 0 and p[-1] == len(t) == len(u)):
            raise ValueError(f"invalid block {path}")
        if not np.array_equal(np.sort(y), np.arange(K)) or not np.all(z["repeat_index"] == repeat):
            raise ValueError(f"invalid metadata {path}")
        units = UNITS[mouse]
        n_bins = 250 // bin_ms
        samples_per_bin = 30000 * bin_ms // 1000
        row = np.repeat(np.arange(K, dtype=np.int64), np.diff(p))
        flat = (row * n_bins + t // samples_per_bin) * units + u
        counts = np.bincount(flat, minlength=K * n_bins * units).reshape(K, n_bins, units).astype(np.float32)
    CACHE[key] = (counts, y)
    return counts, y


def load_split(mouse, sessions, repeat, bin_ms):
    xs, ys = [], []
    for s in sessions:
        x, y = load_counts(mouse, s, repeat, bin_ms)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def features(counts, name):
    if name == "rate250":
        x = counts.sum(axis=1)
    elif name == "rate50":
        x = counts.reshape(len(counts), -1)
    elif name == "temporal25":
        x = counts.reshape(len(counts), -1)
    elif name == "temporal10":
        x = counts.reshape(len(counts), -1)
    elif name == "temporal10_rate250":
        x = np.concatenate([counts.reshape(len(counts), -1), counts.sum(axis=1)], axis=1)
    else:
        raise ValueError(name)
    return np.log1p(np.maximum(x, 0)).astype(np.float32)


def standardize(train, test):
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1
    return (train - mean) / scale, (test - mean) / scale


class LinearDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, K)

    def forward(self, x):
        return self.fc(x)


def train_linear(x, y, epochs, device, seed=42, batch=512, lr=1e-3, wd=0.1):
    torch.manual_seed(seed)
    model = LinearDecoder(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def predict(model, x, device, batch=1024):
    model.eval()
    out = []
    xt = torch.from_numpy(x).to(device)
    for i in range(0, len(x), batch):
        out.append(model(xt[i:i + batch]).float().cpu().numpy())
    return np.concatenate(out)


def metrics(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    pred = logits.argmax(axis=1)
    top32 = np.argpartition(-logits, 32, axis=1)[:, :32]
    den = np.bincount(y, minlength=K) + np.bincount(pred, minlength=K)
    tp = np.bincount(y[y == pred], minlength=K)
    return {
        "n": int(len(y)),
        "top1": float((pred == y).mean()),
        "recall32": float((top32 == y[:, None]).any(1).mean()),
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "macro_f1": float(np.mean(2 * tp / np.maximum(den, 1))),
    }


def run(mouse, train_sessions, test_session, name, bin_ms, epochs, device):
    xs, ys = [], []
    for s in train_sessions:
        c, y = load_counts(mouse, s, 1, bin_ms)
        xs.append(c); ys.append(y)
        for r in (2, 3):
            c, y = load_counts(mouse, s, r, bin_ms)
            xs.append(c); ys.append(y)
    train_counts = np.concatenate(xs)
    y_train = np.concatenate(ys)
    test_counts, y_test = load_counts(mouse, test_session, 4, bin_ms)
    x_train = features(train_counts, name)
    x_test = features(test_counts, name)
    x_train, x_test = standardize(x_train, x_test)
    start = time.monotonic()
    model = train_linear(x_train, y_train, epochs, device)
    logits = predict(model, x_test, device)
    result = metrics(logits, y_test)
    result["seconds"] = round(time.monotonic() - start, 1)
    result["train_rows"] = int(len(y_train))
    result["test_rows"] = int(len(y_test))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_r4(mouse, name, bin_ms, epochs, device):
    xs, ys, tests = [], [], []
    for s in SESSIONS:
        tr = [load_counts(mouse, s, r, bin_ms)[0] for r in (1, 2, 3)]
        ytr = [load_counts(mouse, s, r, bin_ms)[1] for r in (1, 2, 3)]
        x_tr = features(np.concatenate(tr), name)
        y_tr = np.concatenate(ytr)
        x_te, y_te = load_counts(mouse, s, 4, bin_ms)
        x_te = features(x_te, name)
        x_tr, x_te = standardize(x_tr, x_te)
        xs.append(x_tr); ys.append(y_tr); tests.append((s, x_te, y_te))
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    start = time.monotonic()
    model = train_linear(x_train, y_train, epochs, device)
    per_session = {}
    for s, x_te, y_te in tests:
        per_session[f"session{s}"] = metrics(predict(model, x_te, device), y_te)
    macro = {k: float(np.mean([v[k] for v in per_session.values()]))
             for k in ("top1", "recall32", "nll", "macro_f1")}
    result = {"macro": macro, "per_session": per_session,
              "train_rows": int(len(y_train)), "seconds": round(time.monotonic() - start, 1)}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", choices=("forward", "loso", "r4", "all"), default="r4")
    ap.add_argument("--features", nargs="+", default=["rate250", "rate50", "temporal25", "temporal10", "temporal10_rate250"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    folds = []
    if args.folds in ("forward", "all"):
        folds.append(("forward", (1, 2, 3), 4))
    if args.folds in ("loso", "all"):
        for s in SESSIONS:
            folds.append((f"loso{s}", tuple(x for x in SESSIONS if x != s), s))
    mice = ("mice1",) if args.pilot else MICE
    results = {}
    if args.folds == "r4":
        for mouse in mice:
            for name in args.features:
                key = f"{mouse}/r4/{name}"
                print(f"run {key}", flush=True)
                r = run_r4(mouse, name, BINS[name], args.epochs, device)
                results[key] = r
                m = r["macro"]
                print(f"{key}: top1={m['top1']:.4f} r32={m['recall32']:.4f} "
                      f"nll={m['nll']:.4f} f1={m['macro_f1']:.4f} {r['seconds']}s", flush=True)
                (OUT / "TEMPORAL_RESULTS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return
    for mouse in mice:
        for fold, train_sessions, test_session in folds:
            for name in args.features:
                bin_ms = BINS[name]
                key = f"{mouse}/{fold}/{name}"
                print(f"run {key}", flush=True)
                r = run(mouse, train_sessions, test_session, name, bin_ms, args.epochs, device)
                results[key] = r
                print(f"{key}: top1={r['top1']:.4f} r32={r['recall32']:.4f} "
                      f"nll={r['nll']:.4f} f1={r['macro_f1']:.4f} {r['seconds']}s", flush=True)
                (OUT / "TEMPORAL_RESULTS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
