"""Development-fold classification metrics and epoch lock for repeat-holdout v1.

Official OCF is deliberately not computed here. Epochs are locked by mean validation
NLL over F1/F2 and both mice (tie-break: macro Top-1, then earlier epoch). This is a
recorded protocol deviation requested by the user; R4 is still untouched.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_repeat as T

CACHE = T.RUN / "DEV_CLASSIFICATION.json"
LOCK = T.RUN / "DEV_LOCK.json"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def predict_probs(model, x, device, batch=2048):
    outs = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            logits, _ = model(x[i:i + batch])
            outs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(outs)


def metrics(prob, y):
    n = len(y)
    top1 = float((prob.argmax(1) == y).mean())
    top32 = np.argpartition(-prob, 32, axis=1)[:, :32]
    recall32 = float((top32 == y[:, None]).any(1).mean())
    nll = float(-np.log(np.maximum(prob[np.arange(n), y], 1e-12)).mean())
    pred = prob.argmax(1)
    den = np.bincount(y, minlength=T.K) + np.bincount(pred, minlength=T.K)
    tp = np.bincount(y[y == pred], minlength=T.K)
    macro_f1 = float(np.mean(2 * tp / np.maximum(den, 1)))
    return {"top1": top1, "recall32": recall32, "nll": nll, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    T.RUN.mkdir(parents=True, exist_ok=True)

    results = {}
    for fold in ("F1", "F2"):
        for mouse in T.UNITS:
            x, y, _, _ = T.load_data(fold, mouse, "val")
            mean, scale = T.load_scaler(fold, mouse)
            x = (x - mean) / scale
            xt = torch.from_numpy(x).to(device)
            for model in ("B", "C"):
                for epoch in T.EPOCHS:
                    probs, per_seed = [], []
                    for seed in T.SEEDS:
                        net = T.load_model(fold, mouse, model, seed, epoch, device)
                        p = predict_probs(net, xt, device)
                        probs.append(p)
                        m = metrics(p, y)
                        per_seed.append({"seed": seed, "top1": m["top1"], "nll": m["nll"]})
                        del net
                    ens = np.mean(probs, axis=0)
                    row = metrics(ens, y)
                    row["per_seed"] = per_seed
                    row["top1_seed_sd"] = float(np.std([r["top1"] for r in per_seed], ddof=1))
                    row["nll_seed_sd"] = float(np.std([r["nll"] for r in per_seed], ddof=1))
                    results[f"{fold}/m{mouse}/{model}/{epoch}"] = row
                    print(f"{fold} m{mouse} {model} e{epoch}: top1={row['top1']:.4f} "
                          f"r32={row['recall32']:.4f} nll={row['nll']:.4f} "
                          f"macroF1={row['macro_f1']:.4f}", flush=True)
            del xt

    selected = {}
    for model in ("B", "C"):
        rows = []
        for epoch in T.EPOCHS:
            keys = [f"{f}/m{m}/{model}/{epoch}" for f in ("F1", "F2") for m in T.UNITS]
            nll = float(np.mean([results[k]["nll"] for k in keys]))
            top1 = float(np.mean([results[k]["top1"] for k in keys]))
            worst_top1 = float(min(results[k]["top1"] for k in keys))
            rows.append({"epoch": epoch, "mean_nll": nll, "mean_top1": top1, "worst_top1": worst_top1})
        best = min(rows, key=lambda r: (r["mean_nll"], -r["mean_top1"], r["epoch"]))
        selected[model] = dict(best, table=rows)
        print(f"selected {model}: epoch {best['epoch']} (mean NLL {best['mean_nll']:.4f})", flush=True)

    payload = {
        "protocol": "repeat-holdout v1",
        "deviation": "official OCF not computed; epoch locked by F1/F2 mean validation NLL",
        "utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "script_sha256": sha(Path(__file__)),
        "train_done_sha256": sha(T.RUN / "TRAIN_DONE.json"),
        "selection": selected,
        "metrics": results,
    }
    CACHE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lock = {k: payload[k] for k in ("protocol", "deviation", "utc", "script_sha256", "train_done_sha256")}
    lock.update({"selected_epochs": {m: selected[m]["epoch"] for m in selected},
                 "selection_metric": {m: selected[m]["table"] for m in selected}})
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(lock["selected_epochs"]))


if __name__ == "__main__":
    main()
