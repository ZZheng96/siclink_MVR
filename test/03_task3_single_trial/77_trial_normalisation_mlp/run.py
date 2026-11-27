"""Does trial-wise normalisation still help a learnable MLP readout?"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = load("probe", ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py")
exp74 = load("exp74", ROOT / "test/03_task3_single_trial/74_feature_rerank_verifier/run.py")

HASH = "e02c782a695cf0c18911b1f4f7d2124868d170212f7ec03ec36c96566465efae"


def trial_z(x):
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)


def arm_input(x, arm):
    return trial_z(x) if arm == "trial_z" else x


def recall(scores, labels, ks=(1, 32, 100)):
    width = max(ks)
    idx = np.argpartition(-scores, width, axis=1)[:, :width]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.take_along_axis(idx, np.argsort(-vals, axis=1), axis=1)
    hit = order == labels[:, None]
    return {k: float(hit[:, :k].any(1).mean()) for k in ks}


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("preregistration changed")
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(probe.C)[256:768])
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "77_trial_normalisation_mlp", "preregistration_sha256": HASH,
              "folds": {}, "summary": {}, "ocf_fold4": {}}
    for mouse, blocks in data.items():
        for arm in ("base", "trial_z"):
            per_fold = {}
            for k in probe.FOLDS:
                x = arm_input(probe.normalize(blocks, k), arm)
                r = np.concatenate([b["r"] for b in blocks])
                y = np.concatenate([b["y"] for b in blocks])
                tr, te = r != k, r == k
                model = exp74.fit_mlp(x[tr], y[tr])
                logits = exp74.logits_of(model, x[te])
                del model
                per_fold[k] = recall(logits, y[te])
                if k == 4:
                    for s, block in enumerate(blocks, start=1):
                        labels = block["y"][block["r"] == 4]
                        sl = logits[(s - 1) * probe.C:s * probe.C]
                        assigned = probe.unique_assignment(sl)
                        mask = np.isin(labels, heldout)
                        values = probe.score_predictions(
                            gt[assigned[mask]], gt[labels[mask]], device="cuda", batch_size=16)
                        result["ocf_fold4"].setdefault(f"{mouse}/session{s}", {})[arm] = {
                            "ocf": float(values.mean()),
                            "top1_assigned": float((assigned == labels).mean())}
                del logits
                print(mouse, arm, k, {kk: round(v, 4) for kk, v in per_fold[k].items()},
                      flush=True)
            result["folds"][f"{mouse}/{arm}"] = {str(k): v for k, v in per_fold.items()}
            result["summary"].setdefault(mouse, {})[arm] = {
                f"recall{kk}": float(np.mean([per_fold[k][kk] for k in probe.FOLDS]))
                for kk in (1, 32, 100)}
            result["summary"][mouse][arm]["folds_better_recall32"] = int(sum(
                per_fold[k][32] > result["folds"][f"{mouse}/base"][str(k)][32]
                for k in probe.FOLDS))
            result["summary"][mouse][arm]["ocf"] = float(np.mean(
                [result["ocf_fold4"][f"{mouse}/session{s}"][arm]["ocf"] for s in range(1, 5)]))
    result["pass_rule"] = evaluate(result)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print("summary", json.dumps(result["summary"], indent=2), flush=True)
    print("pass_rule", result["pass_rule"], flush=True)


def evaluate(result):
    mice = ("mice1", "mice2")
    wins = all(result["summary"][m]["trial_z"]["folds_better_recall32"] >= 3 for m in mice)
    ocf = all(result["summary"][m]["trial_z"]["ocf"] > result["summary"][m]["base"]["ocf"]
              for m in mice)
    return {"recall32_wins": {m: result["summary"][m]["trial_z"]["folds_better_recall32"]
                              for m in mice},
            "ocf_up_both": bool(ocf), "advance": bool(wins and ocf)}


if __name__ == "__main__":
    main()
