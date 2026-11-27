"""Does PCA truncation to 128 dimensions help the MLP readout?"""

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

spec = importlib.util.spec_from_file_location(
    "probe", str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
spec74 = importlib.util.spec_from_file_location(
    "exp74", str(ROOT / "test/03_task3_single_trial/74_feature_rerank_verifier/run.py"))
exp74 = importlib.util.module_from_spec(spec74)
spec74.loader.exec_module(exp74)

HASH = "1628b900345ff7b3f05cdbddc9ae8a3bfc97d7f312549a2d51f3f4de3ea38d1c"
WIDTH = 128


def truncated(blocks, k):
    """Per-session PCA fitted on the training repeats of this fold."""
    parts = []
    for block in blocks:
        tr = block["r"] != k
        x = block["x"]
        mu = x[tr].mean(0, dtype=np.float64).astype(np.float32)
        sd = np.maximum(x[tr].std(0, dtype=np.float64).astype(np.float32), 1e-5)
        z = (x - mu) / sd
        centre = z[tr].mean(0)
        _, _, vh = np.linalg.svd(z[tr] - centre, full_matrices=False)
        w = vh[:WIDTH].T
        parts.append(((z - centre) @ w).astype(np.float32))
    return np.concatenate(parts)


def recall32(scores, labels):
    idx = np.argpartition(-scores, 32, axis=1)[:, :32]
    return float((idx == labels[:, None]).any(1).mean())


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("preregistration changed")
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(probe.C)[256:768])
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "82_pca_truncation_mlp", "preregistration_sha256": HASH,
              "folds": {}, "summary": {}, "ocf_fold4": {}}
    for mouse, blocks in data.items():
        for arm in ("base", "pca128"):
            per_fold = {}
            for k in probe.FOLDS:
                x = truncated(blocks, k) if arm == "pca128" else probe.normalize(blocks, k)
                r = np.concatenate([b["r"] for b in blocks])
                y = np.concatenate([b["y"] for b in blocks])
                tr, te = r != k, r == k
                model = exp74.fit_mlp(x[tr], y[tr])
                logits = exp74.logits_of(model, x[te])
                del model, x
                per_fold[k] = {"top1": float((logits.argmax(1) == y[te]).mean()),
                               "recall32": recall32(logits, y[te])}
                if k == 4:
                    for s, block in enumerate(blocks, start=1):
                        labels = block["y"][block["r"] == 4]
                        sl = exp74.row_z(logits[(s - 1) * probe.C:s * probe.C])
                        assigned = probe.unique_assignment(sl)
                        mask = np.isin(labels, heldout)
                        values = probe.score_predictions(
                            gt[assigned[mask]], gt[labels[mask]], device="cuda", batch_size=16)
                        result["ocf_fold4"].setdefault(f"{mouse}/session{s}", {})[arm] = {
                            "ocf": float(values.mean()),
                            "top1_assigned": float((assigned == labels).mean())}
                del logits
                print(mouse, arm, k, per_fold[k], flush=True)
            result["folds"][f"{mouse}/{arm}"] = {str(k): v for k, v in per_fold.items()}
            result["summary"].setdefault(mouse, {})[arm] = {
                "top1": float(np.mean([per_fold[k]["top1"] for k in probe.FOLDS])),
                "recall32": float(np.mean([per_fold[k]["recall32"] for k in probe.FOLDS])),
                "ocf": float(np.mean([result["ocf_fold4"][f"{mouse}/session{s}"][arm]["ocf"]
                                      for s in range(1, 5)]))}
            result["summary"][mouse][arm]["folds_better_recall32"] = int(sum(
                per_fold[k]["recall32"] > result["folds"][f"{mouse}/base"][str(k)]["recall32"]
                for k in probe.FOLDS))
    result["pass_rule"] = {
        "recall32_wins": {m: result["summary"][m]["pca128"]["folds_better_recall32"]
                          for m in probe.MICE},
        "ocf_delta": {m: result["summary"][m]["pca128"]["ocf"]
                      - result["summary"][m]["base"]["ocf"] for m in probe.MICE}}
    result["pass_rule"]["advance"] = bool(
        all(result["pass_rule"]["recall32_wins"][m] >= 3 for m in probe.MICE)
        and all(result["pass_rule"]["ocf_delta"][m] > 0 for m in probe.MICE))
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(result["pass_rule"], flush=True)


if __name__ == "__main__":
    main()
