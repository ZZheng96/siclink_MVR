"""Distribution of official OCF under random positive per-row score weights."""

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

FOLD = 4
DRAWS = 20


def ocf(sc, blocks, gt, heldout):
    per_session = []
    for s, block in enumerate(blocks, start=1):
        labels = block["y"][block["r"] == FOLD]
        sl = sc[(s - 1) * probe.C:s * probe.C]
        assigned = probe.unique_assignment(sl)
        mask = np.isin(labels, heldout)
        per_session.append(float(probe.score_predictions(
            gt[assigned[mask]], gt[labels[mask]], device="cuda", batch_size=16).mean()))
    return float(np.mean(per_session))


def main():
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(probe.C)[256:768])
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "79_assignment_noise_band", "fold": FOLD, "draws": DRAWS,
              "runs": {}, "summary": {}}
    rng = np.random.default_rng(20261127)
    for mouse, blocks in data.items():
        x = probe.normalize(blocks, FOLD)
        r = np.concatenate([b["r"] for b in blocks])
        y = np.concatenate([b["y"] for b in blocks])
        tr, te = r != FOLD, r == FOLD
        model = exp74.fit_mlp(x[tr], y[tr])
        logits = exp74.logits_of(model, x[te])
        del model
        base = exp74.row_z(logits).astype(np.float32)
        canonical = ocf(base, blocks, gt, heldout)
        values = []
        for draw in range(DRAWS):
            weight = rng.uniform(0.5, 2.0, size=(len(base), 1)).astype(np.float32)
            values.append(ocf(base * weight, blocks, gt, heldout))
            print(mouse, draw + 1, round(values[-1], 4), flush=True)
        values = np.array(values)
        result["runs"][mouse] = {"canonical_row_z": canonical, "draws": values.tolist()}
        result["summary"][mouse] = {
            "canonical": canonical, "mean": float(values.mean()),
            "std": float(values.std(ddof=1)), "min": float(values.min()),
            "max": float(values.max()), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "range": float(values.max() - values.min())}
        print(mouse, "summary", result["summary"][mouse], flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
