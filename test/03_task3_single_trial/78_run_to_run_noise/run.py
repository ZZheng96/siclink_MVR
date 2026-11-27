"""Run-to-run variability of the Task 3 pipeline under identical settings."""

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
REPEATS = 3


def recall32(scores, labels):
    idx = np.argpartition(-scores, 32, axis=1)[:, :32]
    return float((idx == labels[:, None]).any(1).mean())


def ocf_of(logits, blocks, gt, heldout, device="cuda"):
    per_session = []
    for s, block in enumerate(blocks, start=1):
        labels = block["y"][block["r"] == FOLD]
        sl = logits[(s - 1) * probe.C:s * probe.C]
        assigned = probe.unique_assignment(sl)
        mask = np.isin(labels, heldout)
        values = probe.score_predictions(gt[assigned[mask]], gt[labels[mask]],
                                         device=device, batch_size=16)
        per_session.append(float(values.mean()))
    return float(np.mean(per_session)), per_session


def main():
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    heldout = np.sort(np.random.default_rng(123).permutation(probe.C)[256:768])
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "78_run_to_run_noise", "fold": FOLD, "repeats": REPEATS,
              "runs": {}, "summary": {}, "control": {}}
    for mouse, blocks in data.items():
        x = probe.normalize(blocks, FOLD)
        r = np.concatenate([b["r"] for b in blocks])
        y = np.concatenate([b["y"] for b in blocks])
        tr, te = r != FOLD, r == FOLD
        rows = []
        for rep in range(1, REPEATS + 1):
            model = exp74.fit_mlp(x[tr], y[tr])
            logits = exp74.logits_of(model, x[te])
            del model
            row = {"repeat": rep, "top1": float((logits.argmax(1) == y[te]).mean()),
                   "recall32": recall32(logits, y[te])}
            row["ocf"], row["per_session"] = ocf_of(logits, blocks, gt, heldout)
            if rep == 1:
                again, _ = ocf_of(logits, blocks, gt, heldout)
                result["control"][mouse] = {"ocf_first": row["ocf"], "ocf_second": again,
                                            "identical": bool(abs(again - row["ocf"]) < 1e-12)}
            rows.append(row)
            print(mouse, row, flush=True)
            del logits
        result["runs"][mouse] = rows
        result["summary"][mouse] = {}
        for key in ("top1", "recall32", "ocf"):
            vals = np.array([row[key] for row in rows])
            result["summary"][mouse][key] = {
                "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                "min": float(vals.min()), "max": float(vals.max()),
                "range": float(vals.max() - vals.min())}
        print(mouse, "summary", json.dumps(result["summary"][mouse], indent=2), flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
