"""How many dimensions does the character information occupy?"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "probe", str(ROOT / "test/03_task3_single_trial/70_glyph_feature_probe/run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

KS = (8, 16, 32, 64, 128, 256, None)
SPECTRUM_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def recall32(scores, labels):
    idx = np.argpartition(-scores, 32, axis=1)[:, :32]
    return float((idx == labels[:, None]).any(1).mean())


def centroids(x, y):
    return np.stack([x[y == c].mean(0) for c in range(probe.C)])


def main():
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "81_information_dimensionality", "spectrum": {}, "curve": {}}
    for mouse, blocks in data.items():
        spectrum = {k: [] for k in SPECTRUM_KS}
        curve = {str(k): [] for k in KS}
        for k in probe.FOLDS:
            for block in blocks:
                tr = block["r"] != k
                te = block["r"] == k
                x = block["x"]
                mu = x[tr].mean(0, dtype=np.float64).astype(np.float32)
                sd = np.maximum(x[tr].std(0, dtype=np.float64).astype(np.float32), 1e-5)
                z = (x - mu) / sd
                ytr, yte = block["y"][tr], block["y"][te]
                ztr, zte = z[tr], z[te]
                # between-class spectrum on training repeats
                means = centroids(ztr, ytr)
                s = np.linalg.svd(means, compute_uv=False)
                cum = np.cumsum(s ** 2) / (s ** 2).sum()
                for kk in SPECTRUM_KS:
                    spectrum[kk].append(float(cum[kk - 1]))
                # decoding curve
                u, sv, vh = np.linalg.svd(ztr - ztr.mean(0), full_matrices=False)
                centre = ztr.mean(0)
                for kk in KS:
                    width = vh.shape[0] if kk is None else min(kk, vh.shape[0])
                    w = vh[:width].T
                    a = (ztr - centre) @ w
                    b = (zte - centre) @ w
                    mu_c = centroids(a, ytr)
                    sc = -(np.square(b).sum(1)[:, None] + np.square(mu_c).sum(1)[None, :]
                           - 2 * b @ mu_c.T)
                    curve[str(kk)].append(recall32(sc, yte))
        result["spectrum"][mouse] = {str(kk): float(np.mean(v)) for kk, v in spectrum.items()}
        result["curve"][mouse] = {kk: float(np.mean(v)) for kk, v in curve.items()}
        print(mouse, "spectrum", result["spectrum"][mouse], flush=True)
        print(mouse, "curve", result["curve"][mouse], flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
