"""Feature likelihood as a ranking term on top of a 50 ms MLP classifier (frozen protocol)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

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

HASH = "fb86d45691cee89bf204aeae7855b27532a2180e5fcf2eb5d85bb259cb0e8c82"
LAMBDAS = (0.1, 0.25, 0.5, 1.0)
PRIMARY = 0.25
ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)


def row_z(scores):
    return (scores - scores.mean(1, keepdims=True)) / np.maximum(scores.std(1, keepdims=True), 1e-6)


def fit_mlp(x, y, epochs=10):
    torch.manual_seed(42)
    model = probe.SpikeClassifier(x.shape[1], probe.C, 1024, 0.0, False)
    model.set_normalization(x)
    model.cuda()
    xt = torch.from_numpy(x).cuda()
    yt = torch.from_numpy(y).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0, fused=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.from_numpy(np.random.default_rng(42 + epoch).permutation(len(yt))).cuda()
        for part in order.split(256):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = cross_entropy(model(xt[part]).float(), yt[part])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
    return model


def logits_of(model, x):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        parts = [model(a) for a in torch.from_numpy(x).cuda().split(256)]
    return torch.cat(parts).float().cpu().numpy()


def restricted_top32(fused, base):
    order = np.argpartition(-base, 32, axis=1)[:, :32]
    out = np.full_like(fused, -1e9)
    np.put_along_axis(out, order, np.take_along_axis(fused, order, axis=1), axis=1)
    return out


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("preregistration changed")
    gt = (np.asarray(np.load(probe.CACHE / "pictures.npy")) >= 128).astype(np.float32)
    bank = probe.glyph_features(gt)
    heldout = np.sort(np.random.default_rng(123).permutation(probe.C)[256:768])
    data = {m: probe.load_mouse(m, u) for m, u in probe.MICE.items()}
    result = {"experiment": "74_feature_rerank_verifier", "preregistration_sha256": HASH,
              "data_scope": "public train only", "folds": {}, "ocf_fold4": {}}
    for mouse, blocks in data.items():
        for k in probe.FOLDS:
            x = probe.normalize(blocks, k)
            r = np.concatenate([b["r"] for b in blocks])
            y = np.concatenate([b["y"] for b in blocks])
            tr, te = r != k, r == k
            val = [j for j in probe.FOLDS if j != k][0]
            inner = tr & (r != val)
            model = fit_mlp(x[tr], y[tr])
            logits = logits_of(model, x[te])
            del model
            ridge_inner = probe.Ridge().fit_input(x[inner]).fit_targets(bank[y[inner]])
            alpha = max(((a, float(np.median(probe.report(
                ridge_inner.predict(x[r == val], a), bank[y[r == val]], bank[y[inner]].mean(0))[0])))
                for a in ALPHAS), key=lambda t: t[1])[0]
            inner_r2 = probe.report(ridge_inner.predict(x[r == val], alpha),
                                    bank[y[r == val]], bank[y[inner]].mean(0))[0]
            ridge = probe.Ridge().fit_input(x[tr]).fit_targets(bank[y[tr]])
            sigma2 = np.clip(1.0 - inner_r2, 1e-3, 1.0).astype(np.float32)
            scale = (1.0 / np.sqrt(sigma2)).astype(np.float32)
            ps, bs = ridge.predict(x[te], alpha) * scale, bank * scale
            dist = (ps ** 2).sum(1)[:, None] + (bs ** 2).sum(1)[None, :] - 2.0 * ps @ bs.T
            feat = row_z(-dist).astype(np.float32)
            del dist, ps, bs
            base = row_z(logits).astype(np.float32)
            del logits
            row = {"alpha": alpha, "arms": {}}
            variants = {"base": base, "top32": restricted_top32(base + PRIMARY * feat, base)}
            for lam in LAMBDAS:
                variants[f"fused_{lam}"] = base + lam * feat
            for name, sc in variants.items():
                row["arms"][name] = {"top1": float((sc.argmax(1) == y[te]).mean())}
                if k == 4 and name in ("base", "fused_0.25", "top32"):
                    per_session = []
                    for s, block in enumerate(blocks, start=1):
                        labels = block["y"][block["r"] == 4]
                        sl = sc[(s - 1) * probe.C:s * probe.C]
                        assigned = probe.unique_assignment(sl)
                        mask = np.isin(labels, heldout)
                        values = probe.score_predictions(
                            gt[assigned[mask]], gt[labels[mask]], device="cuda", batch_size=16)
                        entry = {"ocf": float(values.mean()),
                                 "top1_assigned": float((assigned == labels).mean())}
                        result["ocf_fold4"].setdefault(f"{mouse}/session{s}", {})[name] = entry
                        per_session.append(entry["ocf"])
                    row["arms"][name]["ocf_mean"] = float(np.mean(per_session))
            row["recall32_base"] = float(
                (np.argpartition(-base, 32, axis=1)[:, :32] == y[te][:, None]).any(1).mean())
            result["folds"][f"{mouse}/fold{k}"] = row
            print(mouse, k, {a: round(v["top1"], 5) for a, v in row["arms"].items()},
                  "recall32", round(row["recall32_base"], 4),
                  "ocf", {a: round(row["arms"][a]["ocf_mean"], 4)
                          for a in ("base", "fused_0.25", "top32") if "ocf_mean" in row["arms"][a]},
                  flush=True)
    summary = {}
    for mouse in probe.MICE:
        summary[mouse] = {}
        for arm in result["folds"][f"{mouse}/fold1"]["arms"]:
            summary[mouse][arm] = {
                "top1": float(np.mean([result["folds"][f"{mouse}/fold{k}"]["arms"][arm]["top1"]
                                       for k in probe.FOLDS]))}
        for arm in ("base", "fused_0.25", "top32"):
            vals = [result["ocf_fold4"][f"{mouse}/session{s}"][arm]["ocf"] for s in range(1, 5)]
            summary[mouse][arm]["ocf_mean"] = float(np.mean(vals))
            summary[mouse][arm]["sessions_positive_vs_base"] = int(sum(
                v >= result["ocf_fold4"][f"{mouse}/session{s}"]["base"]["ocf"]
                for s, v in enumerate(vals, start=1)))
    result["summary"] = summary
    result["pass_rule"] = evaluate(result)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print("summary", json.dumps(summary, indent=2), flush=True)
    print("pass_rule", result["pass_rule"], flush=True)


def evaluate(result):
    s = result["summary"]
    top1_up = all(s[m]["fused_0.25"]["top1"] > s[m]["base"]["top1"] for m in probe.MICE)
    ocf_up = all(s[m]["fused_0.25"]["ocf_mean"] > s[m]["base"]["ocf_mean"] for m in probe.MICE)
    positive = sum(s[m]["fused_0.25"]["sessions_positive_vs_base"] for m in probe.MICE)
    return {"top1_up_both": bool(top1_up), "ocf_up_both": bool(ocf_up),
            "sessions_non_negative": int(positive), "advance": bool(top1_up and ocf_up and positive >= 5)}


if __name__ == "__main__":
    main()
