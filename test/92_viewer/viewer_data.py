"""Data access for the interactive spike-glyph viewer.

Recovers the exact public train trials behind every local heldout prediction row by replaying the
row construction of ``test/evaluate_final.py`` + ``src.final.predict_task`` with the production split
code (``src.splits.make_split``). Hidden test targets do not exist and are never read.
"""

import json
import struct
import sys
import threading
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import FeatureBlock  # noqa: E402
from src.splits import make_split  # noqa: E402

DATASET = ROOT / "data" / "siclink-mouse-visual-reconstruction-bci-challenge" / "siclink_chinese_character_decoding_dataset"
SUBMISSION_PNG = ROOT / "test/90_submission/artifacts/mlp_encoding_task2_rerank_task3_soft_v5/png"
SCORE_CACHE = ROOT / "test/.cache/92_viewer"
HELDOUT = {
    "task1": (ROOT / "test/.cache/evaluate_final_task1_v2/task1", ROOT / "test/.cache/evaluate_final_task1_v2/results.json"),
    "task2": (ROOT / "test/.cache/evaluate_final_task23_v1/task2", ROOT / "test/.cache/evaluate_final_task23_v1/resumed/results.json"),
    "task3": (ROOT / "test/.cache/evaluate_final_task23_v1/resumed/task3", ROOT / "test/.cache/evaluate_final_task23_v1/resumed/results.json"),
}
SPLIT_SEED = 42
PICTURE_SEED = 123
TASKS = ("task1", "task2", "task3")

_lock = threading.RLock()
_cache = {}


def cached(key, build):
    with _lock:
        if key not in _cache:
            _cache[key] = build()
        return _cache[key]


def catalog():
    return cached("catalog", lambda: json.loads((DATASET / "catalog.json").read_text(encoding="utf-8")))


def bin_ms():
    config = json.loads((ROOT / "src/final_config.json").read_text(encoding="utf-8"))
    return {t: int(config[t]["bin_ms"]) for t in TASKS}


def pictures():
    def build():
        with np.load(DATASET / "train" / "pictures.npz", allow_pickle=False) as z:
            return np.where(z["images"] >= 128, 255, 0).astype(np.uint8)
    return cached("pictures", build)


def picture_lookup():
    def build():
        return {p.tobytes(): i for i, p in enumerate(pictures())}
    return cached("picture_lookup", build)


def block_entry(split, task, mouse, session):
    for b in catalog()["blocks"]:
        if b["split"] == split and b["mouse"] == mouse and b["session"] == session and (split == "train" or b["task"] == task):
            return b
    raise KeyError(f"no block {split}/{task}/{mouse}/{session}")


def load_block(split, task, mouse, session):
    """Raw spike block (lazy, cached)."""
    def build():
        entry = block_entry(split, task, mouse, session)
        with np.load(DATASET / entry["path"], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data["n_units"] = int(entry["n_units"])
        data["mouse"], data["session"] = mouse, session
        return data
    return cached(("block", split, task, mouse, session), build)


def train_keys():
    return sorted((b["mouse"], b["session"]) for b in catalog()["blocks"] if b["split"] == "train")


def stub_blocks():
    """FeatureBlocks with real sample_ids/group_ptr/targets; features are zero-width (unused by make_split)."""
    def build():
        out = []
        for mouse, session in train_keys():
            d = load_block("train", "train", mouse, session)
            n_trials = len(d["trial_ptr"]) - 1
            out.append(FeatureBlock(mouse=mouse, session=session, split="train", task="train",
                                    sample_ids=d["sample_ids"], group_ptr=d["group_ptr"], targets=d["target"],
                                    features=np.zeros((n_trials, 0), np.float32), n_units=d["n_units"], bin_ms=1))
        return sorted(out, key=lambda b: (b.mouse, b.session))
    return cached("stub_blocks", build)


def heldout_pictures(task, split):
    if task == "task1":
        shuffled = np.random.default_rng(PICTURE_SEED).permutation(split.validation_pictures)
        return np.sort(shuffled[64:])
    order = np.random.default_rng(PICTURE_SEED).permutation(len(pictures()))
    return np.sort(order[256:768])


def reconstruct_rows(task):
    """Replay predict_task row order on the public split; return per-row group, target and source trials."""
    blocks = stub_blocks()
    split = make_split(blocks, task, seed=SPLIT_SEED)
    groups, targets, trials = [], [], []
    for block in blocks:
        vt = split.blocks[block.key].validation_trials
        if not len(vt):
            continue
        tt = block.trial_targets[vt]
        if task == "task3":
            for t, target in zip(vt, tt):
                groups.append(block.key)
                targets.append(int(target))
                trials.append([int(t)])
        else:
            order = np.argsort(tt, kind="stable")
            for target in np.unique(tt):
                groups.append(block.key)
                targets.append(int(target))
                trials.append([int(x) for x in vt[order][tt[order] == target]])
    held = heldout_pictures(task, split)
    keep = np.isin(np.asarray(targets), held)
    idx = np.flatnonzero(keep)
    return [groups[i] for i in idx], [targets[i] for i in idx], [trials[i] for i in idx]


def ocf_scores(task, preds, gts, kind):
    path = SCORE_CACHE / f"{task}_{kind}_ocf.npy"
    if path.exists():
        s = np.load(path)
        if len(s) == len(preds):
            return s
    import torch
    from src.evaluation import score_predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = np.asarray(score_predictions(preds, gts, device=device, batch_size=64), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, s)
    return s


def heldout(task):
    """All heldout rows for a task plus mapping status."""
    def build():
        folder, results = HELDOUT[task]
        with np.load(folder / "heldout_final.npz", allow_pickle=False) as z:
            pred, gt, grp, tidx = z["predictions"], z["targets"], z["group_ids"], z["target_indices"]
        oracle = None
        if (folder / "heldout_oracle.npz").exists():
            with np.load(folder / "heldout_oracle.npz", allow_pickle=False) as z:
                if np.array_equal(z["target_indices"], tidx) and np.array_equal(z["group_ids"], grp):
                    oracle = z["predictions"]
        recorded = json.loads(results.read_text(encoding="utf-8"))["tasks"][task]["heldout"]
        score = ocf_scores(task, pred, gt, "final")
        oracle_score = ocf_scores(task, oracle, gt, "oracle") if oracle is not None else None
        r_groups, r_targets, r_trials = reconstruct_rows(task)
        sequence_ok = (len(r_groups) == len(grp) and list(map(str, grp)) == r_groups
                       and np.array_equal(np.asarray(r_targets), tidx))
        lookup = picture_lookup()
        bank = pictures()
        rows = []
        for i in range(len(grp)):
            mouse, session = str(grp[i]).split("/")
            pred_idx = lookup.get(pred[i].tobytes()) if task != "task1" else None
            src = r_trials[i] if sequence_ok else None
            if src is not None:
                d = load_block("train", "train", mouse, session)
                if not all(int(d["target"][np.searchsorted(d["group_ptr"], t, side="right") - 1]) == int(tidx[i]) for t in src):
                    src = None
            row = {
                "row": i, "mouse": mouse, "session": session, "target": int(tidx[i]),
                "ocf": float(score[i]),
                "oracle_ocf": None if oracle_score is None else float(oracle_score[i]),
                "pred_index": pred_idx,
                "correct": None if task == "task1" else (pred_idx == int(tidx[i])),
                "trials": src,
                "sample_ids": None if src is None else [str(load_block("train", "train", mouse, session)["sample_ids"][
                    np.searchsorted(load_block("train", "train", mouse, session)["group_ptr"], t, side="right") - 1]) for t in src],
            }
            assert np.array_equal(gt[i], bank[tidx[i]])
            rows.append(row)
        return {
            "pred": pred, "gt": gt, "oracle": oracle, "rows": rows, "sequence_ok": sequence_ok,
            "recorded_mean": recorded["final"]["ocf_mean"],
            "recorded_oracle_mean": recorded.get("oracle", {}).get("ocf_mean"),
            "mean": float(score.mean()),
            "oracle_mean": None if oracle_score is None else float(oracle_score.mean()),
        }
    return cached(("heldout", task), build)


def trial_events(split, task, mouse, session, trial_indices):
    d = load_block(split, task, mouse, session)
    tp = d["trial_ptr"]
    out = []
    for t in trial_indices:
        a, b = int(tp[t]), int(tp[t + 1])
        out.append({"trial": int(t), "t": d["spike_samples"][a:b].tolist(), "u": d["unit_index"][a:b].tolist()})
    return out, d["n_units"]


def sample_trials(split, task, mouse, session, sample):
    d = load_block(split, task, mouse, session)
    g = d["group_ptr"]
    return list(range(int(g[sample]), int(g[sample + 1])))


def train_trials_for_glyph(glyph):
    out = []
    for mouse, session in train_keys():
        d = load_block("train", "train", mouse, session)
        samples = np.flatnonzero(d["target"] == glyph)
        trials = [t for s in samples for t in range(int(d["group_ptr"][s]), int(d["group_ptr"][s + 1]))]
        out.append({"mouse": mouse, "session": session, "samples": samples.tolist(),
                    "sample_ids": [str(d["sample_ids"][s]) for s in samples], "trials": trials})
    return out


def png_bytes(image):
    """Encode uint8 (H,W) gray or (H,W,3) RGB as PNG with stdlib only."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    color = 2 if image.ndim == 3 else 0
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def glyph_display(binary):
    """Black strokes on white, as in the GT glyph bank."""
    return (255 - np.asarray(binary, dtype=np.uint8))


def diff_image(gt, pred):
    """RGB: both=black, GT only=red (missed), pred only=blue (extra), neither=white."""
    g, p = gt >= 128, pred >= 128
    out = np.full(gt.shape + (3,), 255, np.uint8)
    out[g & p] = (20, 20, 20)
    out[g & ~p] = (230, 60, 60)
    out[~g & p] = (40, 110, 240)
    return out
