"""Exploratory diagnostic: Task1 font-bank rendering ceiling.

exploratory; not preregistered; not used for production selection.

The script measures, on public data only, how well a NotoSansSC-VF rendering of the
6,763 GB2312 characters scores on the official OCF metric when the *true* binary
picture is used as the query.  This separates "the font bank cannot render a
character" (rendering ceiling) from "the neural decoder cannot retrieve/rank it"
(decoder loss).

`--smoke-test` exercises retrieval, official OCF, streamed chunk maxima and IoU on a
synthetic six-glyph bank and never reads the real font or pictures.  `--run` uses the
real font and ``test/.cache/shared_preprocessing_v1/pictures.npy``.

Official OCF costs roughly 260-330 image pairs/second on this RTX 3080, so the full
2,748 x 6,763 scan (18.6M pairs, ~16-20h) is not runnable.  K=1/8/32 is evaluated on
all 2,748 pictures; K=100/500/6763 is evaluated on a wall-clock-budgeted sample of the
211 fixed-protocol held-out pictures.  Every scope is recorded in the JSON.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation import score_predictions
from src.retrieval import _retrieval_embedding, render_gb2312_bank

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "test/.cache/shared_preprocessing_v1"
PICTURES = CACHE / "pictures.npy"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
RESULTS = HERE / "exploratory_results.json"
SIMILARITY_NPZ = HERE / "similarity_directional_multi.npz"

DEVICE = "cuda"
KS = (1, 8, 32, 100, 500, 6763)
SMALL_KS = (1, 8, 32)
EARLY_KS = (100, 500)
SPLIT_SEED = 42
PICTURE_SEED = 123
FONT_SIZE = 52
VARIATION = "Regular"
PAIR_CHUNK = 256
EMBED_CHUNK = 1024
MAX_FULL_ROWS = 64
MIN_FULL_ROWS = 8
V5_HELDOUT_FINAL = 18.43706
V5_HELDOUT_ORACLE32 = 33.01849
DISCLAIMER = "exploratory; not preregistered; not used for production selection"


# ---------------------------------------------------------------- data helpers


def load_gt():
    pictures = np.load(PICTURES)
    if pictures.shape != (2748, 64, 64):
        raise ValueError(f"Unexpected pictures shape: {pictures.shape}")
    return (pictures >= 128).astype(np.float32)


def heldout_picture_rows():
    """Recreate evaluate_final's fixed Task1 held-out picture list (211 rows)."""
    from src.preprocessing import load_cached_blocks
    from src.splits import make_split

    blocks = load_cached_blocks(CACHE)
    split = make_split(blocks, "task1", seed=SPLIT_SEED)
    shuffled = np.random.default_rng(PICTURE_SEED).permutation(split.validation_pictures)
    return np.sort(shuffled[64:]).astype(np.int64)


def foreground(images):
    return images.reshape(len(images), -1).sum(axis=1)


def iou(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum(axis=(1, 2))
    return (np.logical_and(a, b).sum(axis=(1, 2)) / np.maximum(union, 1)).astype(np.float64)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


# ---------------------------------------------------------------- embeddings


@torch.inference_mode()
def embed(images, metric, chunk=EMBED_CHUNK):
    """L2-normalized `_retrieval_embedding` for a binary image bank."""
    output = []
    for start in range(0, len(images), chunk):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + chunk, None])).to(DEVICE)
        features = _retrieval_embedding(batch, metric)
        output.append(F.normalize(features, dim=1))
    return torch.cat(output, dim=0)


@torch.inference_mode()
def similarity(gt_emb, bank_emb, chunk=256):
    bank_t = bank_emb.to(DEVICE)
    output = np.empty((len(gt_emb), len(bank_emb)), dtype=np.float32)
    for start in range(0, len(gt_emb), chunk):
        product = gt_emb[start:start + chunk].to(DEVICE) @ bank_t.T
        output[start:start + chunk] = product.cpu().numpy()
    del bank_t
    return output


def ranked_candidates(sim_rows):
    """Stable descending similarity order for each row."""
    return np.argsort(-np.asarray(sim_rows, dtype=np.float32), axis=1, kind="stable")


# ---------------------------------------------------------------- official OCF


def ocf_pairs(bank, gt, bank_index, gt_index, chunk=PAIR_CHUNK, label=None):
    """Official per-pair OCF (PNG path) for aligned candidate/target index arrays."""
    bank_index = np.asarray(bank_index, dtype=np.int64)
    gt_index = np.asarray(gt_index, dtype=np.int64)
    if len(bank_index) != len(gt_index):
        raise ValueError("candidate and target index arrays must align.")
    scores = np.empty(len(bank_index), dtype=np.float64)
    start_time = time.monotonic()
    for start in range(0, len(bank_index), chunk):
        stop = min(len(bank_index), start + chunk)
        scores[start:stop] = score_predictions(
            bank[bank_index[start:stop]],
            gt[gt_index[start:stop]],
            device=DEVICE,
            batch_size=stop - start,
        )
        if label is not None and start and (start // chunk) % 25 == 0:
            rate = start / max(time.monotonic() - start_time, 1e-9)
            print(f"  {label}: {start}/{len(bank_index)} pairs ({rate:.0f}/s)", flush=True)
    elapsed = time.monotonic() - start_time
    if label is not None:
        rate = len(bank_index) / max(elapsed, 1e-9)
        print(f"  {label}: done {len(bank_index)} pairs in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return scores.astype(np.float32), elapsed


def topk_grid_scores(bank, gt, order, order_positions, gt_rows, ks):
    """Official OCF for the first max(ks) candidates of each row; row-wise cummax."""
    width = max(ks)
    selected = np.asarray(order)[np.asarray(order_positions, dtype=np.int64), :width]
    gt_index = np.repeat(np.asarray(gt_rows, dtype=np.int64), width)
    flat_scores, elapsed = ocf_pairs(
        bank, gt, selected.reshape(-1), gt_index, label=f"top{width}-grid"
    )
    matrix = flat_scores.reshape(len(gt_index) // width, width)
    grid = {k: np.maximum.accumulate(matrix[:, :k], axis=1)[:, -1] for k in ks}
    return grid, matrix, selected, elapsed


def stream_full_bank(bank, gt, orders, rows, early_ks=EARLY_KS, chunk=PAIR_CHUNK,
                     deadline=None):
    """Exact per-row max over all 6,763 bank glyphs in two-dimensional chunks.

    Never materialises a (rows, 6763) score matrix: every chunk is at most ``chunk``
    pairs (at most chunk x 64 x 64 floats) and only running maxima are retained.
    """
    width = max(early_ks) if early_ks else 0
    best = np.empty(len(rows), dtype=np.float64)
    argmax = np.empty(len(rows), dtype=np.int64)
    early = np.empty((len(rows), width), dtype=np.float32) if width else None
    start_time = time.monotonic()
    scanned = 0
    for position, (row, order) in enumerate(zip(rows, orders)):
        if deadline is not None and time.monotonic() > deadline and position >= MIN_FULL_ROWS:
            best = best[:position]
            argmax = argmax[:position]
            if early is not None:
                early = early[:position]
            print(f"  full-bank: budget reached after {position} rows", flush=True)
            break
        target = np.repeat(gt[row][None], min(chunk, len(order)), axis=0)
        row_best = -np.inf
        row_arg = -1
        for start in range(0, len(order), chunk):
            stop = min(len(order), start + chunk)
            candidates = order[start:stop]
            values = score_predictions(
                bank[candidates], target[:stop - start], device=DEVICE, batch_size=stop - start
            )
            if early is not None and start < width:
                tail = min(stop, width)
                early[position, start:tail] = values[:tail - start]
            offset = int(np.argmax(values))
            if values[offset] > row_best:
                row_best = float(values[offset])
                row_arg = int(candidates[offset])
        best[position] = row_best
        argmax[position] = row_arg
        scanned += len(order)
        elapsed = time.monotonic() - start_time
        print(
            f"  full-bank row {position + 1}/{len(rows)} row_max={row_best:.3f} "
            f"argmax={row_arg} elapsed={elapsed:.0f}s",
            flush=True,
        )
    elapsed = time.monotonic() - start_time
    early_grid = {}
    if early is not None:
        early_grid = {k: np.maximum.accumulate(early[:, :k], axis=1)[:, -1] for k in early_ks}
    return best, argmax, early_grid, elapsed, scanned


# ---------------------------------------------------------------- diagnostics


def mapping_probe():
    """Look for any public per-picture character label without guessing a mapping."""
    catalog_path = (
        ROOT
        / "data/siclink-mouse-visual-reconstruction-bci-challenge/_/参赛者包和案例/data/catalog.json"
    )
    evidence = []
    suspicious = ("char", "label", "unicode", "glyph", "text", "hanzi", "codepoint")
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        keys = set(catalog)
        for block in catalog.get("blocks", []):
            keys |= set(block)
        hits = sorted(k for k in keys if any(token in k.lower() for token in suspicious))
        evidence.append(f"catalog.json top-level/block keys checked; character-label fields: {hits}")
    else:
        evidence.append("catalog.json not found")
    train_npz = sorted(
        (ROOT / "data/siclink-mouse-visual-reconstruction-bci-challenge/"
         "siclink_chinese_character_decoding_dataset/train").glob("*/session*.npz")
    )
    if train_npz:
        with np.load(train_npz[0]) as data:
            keys = list(data.files)
            sample_ids = data["sample_ids"][:3].tolist() if "sample_ids" in keys else []
        hits = sorted(k for k in keys if any(token in k.lower() for token in suspicious))
        evidence.append(f"{train_npz[0].name} array keys checked; character-label fields: {hits}")
        evidence.append(f"sample_ids example: {sample_ids}")
    else:
        evidence.append("no public train npz found")
    return {
        "available": False,
        "computed": False,
        "evidence": evidence,
        "note": "no public character mapping; not computed",
    }


def conclusion(payload):
    upper = payload["full_bank_sample"]["ocf"]["mean"]
    ci_low, ci_high = payload["full_bank_sample"]["ocf"]["ci95"]
    rows = np.asarray(payload["full_bank_sample"]["row_best"], dtype=np.float64)
    fraction_at_least_40 = float(np.mean(rows >= 40.0))
    oracle32_gt = payload["k_grid_all"]["ocf"]["32"]["mean"]
    if ci_high < 40.0:
        reachability = (
            "UNREACHABLE at the font-bank level: the full-bank ceiling "
            f"{upper:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f}) is below 40, so even "
            "perfect character retrieval cannot reach 40 with this font rendering."
        )
        requirement = (
            "none under this bank; the binding limit is font/bank rendering, not retrieval"
        )
    elif ci_low >= 40.0:
        reachability = (
            "REACHABLE at the font-bank level: the full-bank ceiling "
            f"{upper:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f}) exceeds 40; the gap to "
            "18.43706 is decoder/retrieval loss, not rendering."
        )
        requirement = (
            f"retrieval and ranking must reach the per-row optimum; only "
            f"{fraction_at_least_40 * 100:.1f}% of sampled rows already have a bank "
            "candidate scoring >=40"
        )
    else:
        reachability = (
            "UNCERTAIN at the font-bank level: the full-bank 95% CI "
            f"{ci_low:.3f}-{ci_high:.3f} straddles 40 (mean {upper:.3f}); a larger "
            "full-bank sample is required."
        )
        requirement = "larger full-bank sample"
    return {
        "v5_heldout_final_ocf": V5_HELDOUT_FINAL,
        "v5_heldout_oracle_at_32": V5_HELDOUT_ORACLE32,
        "gt_query_oracle_at_32_mean": oracle32_gt,
        "gt_query_oracle_at_32_minus_v5_oracle": oracle32_gt - V5_HELDOUT_ORACLE32,
        "full_bank_ceiling_mean": upper,
        "full_bank_ceiling_ci95": [ci_low, ci_high],
        "full_bank_ceiling_minus_v5_final": upper - V5_HELDOUT_FINAL,
        "fraction_sampled_rows_with_bank_best_at_least_40": fraction_at_least_40,
        "reachability": reachability,
        "retrieval_requirement": requirement,
    }


# ---------------------------------------------------------------- smoke test


def synthetic_glyphs():
    """Six distinct hand-drawn glyphs plus approximate GT (shifted copies)."""
    size = 64
    specs = [
        [(8, 32, 56, 32)],  # horizontal
        [(32, 8, 32, 56)],  # vertical
        [(8, 32, 56, 32), (32, 8, 32, 56)],  # cross
        [(8, 8, 8, 56), (8, 56, 48, 56)],  # L
        [(8, 8, 56, 8), (32, 8, 32, 56)],  # T
        [(24, 8, 8, 24), (8, 8, 8, 56), (8, 56, 56, 56), (56, 56, 56, 24),
         (56, 24, 24, 24), (24, 24, 8, 24)],  # box outline
    ]
    bank = np.zeros((len(specs), size, size), dtype=np.float32)
    for index, lines in enumerate(specs):
        canvas = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(canvas)
        for line in lines:
            draw.line(line, fill=255, width=5)
        bank[index] = (np.asarray(canvas) >= 128).astype(np.float32)
    gt = bank.copy()
    for index, shift in enumerate((2, -1, 1, 2, -2, 1)):
        axis = index % 2
        gt[index] = np.roll(gt[index], shift, axis=axis)
        corner = (2, 58)[index % 2]
        gt[index, corner:corner + 3, corner:corner + 3] = 1.0
    return bank, gt


def run_smoke():
    bank, gt = synthetic_glyphs()
    assert bank.shape == (6, 64, 64) and gt.shape == (6, 64, 64)
    assert set(np.unique(bank).tolist()) <= {0.0, 1.0}
    gt_emb = embed(gt, "directional_multi")
    bank_emb = embed(bank, "directional_multi")
    sim = similarity(gt_emb, bank_emb)
    assert sim.shape == (6, 6) and np.isfinite(sim).all()
    order = ranked_candidates(sim)
    assert np.array_equal(order, np.argsort(-sim, axis=1, kind="stable"))
    assert np.allclose(
        np.sort(sim, axis=1)[:, ::-1], np.take_along_axis(sim, order, axis=1)
    )

    positions = np.arange(6)
    grid_top, matrix, selected, _ = topk_grid_scores(bank, gt, order, positions, positions, (1, 2, 6))
    assert np.allclose(grid_top[6], matrix.max(axis=1), atol=1e-6)
    gt_index = np.repeat(positions, 6)
    pair_scores, _ = ocf_pairs(bank, gt, order.reshape(-1), gt_index, chunk=4)
    assert np.isfinite(pair_scores).all()
    assert pair_scores.min() >= 0 and pair_scores.max() <= 100

    streamed, argmax, early, _, scanned = stream_full_bank(
        bank, gt, order, positions, early_ks=(2, 4), chunk=2
    )
    assert scanned == 36
    assert np.allclose(streamed, matrix.max(axis=1), atol=1e-6)
    assert np.array_equal(argmax, order[positions, matrix.argmax(axis=1)])
    assert streamed.max() < 100.0  # corner blob prevents an exact-match 100 shortcut
    assert np.allclose(early[2], grid_top[2]) and np.allclose(early[4], matrix[:, :4].max(axis=1))
    assert np.allclose(iou(bank, bank), 1.0)
    shifted_iou = iou(bank, gt)
    assert shifted_iou.min() >= 0 and shifted_iou.max() < 1.0
    assert np.all(grid_top[1] <= grid_top[2]) and np.all(grid_top[2] <= grid_top[6])
    probe = mapping_probe()
    assert probe["available"] is False and probe["computed"] is False
    print(
        json.dumps(
            {
                "smoke_test": "ok",
                "pairs": int(len(pair_scores)),
                "ocf_range": [float(pair_scores.min()), float(pair_scores.max())],
                "row_max_match": True,
                "iou_identical": 1.0,
                "iou_shifted_range": [float(shifted_iou.min()), float(shifted_iou.max())],
                "grid_monotonic": True,
                "mapping_note": probe["note"],
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------- real run


def run_real(args):
    started = time.monotonic()
    payload = {
        "protocol": DISCLAIMER,
        "data_scope": "public train pictures and a public system font only; no hidden test data",
        "font": str(FONT_PATH),
        "font_size": FONT_SIZE,
        "variation": VARIATION,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ks": list(KS),
        "pair_chunk": PAIR_CHUNK,
        "ocf_budget_seconds": args.ocf_budget_seconds,
    }

    print("[1/6] rendering GB2312 bank", flush=True)
    bank = render_gb2312_bank(FONT_PATH, font_size=FONT_SIZE, variation=VARIATION)
    bank_fg = foreground(bank)
    payload["bank"] = {
        "n_characters": int(len(bank)),
        "font_rendered_non_empty": int((bank_fg > 0).sum()),
        "empty_glyphs": int((bank_fg == 0).sum()),
        "foreground_pixels": summarize(bank_fg),
    }
    print(f"  bank {bank.shape}, non-empty {int((bank_fg > 0).sum())}", flush=True)

    print("[2/6] loading pictures", flush=True)
    gt = load_gt()
    gt_fg = foreground(gt)
    payload["gt"] = {"n_pictures": int(len(gt)), "foreground_pixels": summarize(gt_fg)}

    heldout = heldout_picture_rows()
    payload["heldout_rows"] = {
        "n": int(len(heldout)),
        "rule": "make_split task1 seed42 character holdout; RNG123 permutation [64:] heldout",
        "indices_head": heldout[:16].tolist(),
    }
    print(f"  heldout pictures: {len(heldout)}", flush=True)

    print("[3/6] embeddings and directional similarity", flush=True)
    t0 = time.monotonic()
    gt_emb = embed(gt, "directional_multi")
    bank_emb = embed(bank, "directional_multi")
    sim = similarity(gt_emb, bank_emb)
    embed_seconds = time.monotonic() - t0
    payload["embedding"] = {
        "metric": "directional_multi",
        "gt_dim": int(gt_emb.shape[1]),
        "bank_dim": int(bank_emb.shape[1]),
        "similarity_shape": list(sim.shape),
        "finite": bool(np.isfinite(sim).all()),
        "seconds": float(embed_seconds),
    }
    np.savez(
        SIMILARITY_NPZ,
        similarity_directional_multi=sim,
        bank_foreground=bank_fg.astype(np.int32),
        gt_foreground=gt_fg.astype(np.int32),
        heldout_rows=heldout,
    )
    print(f"  similarity {sim.shape} finite={np.isfinite(sim).all()} ({embed_seconds:.1f}s)", flush=True)

    order_multi = ranked_candidates(sim)
    ocf_start = time.monotonic()

    print("[4/6] directional2 vs directional_multi (heldout rows, top32)", flush=True)
    gt_emb2 = embed(gt, "directional2")
    bank_emb2 = embed(bank, "directional2")
    sim2_heldout = similarity(gt_emb2[heldout], bank_emb2)
    order2_heldout = ranked_candidates(sim2_heldout)
    grid2, _, _, elapsed2 = topk_grid_scores(
        bank, gt, order2_heldout, np.arange(len(heldout)), heldout, SMALL_KS
    )
    del gt_emb2, bank_emb2, sim2_heldout
    payload["metric_compare_heldout"] = {
        "rows": int(len(heldout)),
        "directional2": {str(k): summarize(grid2[k]) for k in SMALL_KS},
        "seconds": float(elapsed2),
    }

    print("[5/6] all-picture K=1/8/32 grid (directional_multi)", flush=True)
    rows_all = np.arange(len(gt))
    grid_all, _, selected_all, elapsed_all = topk_grid_scores(
        bank, gt, order_multi, rows_all, rows_all, SMALL_KS
    )
    payload["k_grid_all"] = {
        "scope": "all 2,748 pictures",
        "ocf": {str(k): summarize(grid_all[k]) for k in SMALL_KS},
        "seconds": float(elapsed_all),
        "pairs": int(len(gt) * max(SMALL_KS)),
    }
    payload["k_grid_heldout"] = {
        "scope": "211 fixed-protocol held-out pictures",
        "ocf": {str(k): summarize(grid_all[k][heldout]) for k in SMALL_KS},
    }

    top1_index = selected_all[:, 0]
    payload["top1_candidate_diagnostics_all"] = {
        "foreground_pixel_ratio": summarize(bank_fg[top1_index] / np.maximum(gt_fg, 1)),
        "iou": summarize(iou(bank[top1_index], gt)),
        "ocf_mean": float(grid_all[1].mean()),
    }

    print("[6/6] full-bank streamed ceiling", flush=True)
    spent = time.monotonic() - ocf_start
    done_pairs = len(gt) * max(SMALL_KS) + len(heldout) * max(SMALL_KS)
    rate = done_pairs / max(spent, 1e-9)
    remaining = max(args.ocf_budget_seconds - spent, 0.0)
    n_full = int(remaining * rate / len(bank))
    n_full = max(MIN_FULL_ROWS, min(MAX_FULL_ROWS, n_full, len(heldout)))
    rng = np.random.default_rng(args.sample_seed)
    full_rows = np.sort(rng.choice(heldout, size=n_full, replace=False))
    print(
        f"  measured {rate:.0f} pairs/s; budget {args.ocf_budget_seconds}s; "
        f"sampling {n_full} held-out rows for the full 6,763 scan",
        flush=True,
    )
    deadline = time.monotonic() + remaining + 120.0
    best, argmax, early_grid, elapsed_full, scanned = stream_full_bank(
        bank, gt, order_multi[full_rows], full_rows, deadline=deadline
    )
    n_scanned = len(best)
    used_rows = full_rows[:n_scanned]
    payload["full_bank_sample"] = {
        "scope": f"{n_scanned} sampled fixed-protocol held-out pictures x all 6,763 glyphs",
        "sample_seed": int(args.sample_seed),
        "rows": used_rows.tolist(),
        "bank_size": int(len(bank)),
        "pairs_attempted": int(n_scanned * len(bank)),
        "pairs_scored": int(scanned),
        "seconds": float(elapsed_full),
        "pairs_per_second": float(scanned / max(elapsed_full, 1e-9)),
        "ocf": summarize(best),
        "row_best": [float(x) for x in best],
    }
    payload["full_bank_sample"]["ocf"]["ci95"] = mean_ci(best)

    row_position = {int(row): index for index, row in enumerate(rows_all)}
    sample_positions = np.array([row_position[int(row)] for row in used_rows], dtype=np.int64)
    grid_sample = {
        k: (early_grid[k] if k in early_grid else grid_all[k][sample_positions]) for k in KS
    }
    grid_sample[6763] = best
    payload["k_grid_sample"] = {
        "scope": (
            f"same {n_scanned} sampled held-out pictures; K=100/500 are the directional_multi "
            "top-K maxima, K=6763 is the exact full-bank maximum"
        ),
        "ocf": {str(k): summarize(grid_sample[k]) for k in KS},
        "fraction_rows_with_max_at_least_40": float(np.mean(best >= 40.0)),
    }

    ratio = bank_fg[argmax] / np.maximum(gt_fg[used_rows], 1)
    payload["full_bank_sample"]["foreground_pixel_ratio"] = summarize(ratio)
    payload["full_bank_sample"]["iou"] = summarize(iou(bank[argmax], gt[used_rows]))
    payload["full_bank_sample"]["argmax_indices"] = argmax.tolist()

    payload["mapping"] = mapping_probe()
    payload["anchor"] = conclusion(payload)
    payload["totals"] = {
        "seconds": float(time.monotonic() - started),
        "ocf_pairs_scored": int(
            len(gt) * max(SMALL_KS) + len(heldout) * max(SMALL_KS) + scanned
        ),
    }
    RESULTS.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload["anchor"], indent=2), flush=True)
    print(f"wrote {RESULTS}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run synthetic checks only")
    parser.add_argument("--run", action="store_true", help="run the real font/picture diagnostic")
    parser.add_argument("--ocf-budget-seconds", type=int, default=1500)
    parser.add_argument("--sample-seed", type=int, default=56)
    args = parser.parse_args()
    if not args.smoke_test and not args.run:
        parser.error("choose --smoke-test or --run")
    torch.set_grad_enabled(False)
    if args.smoke_test:
        run_smoke()
    else:
        run_real(args)


if __name__ == "__main__":
    main()
