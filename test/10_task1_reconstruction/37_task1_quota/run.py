"""Fixed baseline24 + raw delay50 quota experiment on public Task1 data."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "temporal_candidates", ROOT / "test/10_task1_reconstruction/33_task1_temporal_candidates/run.py"
)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
BASELINE = "baseline"
QUOTA = "baseline24_delay50_raw8"
ARTIFACTS = HERE / "artifacts"
SELECTION_PATH = ARTIFACTS / "selection.json"
RESULTS_PATH = HERE / "results.json"


def quota_records(baseline_parts, delay_parts, device):
    baseline = source.parts_to_records(baseline_parts, device=device)
    delay = source.parts_to_records(delay_parts, device=device)
    for key in ("targets", "pictures", "groups", "mice"):
        assert np.array_equal(baseline[key], delay[key]), f"Unaligned {key}"
    quota = {key: value.copy() for key, value in baseline.items()}
    for index, (base_top, delay_top) in enumerate(zip(baseline["top32"], delay["top32"])):
        selected = base_top[:24].tolist()
        for candidate in delay_top:
            if int(candidate) not in selected:
                selected.append(int(candidate))
            if len(selected) == 32:
                break
        assert len(selected) == 32
        quota["top32"][index] = selected
    assert np.array_equal(quota["top32"][:, :24], baseline["top32"][:, :24])
    assert np.array_equal(quota["top32"][:, 0], baseline["top32"][:, 0]), "Top1 changed"
    for records in (baseline, quota):
        assert records["top32"].shape[1] == 32
        assert all(len(set(row.tolist())) == 32 for row in records["top32"])
    return {BASELINE: baseline, QUOTA: quota}


def evaluate(protocol, pictures, bank_embedding, bank, images, cache, device, name):
    baseline_parts = source.predict_parts(
        protocol, BASELINE, pictures, {}, bank_embedding, device
    )
    delay_parts = source.predict_parts(
        protocol, "delay50", pictures, {mouse: None for mouse in protocol["training"]},
        bank_embedding, device,
    )
    records = quota_records(baseline_parts, delay_parts, device)
    np.savez_compressed(
        ARTIFACTS / f"{name}_candidates.npz",
        targets=records[BASELINE]["targets"],
        pictures=records[BASELINE]["pictures"],
        groups=records[BASELINE]["groups"],
        mice=records[BASELINE]["mice"],
        baseline_top32=records[BASELINE]["top32"],
        quota_top32=records[QUOTA]["top32"],
    )
    source.fill_ocf_cache(records, images, bank, cache, device)
    metrics = {variant: source.metric_summary(value, cache) for variant, value in records.items()}
    assert metrics[BASELINE]["overall"]["top1_ocf"] == metrics[QUOTA]["overall"]["top1_ocf"]
    expected = source.BASELINE_TUNE_OCF if name == "tune" else source.BASELINE_HELDOUT_OCF
    error = abs(metrics[BASELINE]["overall"]["top1_ocf"] - expected)
    assert error <= source.BASELINE_ATOL, f"Baseline reproduction failed: {error}"
    for variant, summary in metrics.items():
        print(f"{name.upper()} {variant} {summary['overall']}", flush=True)
    return metrics


def main():
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Official CUDA OCF is required")
    checkpoints = [
        source.checkpoint_path(variant, mouse)
        for variant in (BASELINE, "delay50") for mouse in ("mice1", "mice2")
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    images = (source.load_images(source.DATA) >= 128).astype(np.float32)
    assert images.shape == (2748, 64, 64)
    bank = source.render_gb2312_bank(source.FONT_PATH, font_size=52, variation="Regular")
    assert bank.shape == (6763, 64, 64)
    blocks = source.load_feature_blocks(source.DATA, "train", bin_ms=25)
    assert sorted(block.key for block in blocks) == [
        f"{mouse}/session{session}" for mouse in ("mice1", "mice2") for session in range(1, 5)
    ]
    protocol = source.prepare_protocol(blocks)
    del blocks
    for training in protocol["training"].values():
        for key in ("train_x", "train_targets", "validation_x", "validation_targets"):
            del training[key]
    source.release_memory()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bank_embedding = source.make_bank_embedding(bank, device)
    cache = {}
    tune = evaluate(
        protocol, protocol["tune_pictures"], bank_embedding, bank, images, cache, device, "tune"
    )
    delta = tune[QUOTA]["overall"]["oracle_top32_ocf"] - tune[BASELINE]["overall"]["oracle_top32_ocf"]
    improved = delta > 0.0
    selection = {
        "experiment": "37_task1_quota",
        "data_scope": "public train only",
        "source_experiment": "test/10_task1_reconstruction/33_task1_temporal_candidates/run.py",
        "checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
        "protocol": {
            "split_seed": source.SEED,
            "picture_seed": source.PICTURE_SEED,
            "validation_picture_count": protocol["validation_picture_count"],
            "model_validation_pictures": protocol["model_validation_pictures"].tolist(),
            "tune_pictures": protocol["tune_pictures"].tolist(),
            "heldout_pictures": protocol["heldout_pictures"].tolist(),
            "selection_rule": "Evaluate the single fixed quota on tune16. Lock quota and evaluate heldout211 only if oracle32 is strictly greater than baseline top32, with no tolerance or secondary criterion.",
            "heldout_used_for_selection": False,
            "historical_heldout_reuse": True,
            "historical_heldout_disclosure": "The fixed heldout211 was evaluated in earlier experiments and is historically reused. It is not a pristine external confirmation.",
        },
        "config": {
            "quota": "Preserve baseline top24 in order, append the first eight distinct bank glyph IDs from the same sample's raw single-model delay50 ranking, skipping retained IDs.",
            "delay50": "Existing all-unit 50-250ms eight-25ms-bin checkpoint, directional2 cosine ranking without score fusion",
            "raw_ranking_prefix": "Raw top32 suffices because at most 24 candidates are excluded.",
            "baseline": "Existing baseline checkpoint and original full-275-block inference, directional2 top32",
            "scoring": "Reuse source fill_ocf_cache and metric_summary with official CUDA OCF",
            "top_k": 32,
        },
        "assertions": {"top1_unchanged": True, "baseline_top24_preserved": True, "unique_32_candidates_per_row": True},
        "tune": tune,
        "tune_oracle32_delta": delta,
        "strict_improvement": improved,
        "selected": QUOTA if improved else BASELINE,
        "quota_locked": improved,
        "heldout_authorized_by_tune": improved,
    }
    source.write_json(SELECTION_PATH, selection)
    print(f"SELECTION PERSISTED improved={improved} delta={delta}", flush=True)
    result = dict(selection)
    result["heldout_evaluated"] = False
    if improved:
        heldout = evaluate(
            protocol, protocol["heldout_pictures"], bank_embedding, bank, images, cache, device, "heldout"
        )
        result["heldout"] = heldout
        result["heldout_evaluated"] = True
        result["heldout_quota_minus_baseline"] = {
            metric: heldout[QUOTA]["overall"][metric] - heldout[BASELINE]["overall"][metric]
            for metric in ("top1_ocf", "oracle_top32_ocf")
        }
    else:
        result["heldout_skip_reason"] = "Quota did not strictly improve tune16 oracle32. No heldout retrieval or scoring performed."
    result["environment"] = {
        "python": sys.executable, "torch": str(torch.__version__), "numpy": np.__version__,
        "device": device, "gpu": torch.cuda.get_device_name(0),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["artifacts"] = {
        "selection": str(SELECTION_PATH.relative_to(ROOT)),
        "tune_candidates": str((ARTIFACTS / "tune_candidates.npz").relative_to(ROOT)),
    }
    if improved:
        result["artifacts"]["heldout_candidates"] = str((ARTIFACTS / "heldout_candidates.npz").relative_to(ROOT))
    source.write_json(RESULTS_PATH, result)
    print(f"FINAL {RESULTS_PATH} elapsed_seconds={result['elapsed_seconds']}", flush=True)


if __name__ == "__main__":
    main()
