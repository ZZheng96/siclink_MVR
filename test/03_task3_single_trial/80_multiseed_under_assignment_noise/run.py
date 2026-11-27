"""Re-score the cached Task 3 multi-seed arms under assignment-noise-neutral OCF."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import evaluate_final as ef
from src.dataloader import load_images
from src.encoding import response_scores
from src.evaluation import score_predictions
from src.features import load_feature_blocks
from src.final import DATA, apply_session_normalizer, checkpoint_path, row_zscore
from src.neural import load_classifier, predict_logits
from src.retrieval import softmax_dct_similarity, unique_assignment
from src.ridge import dct_encode

ARMS = {"base": [42], "seed3": [42, 202, 777]}
FOLD_MODELS = ROOT / "test/03_task3_single_trial/75_task3_multiseed/artifacts"
DRAWS = 6
RECORDED_HELDOUT = 16.759832


def arm_scores(arm, seeds, blocks, split, images, device="cuda"):
    models = FOLD_MODELS / arm / "models"
    coefficients = dct_encode(images, 16)
    centered = coefficients - coefficients.mean(axis=0, keepdims=True)
    out = {}
    for block in blocks:
        part = split.blocks[block.key]
        trials = part.validation_trials
        mouse = block.mouse
        with np.load(models / "task3" / mouse / f"{block.session}_normalizer.npz") as data:
            normalizer = (data["mean"], data["scale"])
        raw = block.features[trials]
        counts = apply_session_normalizer(raw, normalizer)
        logits = []
        for seed in seeds:
            model = load_classifier(checkpoint_path(models, "task3", mouse, seed), device)
            logits.append(predict_logits(model, counts, device=device))
            del model
        mlp = np.mean(np.stack(logits), axis=0, dtype=np.float32)
        prototypes = np.load(models / "task3" / mouse / "encoding_prototypes.npy",
                             allow_pickle=False)
        encoding = response_scores(np.log1p(raw), prototypes, metric="mse", device=device)
        continuous = softmax_dct_similarity(mlp, centered, device=device)
        out[block.key] = ((row_zscore(mlp) + row_zscore(encoding) + row_zscore(continuous)) / 3.0,
                          block.trial_targets[trials])
    return out


def ocf_block(scores, labels, gt, heldout, weight=None):
    sc = scores if weight is None else scores * weight
    assigned = unique_assignment(sc)
    mask = np.isin(labels, heldout)
    return float(score_predictions(gt[assigned[mask]], gt[labels[mask]],
                                   device="cuda", batch_size=16).mean())


def main():
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, "train", bin_ms=50)
    split, partitions, protocol = ef.public_protocol("task3", blocks, len(images))
    heldout = partitions["heldout"]
    result = {"experiment": "80_multiseed_under_assignment_noise",
              "protocol": {k: protocol[k] for k in ("rule", "picture_seed", "split_seed")},
              "draws": DRAWS, "arms": {}, "paired": {}}
    scores = {arm: arm_scores(arm, seeds, blocks, split, images)
              for arm, seeds in ARMS.items()}
    rng = np.random.default_rng(20261128)
    weights = {key: rng.uniform(0.6, 1.6, size=(len(scores["base"][key][0]), 1)).astype(np.float32)
               for key in scores["base"]}
    for arm in ARMS:
        canonical, neutral, per_session = [], [], {}
        for key, (sc, labels) in scores[arm].items():
            values = [ocf_block(sc, labels, images, heldout, weights[key]) for _ in range(DRAWS)]
            neutral.append(float(np.mean(values)))
            canonical.append(ocf_block(sc, labels, images, heldout))
            per_session[key] = float(np.mean(values))
        result["arms"][arm] = {"canonical_mean": float(np.mean(canonical)),
                               "neutral_mean": float(np.mean(neutral)),
                               "per_session_neutral": per_session}
        print(arm, result["arms"][arm], flush=True)
    paired = {key: result["arms"]["seed3"]["per_session_neutral"][key]
              - result["arms"]["base"]["per_session_neutral"][key]
              for key in result["arms"]["base"]["per_session_neutral"]}
    result["paired"] = {"per_session": paired,
                        "mean": float(np.mean(list(paired.values()))),
                        "positive_blocks": int(sum(v > 0 for v in paired.values())),
                        "canonical_delta": result["arms"]["seed3"]["canonical_mean"]
                        - result["arms"]["base"]["canonical_mean"],
                        "neutral_delta": result["arms"]["seed3"]["neutral_mean"]
                        - result["arms"]["base"]["neutral_mean"]}
    base_heldout = result["arms"]["base"]["canonical_mean"]
    result["pass_rule"] = {"base_reproduces_recorded": bool(
        abs(base_heldout - RECORDED_HELDOUT) < 1e-3),
        "retain": bool(result["paired"]["neutral_delta"] >= 0.30
                       and result["paired"]["positive_blocks"] >= 6)}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(json.dumps(result["paired"], indent=2), flush=True)
    print(result["pass_rule"], flush=True)


if __name__ == "__main__":
    main()
