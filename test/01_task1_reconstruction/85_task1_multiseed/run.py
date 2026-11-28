"""Task 1 multi-seed paired evaluation under the frozen public protocol."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import evaluate_final as ef
from src.dataloader import load_images
from src.features import load_feature_blocks
from src.final import DATA, predict_task, train_task

HASH = "9447744f3e99a3d44fc30afc4b11eafac2313bccd21d736fa2fb6de30f62602d"
ARMS = {"base": [42], "seed3": [42, 202, 777]}
RECORDED = {"tune": 16.81193, "heldout": 18.43706}


def per_mouse(groups):
    out = {}
    for block, row in groups.items():
        out.setdefault(block.split("/")[0], []).append(row["ocf_mean"])
    return {m: float(np.mean(v)) for m, v in out.items()}


def main():
    digest = hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("preregistration changed")
    base_config = json.loads((ROOT / "src/final_config.json").read_text(encoding="utf-8"))
    blocks = load_feature_blocks(DATA, "train", bin_ms=base_config["task1"]["bin_ms"])
    images = (load_images(DATA) >= 128).astype(np.float32)
    split, partitions, protocol = ef.public_protocol("task1", blocks, len(images))
    result = {"experiment": "85_task1_multiseed", "preregistration_sha256": HASH,
              "protocol": protocol, "arms": {}}
    for arm, seeds in ARMS.items():
        config = copy.deepcopy(base_config)
        config["task1"]["seeds"] = seeds
        config["name"] = f"v5_task1_multiseed_{arm}"
        models = OUT / "artifacts" / arm / "models"
        train_task(config, "task1", models, images=images, blocks=blocks, split=split,
                   device="cuda")
        prediction = predict_task(config, "task1", models, images=images, blocks=blocks,
                                  split=split, device="cuda")
        row = {"seeds": seeds}
        for part in ("tune", "heldout"):
            mask = np.isin(prediction["targets"], partitions[part])
            metrics = ef.official_evaluate(prediction["predictions"][mask],
                                           images[prediction["targets"][mask]],
                                           prediction["groups"][mask], "cuda", 16)
            row[part] = {"ocf": metrics["ocf_mean"], "n": metrics["n"],
                         "per_mouse": per_mouse(metrics["groups"]),
                         "blocks": {b: v["ocf_mean"] for b, v in metrics["groups"].items()}}
            print(arm, part, row[part]["ocf"], row[part]["per_mouse"], flush=True)
        result["arms"][arm] = row
    check_base(result)
    result["pass_rule"] = evaluate(result)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(result["pass_rule"], flush=True)


def check_base(result):
    row = result["arms"]["base"]
    for part in ("tune", "heldout"):
        if abs(row[part]["ocf"] - RECORDED[part]) > 1e-4:
            raise SystemExit(f"base {part} {row[part]['ocf']} != recorded {RECORDED[part]}")


def evaluate(result):
    base, arm = result["arms"]["base"], result["arms"]["seed3"]
    mice = ("mice1", "mice2")
    held_up = all(arm["heldout"]["per_mouse"][m] > base["heldout"]["per_mouse"][m] for m in mice)
    tune_ok = arm["tune"]["ocf"] >= base["tune"]["ocf"] - 0.02
    blocks = sum(arm["heldout"]["blocks"][b] >= base["heldout"]["blocks"][b]
                 for b in base["heldout"]["blocks"])
    return {"heldout_up_both_mice": bool(held_up), "tune_within_tolerance": bool(tune_ok),
            "heldout_blocks_non_negative": int(blocks),
            "advance": bool(held_up and tune_ok and blocks >= 5)}


if __name__ == "__main__":
    main()
