from pathlib import Path
import json
import zipfile

import numpy as np
import torch
from PIL import Image

from src.ocf.score import score

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "demo_e1"
OUTPUT = ROOT / "outputs" / "demo_e1"
ARCHIVE = ROOT / "outputs" / "demo_e1.zip"
DATA = ROOT / "data"

expected_checkpoints = {
    "mice1/session1.pt": 670,
    "mice1/across_sessions.pt": 670,
    "mice2/session1.pt": 390,
    "mice2/across_sessions.pt": 390,
}
for relative, expected_inputs in expected_checkpoints.items():
    path = RUNS / relative
    assert path.is_file(), path
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["inputs"] == expected_inputs
    assert checkpoint["epoch"] == 1
    assert checkpoint["state_dict"]

expected_by_task = {}
with (DATA / "catalog.json").open(encoding="utf-8") as handle:
    catalog = json.load(handle)
with np.load(DATA / "train" / "pictures.npz", allow_pickle=False) as pictures:
    reference = pictures["images"][0] >= 128

for block in catalog["blocks"]:
    if block["split"] != "test":
        continue
    task = block["task"]
    with np.load(DATA / block["path"], allow_pickle=False) as session:
        expected_by_task.setdefault(task, set()).update(session["sample_ids"].tolist())

expected_zip_entries = set()
for task in ("task1", "task2", "task3"):
    task_dir = OUTPUT / task
    actual = {path.name for path in task_dir.glob("*.png")}
    expected = {f"{sample_id}.png" for sample_id in expected_by_task[task]}
    assert actual == expected, (task, actual ^ expected)
    assert len(actual) == 2
    for filename in sorted(actual):
        with Image.open(task_dir / filename) as image:
            assert image.size == (64, 64)
            assert image.mode == "L"
            image.verify()
    expected_zip_entries.update(f"{task}/{filename}" for filename in actual)

with zipfile.ZipFile(ARCHIVE) as archive:
    actual_entries = {name for name in archive.namelist() if not name.endswith("/")}
    assert actual_entries == expected_zip_entries
    assert archive.testzip() is None

scores = score(
    np.stack([reference.astype(np.float32), np.zeros_like(reference, dtype=np.float32)]),
    np.stack([reference, reference]),
    device="cpu",
)
assert np.allclose(scores, [100.0, 0.0]), scores

print(json.dumps({
    "checkpoints": len(expected_checkpoints),
    "png_files": len(expected_zip_entries),
    "zip_entries": len(expected_zip_entries),
    "ocf_identical": float(scores[0]),
    "ocf_blank": float(scores[1]),
    "status": "PASS",
}, indent=2))
