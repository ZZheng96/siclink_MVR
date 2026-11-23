"""Verify a final Kaggle ZIP without uploading it."""

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

from PIL import Image

TASK_COUNTS = {"task1": 2720, "task2": 5496, "task3": 21984}


def verify_archive(path):
    expected = {
        f"test_{task}_pic{index}"
        for task, count in TASK_COUNTS.items()
        for index in range(count)
    }
    seen = set()
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["submission.csv"]:
            raise ValueError(f"Expected only submission.csv, found {names}.")
        with archive.open("submission.csv") as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            if next(reader, None) != ["sample_id", "prediction"]:
                raise ValueError("CSV header must be sample_id,prediction.")
            for row_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ValueError(f"Row {row_number}: expected two columns.")
                sample_id, encoded = row
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample ID: {sample_id}")
                if sample_id not in expected:
                    raise ValueError(f"Unexpected sample ID: {sample_id}")
                payload = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "PNG" or image.mode != "L" or image.size != (64, 64):
                        raise ValueError(f"{sample_id}: invalid PNG format, mode, or size.")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError(f"{sample_id}: expected one frame.")
                    image.load()
                    extrema = image.getextrema()
                    if extrema[0] not in (0, 255) or extrema[1] not in (0, 255):
                        raise ValueError(f"{sample_id}: prediction is not binary.")
                    colors = image.getcolors(maxcolors=256)
                    if colors is None or any(value not in (0, 255) for _, value in colors):
                        raise ValueError(f"{sample_id}: prediction contains gray pixels.")
                seen.add(sample_id)
                digest.update(sample_id.encode("ascii"))
                digest.update(payload)
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} sample IDs; first: {sorted(missing)[:3]}")
    return {"samples": len(seen), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = verify_archive(args.archive)
    print(f"PASS: {result['samples']} binary PNGs; content SHA256={result['sha256']}")


if __name__ == "__main__":
    main()
