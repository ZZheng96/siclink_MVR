"""Losslessly package the official 64×64 PNG submission as Kaggle CSV."""

import argparse
import base64
import csv
import io
from pathlib import Path
import zipfile

from PIL import Image

COUNTS = {"task1": 2720, "task2": 5496, "task3": 21984}


def convert(source_path, output_path):
    """Validate all official IDs and images, then write submission.csv in a ZIP."""
    rows = [(task, f"test_{task}_pic{i}") for task, n in COUNTS.items() for i in range(n)]
    expected = {f"{task}/{sample_id}.png" for task, sample_id in rows}
    with zipfile.ZipFile(source_path) as source:
        entries = [item for item in source.infolist() if not item.is_dir()]
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Expected exactly 30,200 PNG files under task1/, task2/, and task3/.")
        if any(item.file_size > 65536 or item.flag_bits & 1 for item in entries):
            raise ValueError("Oversized or encrypted image.")
        with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED) as target:
            with target.open("submission.csv", "w") as raw, io.TextIOWrapper(raw, newline="") as dest:
                writer = csv.writer(dest)
                writer.writerow(["sample_id", "prediction"])
                for task, sample_id in rows:
                    with Image.open(io.BytesIO(source.read(f"{task}/{sample_id}.png"))) as image:
                        if (image.format != "PNG" or image.mode != "L" or image.size != (64, 64)
                                or getattr(image, "n_frames", 1) != 1):
                            raise ValueError(f"{sample_id}: expected a single 64×64 grayscale PNG.")
                        image.load()
                        clean = Image.frombytes("L", image.size, image.tobytes())
                        buffer = io.BytesIO()
                        clean.save(buffer, format="PNG")
                    writer.writerow([sample_id, base64.b64encode(buffer.getvalue()).decode("ascii")])
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png_submission", type=Path)
    parser.add_argument("kaggle_submission", type=Path)
    args = parser.parse_args()
    count = convert(args.png_submission, args.kaggle_submission)
    print(f"Packaged {count} unchanged images into {args.kaggle_submission}")


if __name__ == "__main__":
    main()
