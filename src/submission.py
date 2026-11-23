"""Export model predictions as grayscale PNGs and package them in a submission ZIP."""

import re
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image

from src import GLYPH_SIZE


def _png_names(sample_ids):
    """Validate sample IDs and return their submission filenames."""
    names = []
    for sample_id in sample_ids:
        name = str(sample_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid sample ID: {name!r}. Use letters, digits, underscores, or hyphens.")
        names.append(f"{name}.png")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate sample IDs.")
    return names


def save_images(predictions, sample_ids, folder, image_size):
    """Save native 64-by-64 floating-point grayscale predictions as PNGs."""
    if tuple(image_size) != GLYPH_SIZE:
        raise ValueError("Submission images must be 64 by 64.")
    names = _png_names(sample_ids)
    if (not isinstance(predictions, torch.Tensor)
            or predictions.shape != (len(names), 1, *GLYPH_SIZE)
            or not predictions.is_floating_point()):
        raise ValueError("The model must return one floating-point grayscale image per sample, shaped (B, 1, 64, 64).")
    if not torch.isfinite(predictions).all():
        raise ValueError("Model predictions contain non-finite values.")

    paths = [Path(folder) / name for name in names]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Image already exists: {path}")

    pixels = predictions.detach()[:, 0].float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    for path, image in zip(paths, pixels):
        Image.fromarray(image).save(path)


def pack_submission(folder, archive, sample_ids_by_task):
    """Check image names and write all three task directories at the ZIP root."""
    tasks = ("task1", "task2", "task3")
    if set(sample_ids_by_task) != set(tasks):
        raise ValueError("A submission must contain images for task1, task2, and task3.")
    folder = Path(folder)
    files = []
    for task in tasks:
        names = set(_png_names(sample_ids_by_task[task]))
        task_dir = folder / task
        task_files = sorted(task_dir.iterdir()) if task_dir.is_dir() else []
        if not names or {path.name for path in task_files} != names or any(
            not path.is_file() or path.is_symlink() for path in task_files
        ):
            raise ValueError(f"The images for {task} do not match the expected test sample IDs.")
        for path in task_files:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != GLYPH_SIZE:
                    raise ValueError(f"Expected a 64-by-64 single-channel grayscale PNG: {path}")
                image.verify()
        files.extend(task_files)

    # PNGs are already compressed, so store them without further compression.
    with ZipFile(archive, mode="x") as submission:
        for path in files:
            submission.write(path, arcname=path.relative_to(folder).as_posix())
