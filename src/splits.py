"""Fixed, auditable validation protocols for the three competition tasks."""

from dataclasses import dataclass

import numpy as np

from src.features import trials_for_samples


@dataclass
class BlockSplit:
    train_samples: np.ndarray
    validation_samples: np.ndarray
    train_trials: np.ndarray
    validation_trials: np.ndarray


@dataclass
class ProtocolSplit:
    task: str
    blocks: dict
    seed: int
    validation_pictures: np.ndarray
    validation_session: str | None = None

    def metadata(self):
        return {
            "task": self.task,
            "seed": self.seed,
            "validation_session": self.validation_session,
            "validation_pictures": self.validation_pictures.tolist(),
            "blocks": {
                key: {
                    "train_samples": len(split.train_samples),
                    "validation_samples": len(split.validation_samples),
                    "train_trials": len(split.train_trials),
                    "validation_trials": len(split.validation_trials),
                }
                for key, split in self.blocks.items()
            },
        }


def _block_split(block, train_samples, validation_samples):
    train_samples = np.asarray(train_samples, dtype=np.int64)
    validation_samples = np.asarray(validation_samples, dtype=np.int64)
    return BlockSplit(
        train_samples=train_samples,
        validation_samples=validation_samples,
        train_trials=trials_for_samples(block.group_ptr, train_samples),
        validation_trials=trials_for_samples(block.group_ptr, validation_samples),
    )


def make_split(blocks, task, *, seed=42, validation_fraction=0.1, validation_session="session4"):
    """Create a character-held-out, day-held-out, or single-trial split."""
    if task not in ("task1", "task2", "task3"):
        raise ValueError("Unknown validation task.")
    if not blocks or any(block.targets is None for block in blocks):
        raise ValueError("Validation splits require training blocks with targets.")
    blocks = sorted(blocks, key=lambda block: (block.mouse, block.session))
    splits = {}
    validation_pictures = np.empty(0, dtype=np.int64)

    if task == "task1":
        pictures = np.unique(np.concatenate([block.targets for block in blocks]))
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        shuffled = np.random.default_rng(seed).permutation(pictures)
        count = max(1, round(validation_fraction * len(pictures)))
        validation_pictures = np.sort(shuffled[:count])
        held_out = set(validation_pictures.tolist())
        for block in blocks:
            validation = np.flatnonzero(np.isin(block.targets, validation_pictures))
            training = np.flatnonzero(~np.isin(block.targets, validation_pictures))
            if not len(training) or not len(validation):
                raise ValueError(f"Empty character split for {block.key}.")
            assert not set(block.targets[training]) & held_out
            assert set(block.targets[validation]) <= held_out
            splits[block.key] = _block_split(block, training, validation)

    elif task == "task2":
        for mouse in {block.mouse for block in blocks}:
            sessions = {block.session for block in blocks if block.mouse == mouse}
            if validation_session not in sessions:
                raise ValueError(f"{mouse} has no {validation_session} block.")
        for block in blocks:
            samples = np.arange(block.n_samples, dtype=np.int64)
            empty = np.empty(0, dtype=np.int64)
            is_validation = block.session == validation_session
            splits[block.key] = _block_split(
                block, empty if is_validation else samples, samples if is_validation else empty
            )

    else:
        for block_index, block in enumerate(blocks):
            rng = np.random.default_rng(seed + block_index)
            repeats = np.diff(block.group_ptr)
            if np.all(repeats == 1):
                # Formal training blocks store each character's four repeats as independent rows.
                order = np.argsort(block.targets, kind="stable")
                _, counts = np.unique(block.targets, return_counts=True)
                selected = []
                start = 0
                for count in counts:
                    if count < 2:
                        raise ValueError("Task 3 validation requires at least two trials per character.")
                    selected.append(order[start + rng.integers(count)])
                    start += count
                validation_trials = np.asarray(selected, dtype=np.int64)
                sample_count = len(counts)
            else:
                if np.any(repeats < 2):
                    raise ValueError("Task 3 validation requires at least two trials per sample.")
                validation_trials = block.group_ptr[:-1] + np.asarray([
                    rng.integers(repeat) for repeat in repeats
                ])
                sample_count = block.n_samples
            mask = np.ones(block.n_trials, dtype=bool)
            mask[validation_trials] = False
            samples = np.arange(sample_count, dtype=np.int64)
            split = BlockSplit(samples, samples, np.flatnonzero(mask), validation_trials)
            assert not np.intersect1d(split.train_trials, split.validation_trials).size
            assert len(np.unique(block.trial_targets[validation_trials])) == sample_count
            splits[block.key] = split

    return ProtocolSplit(
        task=task,
        blocks=splits,
        seed=seed,
        validation_pictures=validation_pictures,
        validation_session=validation_session if task == "task2" else None,
    )
