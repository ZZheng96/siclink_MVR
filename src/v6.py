"""Frozen experiment-53 Task2/3 classifier. Use separate model roots for fixed and full fits."""

from pathlib import Path

import numpy as np
import torch

from src.features import aggregate_by_target
from src.neural import SpikeClassifier, load_classifier, predict_logits
from src.retrieval import unique_assignment

VIEWS = (10, 25, 50)
EPOCHS = {"task2": {10: 7, 25: 5, 50: 7}, "task3": {10: 7, 25: 12, 50: 9}}
DROPOUT = {"mice1": 0.2, "mice2": 0.35}


def _directory(models, task, mouse, view):
    return Path(models) / "v6" / task / mouse / f"{view}ms"


def _check_views(task, views):
    if task not in EPOCHS or set(views) != set(VIEWS):
        raise ValueError("Expected task2 or task3 and exactly the 10/25/50 ms views.")


def _aligned(blocks):
    reference = blocks[25]
    for view, block in blocks.items():
        if (block.bin_ms != view or block.key != reference.key or block.split != reference.split
                or block.task != reference.task or block.n_units != reference.n_units
                or block.n_features != block.n_units * (250 // view)
                or not np.array_equal(block.sample_ids, reference.sample_ids)
                or not np.array_equal(block.group_ptr, reference.group_ptr)
                or block.n_trials != block.group_ptr[-1]):
            raise ValueError(f"Misaligned {view} ms view for {reference.key}.")


def _indices(values, size):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in 'iu' or np.any(values < 0)
            or np.any(values >= size) or len(np.unique(values)) != len(values)):
        raise ValueError("Expected distinct in-range trial indices.")
    return values


def _fit_norm(raw):
    x = np.log1p(np.maximum(raw, 0))
    return (x.mean(axis=0, dtype=np.float64).astype(np.float32),
            np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5))


def _apply_norm(raw, norm):
    return ((np.log1p(np.maximum(raw, 0)) - norm[0]) / norm[1]).astype(np.float32, copy=False)


def _fit(x, y, mouse, view, task, n_classes, device, directory):
    torch.manual_seed(42)
    model = SpikeClassifier(x.shape[1], n_classes, hidden=512, dropout=DROPOUT[mouse], apply_log=False)
    model.set_normalization(x)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3, fused=device.type == 'cuda')
    epochs = EPOCHS[task][view]
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(42 + epoch).permutation(len(y))
        total = 0.0
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            inputs = torch.from_numpy(x[idx]).to(device)
            labels = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = torch.nn.functional.cross_entropy(model(inputs).float(), labels, label_smoothing=0.05)
            value = loss.item()
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite loss for {task}/{mouse}/{view}ms.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += value * len(idx)
        print(f'{task}/{mouse}/{view}ms epoch={epoch}/{epochs} loss={total / len(y):.6f}', flush=True)
    path = directory / 'classifier.pt'
    tmp = path.with_suffix('.tmp')
    torch.save(dict(n_features=x.shape[1], n_classes=n_classes, hidden=512, dropout=DROPOUT[mouse],
                    apply_log=False, epoch=epochs, state_dict=model.state_dict()), tmp)
    tmp.replace(path)
    return dict(checkpoint=str(path), samples=len(y), epochs=epochs, seed=42, final_train_loss=total / len(y))


def _train_mouse(task, mouse, blocks, split, model_root, view, n_classes, device):
    directory = _directory(model_root, task, mouse, view)
    directory.mkdir(parents=True, exist_ok=True)
    xs, ys, counts = [], [], {}
    for block in blocks:
        if block.split != 'train' or block.targets is None:
            raise ValueError('Training requires official labelled train blocks.')
        if split is None:
            idx = np.arange(block.n_trials)
        else:
            part = split.blocks[block.key]
            idx = _indices(part.train_trials, block.n_trials)
            if np.intersect1d(idx, part.validation_trials).size:
                raise ValueError(f'Train/validation overlap for {block.key}.')
        counts[block.key] = len(idx)
        if not len(idx):
            continue
        raw = np.asarray(block.features[idx], dtype=np.float32)
        labels = np.asarray(block.targets[block.trial_sample_indices[idx]], dtype=np.int64)
        norm = _fit_norm(raw)
        xs.append(_apply_norm(raw, norm))
        ys.append(labels)
        if task == 'task2':
            targets, means = aggregate_by_target(raw, labels)
            xs.append(_apply_norm(means, norm))
            ys.append(targets)
        else:
            np.savez(directory / f'{block.session}_normalizer.npz', mean=norm[0], scale=norm[1])
        del raw
    if not xs:
        raise ValueError(f'No permitted training trials for {task}/{mouse}.')
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    return {**_fit(x, y, mouse, view, task, n_classes, device, directory), 'train_trials': counts}


def train_task(task, models, blocks_by_view, *, split=None, device=None, n_classes=2748):
    """Train one model per mouse and view; split=None uses all official train rows.

    blocks_by_view maps integer bin widths to lists of aligned train FeatureBlocks.
    For fixed evaluation pass the ProtocolSplit from public_protocol / make_split.
    Returns a JSON-compatible training summary; artifacts are rooted under models/v6.
    """
    _check_views(task, blocks_by_view)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    groups = {v: {b.key: b for b in blocks_by_view[v]} for v in VIEWS}
    keys = sorted(groups[25])
    if (not keys or any(set(groups[v]) != set(keys) or len(groups[v]) != len(blocks_by_view[v]) for v in VIEWS)
            or (split is not None and (split.task != task or not set(keys) <= split.blocks.keys()))):
        raise ValueError('View or split block keys do not match.')
    for key in keys:
        _aligned({v: groups[v][key] for v in VIEWS})
    summary = {'task': task, 'models': {}}
    for v in VIEWS:
        for mouse in sorted({b.mouse for b in groups[v].values()}):
            blocks = [groups[v][k] for k in keys if groups[v][k].mouse == mouse]
            summary['models'][f'{task}/{mouse}/{v}ms'] = _train_mouse(
                task, mouse, blocks, split, models, v, n_classes, device)
    return summary


def predict_block(task, models, block_by_view, *, trial_indices=None, group_ids=None, device=None):
    """Return full-gallery scores and uniquely assigned indices for one entire block.

    Supply {10: block, 25: block, 50: block}. Test blocks use their official sample
    grouping and order. For fixed evaluation, pass the full validation trial indices;
    Task2 also needs one opaque dense repeat-group ID per selected trial (rows are
    returned in ascending group order). Task3 preserves trial order. No target access.
    Select tune/heldout rows only *after* this call. Convert indices via images[indices].
    """
    _check_views(task, block_by_view)
    _aligned(block_by_view)
    block = block_by_view[25]
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if trial_indices is None:
        if group_ids is not None or block.split != 'test' or block.task != task:
            raise ValueError('Production prediction requires the official test block for this task.')
        if task == 'task3' and not np.all(np.diff(block.group_ptr) == 1):
            raise ValueError('Task3 requires one trial per sample.')
    else:
        trial_indices = _indices(trial_indices, block.n_trials)
        if block.split != 'train' or not len(trial_indices) or (task == 'task2') != (group_ids is not None):
            raise ValueError('Fixed prediction requires validation rows and repeat groups for Task2 only.')
        if group_ids is not None:
            group_ids = np.asarray(group_ids)
            if (group_ids.shape != trial_indices.shape or group_ids.dtype.kind not in 'iu'
                    or not np.array_equal(np.unique(group_ids), np.arange(len(np.unique(group_ids))))):
                raise ValueError('Expected one dense integer repeat-group ID per selected trial.')
    scores = None
    for v in VIEWS:
        b = block_by_view[v]
        directory = _directory(models, task, b.mouse, v)
        raw = np.asarray(b.features if trial_indices is None else b.features[trial_indices], dtype=np.float32)
        if task == 'task2':
            norm = _fit_norm(raw)  # All unlabeled single trials of this target session.
            if trial_indices is None:
                # Use float64 means as in experiment 53's reduceat/count division.
                raw = np.add.reduceat(raw, b.group_ptr[:-1], axis=0) / np.diff(b.group_ptr)[:, None]
            else:
                _, raw = aggregate_by_target(raw, group_ids)
        else:
            with np.load(directory / f'{b.session}_normalizer.npz', allow_pickle=False) as saved:
                norm = saved['mean'], saved['scale']
        model = load_classifier(directory / 'classifier.pt', device=device)
        logits = predict_logits(model, _apply_norm(raw, norm), batch_size=256, device=device).astype(np.float64)
        logits -= logits.mean(axis=1, keepdims=True)
        logits /= np.maximum(logits.std(axis=1, keepdims=True), 1e-6)
        if scores is None:
            scores = logits
        else:
            if scores.shape != logits.shape:
                raise ValueError('Incompatible view logit matrices.')
            scores += logits
        del model, raw, logits
    scores /= len(VIEWS)
    if len(scores) > scores.shape[1]:
        raise ValueError('Unique assignment requires at least as many classes as block rows.')
    return {'scores': scores, 'indices': unique_assignment(scores)}
