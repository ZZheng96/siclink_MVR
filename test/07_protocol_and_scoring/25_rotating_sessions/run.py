"""Frozen Task2 comparison on three new development folds; no competition test data."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / '24_forward_cnn'
spec = importlib.util.spec_from_file_location('exp24', SOURCE / 'run.py')
exp24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp24)
e = exp24.e
DEVICE = exp24.DEVICE
DAYS = ('session1', 'session2', 'session3', 'session4')
ARMS = ('ridge_mse', 'cnn_poisson')
METRICS = ('ocf', 'top1', 'recall1', 'recall32')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def fit_mlp(x, y, seed, epochs, n_classes):
    # Same training objective as src.neural.fit_classifier, without its
    # validation loader, checkpoint selection or model-file side effects.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = e.SpikeClassifier(x.shape[1], n_classes, hidden=512, dropout=.2, apply_log=False)
    model.set_normalization(x)
    model.to(DEVICE).train()
    torch.manual_seed(seed)  # fit_classifier reseeds after model construction
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=256, shuffle=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001, fused=DEVICE == 'cuda')
    for _ in range(epochs):
        for features, targets in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
                logits = model(features.to(DEVICE))
                loss = torch.nn.functional.cross_entropy(logits.float(), targets.to(DEVICE), label_smoothing=.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
    return model


def measure(images, block, targets, fused, subsets):
    assigned = e.unique_assignment(fused)
    row = dict(block=block.key, targets=targets, assignments={'arm': assigned})
    data = dict(rows=[row], images=images, device=DEVICE)
    candidates = np.arange(len(images))
    result = {}
    for name, pics in subsets.items():
        info, _ = e.evaluate_seen(data, 'arm', pics)
        mask = np.isin(targets, pics)
        t, f = targets[mask], fused[mask]
        truth = f[np.arange(len(t)), t, None]
        rank = 1 + (f > truth).sum(1) + ((f == truth) & (candidates < t[:, None])).sum(1)
        result[name] = dict(n=len(t), ocf=info['ocf_mean'], top1=info['top1_accuracy'],
                            recall1=float(np.mean(rank == 1)), recall32=float(np.mean(rank <= 32)))
    return result


def run_mouse(group, split, day, images, bank, coeff, subsets):
    mouse = group[0].mouse
    if sorted(b.session for b in group) != list(DAYS) or any(b.split != 'train' for b in group):
        raise ValueError(f'Expected four labeled training days for {mouse}')
    glyphs = np.arange(len(images))
    means, xs, ys, repeats = [], [], [], []
    for b in group:
        part = split.blocks[b.key]
        if b.session == day:
            target_block = b
            continue
        raw = b.features[part.train_trials]
        labels = b.trial_targets[part.train_trials]
        ids, counts = e.aggregate_by_target(raw, labels)
        np.testing.assert_array_equal(ids, glyphs)
        repeats.extend(np.unique(np.bincount(labels)).tolist())
        means.append(counts)
        norm = e.fit_session_normalizer(raw)
        xs.extend((e.apply_session_normalizer(raw, norm), e.apply_session_normalizer(counts, norm)))
        ys.extend((labels, ids))
    if len(set(repeats)) != 1:
        raise ValueError('CNN mean-count loss requires equal training repeats')
    x = np.concatenate(xs).astype(np.float32, copy=False)
    y = np.concatenate(ys).astype(np.int64, copy=False)
    del xs, ys
    ridge = e.RidgePath().fit(np.tile(coeff, (len(means), 1)), np.log1p(np.concatenate(means))).predict(coeff, 1.)
    rate, loss = exp24.train(bank, np.mean(means, axis=0))
    idx = split.blocks[target_block.key].validation_trials
    raw, labels = target_block.features[idx], target_block.trial_targets[idx]
    targets, obs = e.aggregate_by_target(raw, labels)
    np.testing.assert_array_equal(targets, glyphs)
    target_x = e.apply_session_normalizer(obs, e.fit_session_normalizer(raw))
    epochs = e.EPOCHS['task2'][mouse]
    logits = []
    for seed in e.SEEDS:
        model = fit_mlp(x, y, seed, epochs, len(images))
        logits.append(e.predict_logits(model, target_x, device=DEVICE))
        del model
    mlp = e.row_zscore(np.mean(np.stack(logits), axis=0, dtype=np.float32))
    del logits, x, y
    metrics = {}
    for arm in ARMS:
        if arm == 'ridge_mse':
            scores = e.response_scores(np.log1p(obs), ridge, metric='mse', device=DEVICE)
        else:
            scores = ((obs.astype(np.float32) @ np.log(rate).T - rate.sum(1)) * np.bincount(labels)[:, None]).astype(np.float32)
        fused = .5 * mlp + .5 * e.row_zscore(scores)
        metrics[arm] = measure(images, target_block, targets, fused, subsets)
    return dict(mouse=mouse, held_session=day, train_sessions=[d for d in DAYS if d != day],
                mlp_epochs=epochs, mlp_seeds=list(e.SEEDS), cnn_epoch_loss=loss,
                train_repeats_per_glyph_per_day=repeats[0], metrics=metrics)


def pool(records, subsets):
    return {arm: {s: {k: float(np.mean([r['metrics'][arm][s][k] for r in records]))
                      for k in METRICS} for s in subsets} for arm in ARMS}


def main():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    historical = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8'))
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError('Unexpected training glyph bank')
    bank = torch.from_numpy(images[:, None]).to(DEVICE)
    coeff = e.dct_encode(images, 16)
    order = np.random.default_rng(123).permutation(len(images))
    subsets = dict(tune=np.sort(order[:256]), held_out=np.sort(order[256:768]))
    blocks = e.load_feature_blocks(e.DATA, 'train', bin_ms=50)
    if {b.mouse for b in blocks} != {'mice1', 'mice2'}:
        raise ValueError('Expected both training mice')
    records, folds = [], {}
    for day in DAYS[:3]:
        split = e.make_split(blocks, 'task2', seed=42, validation_session=day)
        fold = []
        for mouse in ('mice1', 'mice2'):
            record = run_mouse([b for b in blocks if b.mouse == mouse], split, day, images, bank, coeff, subsets)
            write(OUT / day / f'{mouse}.json', record)
            fold.append(record)
            print(day, mouse, record['metrics'], flush=True)
        folds[day] = pool(fold, subsets)
        records.extend(fold)
    write(OUT / 'results.json', dict(
        protocol='Two fixed arms, no selection. Final fixed-epoch MLP weights; no validation checkpoints. Target session normalization uses unlabeled spikes; grouping follows experiment 24.',
        picture_ids={s: p.tolist() for s, p in subsets.items()}, device=DEVICE,
        new_folds=folds, new_macro_mean=pool(records, subsets),
        session4_historical=dict(source='test/05_model_architecture/24_forward_cnn/results.json',
                                note='Not rerun or pooled. Original MLP checkpoints selected using session4 validation.',
                                **{s: {arm: historical[s][arm] for arm in ARMS} for s in subsets})))


if __name__ == '__main__':
    main()
