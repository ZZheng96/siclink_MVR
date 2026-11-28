"""Within-session repeat holdout: 250 ms total vs real/permuted half-window contrasts."""
import json
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/derived/repeat_split_v1'
K, SEED = 2748, 42
MICE = {'mice1': 134, 'mice2': 78}
DAYS = (1, 2, 3, 4)
MODES = ('total', 'real', 'shuffled')
GRID = (0.00003, 0.0003, 0.001)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def block(m, s, r):
    units = MICE[m]
    rng = np.random.default_rng([SEED, units, s, r])
    with np.load(DATA / m / f'session{s}_repeat{r}.npz', allow_pickle=False) as z:
        y, p, t, u = (z[k] for k in ('target', 'trial_ptr', 'spike_samples', 'unit_index'))
        assert np.array_equal(z['group_ptr'], np.arange(K + 1)) and np.all(z['repeat_index'] == r)
        assert np.array_equal(np.sort(y), np.arange(K)) and p.size == K + 1 and p[0] == 0
        assert p[-1] == t.size == u.size and np.all(np.diff(p) >= 0)
        assert np.all((t >= 0) & (t < 7500)) and np.all((u >= 0) & (u < units))
    idx = np.repeat(np.arange(K), np.diff(p)) * units + u
    def count(keys):
        return np.bincount(keys, minlength=K * units).reshape(K, units).astype(np.float32)
    total = count(idx)
    early = t < 3750
    perm = early.copy()
    for lo, hi in zip(p[:-1], p[1:]):
        rng.shuffle(perm[lo:hi])  # Permuting half labels is equivalent to permuting event times.
    features = [np.log1p(total)]
    for mask in (early, perm):
        diff = 2 * count(idx[mask]) - total
        features.append(np.concatenate([features[0], np.sign(diff) * np.log1p(abs(diff))], axis=1))
    return features, y


def prepare(blocks, j, repeats):
    xs, ys, scales = [], [], {}
    for s in DAYS:
        x = np.concatenate([blocks[s, r][0][j] for r in repeats])
        y = np.concatenate([blocks[s, r][1] for r in repeats])
        mu, sd = [fn(x, axis=0, dtype=np.float64).astype(np.float32) for fn in (np.mean, np.std)]
        sd[sd == 0] = 1
        scales[s] = mu, sd
        xs.append((x - mu) / sd)
        ys.append(y)
    return *(torch.as_tensor(np.concatenate(v), device=DEVICE) for v in (xs, ys)), scales


def fit(x, y, penalty):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    model = torch.nn.Linear(x.shape[1], K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(12):
        for ix in torch.randperm(len(y), device=DEVICE, generator=gen).split(512):
            loss = F.cross_entropy(model(x[ix]), y[ix]) + (penalty / 2) * model.weight.square().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score(model, x, y, scale):
    mu, sd = scale
    x = torch.as_tensor((x - mu) / sd, device=DEVICE)
    y = torch.as_tensor(y, device=DEVICE)
    logits = model(x)
    nll = F.cross_entropy(logits, y).item()
    assert np.isfinite(nll)
    true = logits.gather(1, y[:, None])
    rank0 = ((logits > true) | ((logits == true) & (torch.arange(K, device=DEVICE) < y[:, None]))).sum(1)
    hits = (rank0 < 32)[torch.argsort(y)].cpu().numpy()
    return {'top1': (rank0 == 0).float().mean().item(), 'r32': hits.mean().item(), 'nll': nll}, hits


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ('top1', 'r32', 'nll')}


def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {'protocol': {'seed': SEED, 'device': DEVICE, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'n_classes': K, 'n_per_mouse_day_R4': K, 'epochs': 12, 'batch': 512, 'adam_lr': 0.001,
        'selection': 'R1+2 fit, minimum mean four-session R3 NLL per mouse and mode; first on tie',
        'loss': 'mean CE + lambda/2 * sum(weight**2); bias unpenalized', 'grid': GRID,
        'features': 'log1p(total); append signed log1p(abs(early-late)) real or shuffled within trial',
        'scaling': 'per session fit repeats only; std(ddof=0); zero std -> 1',
        'paired_ci': 'R32 proportion differences; percentile bootstrap by character jointly over days and mice; 2000 draws, seed 31415'},
        'mice': {m: {} for m in MICE}}
    locked, hits = {}, {}
    for m in MICE:
        blocks = {(s, r): block(m, s, r) for s in DAYS for r in (1, 2, 3)}
        for j, mode in enumerate(MODES):
            x, y, scales = prepare(blocks, j, (1, 2))
            val = []
            for lam in GRID:
                model = fit(x, y, lam)
                val.append(float(np.mean([score(model, blocks[s, 3][0][j], blocks[s, 3][1], scales[s])[0]['nll'] for s in DAYS])))
            lam = GRID[np.argmin(val)]
            x, y, scales = prepare(blocks, j, (1, 2, 3))
            locked[m, mode] = fit(x, y, lam), scales
            results['mice'][m][mode] = {'lambda': lam, 'r3_nll_in_grid_order': val, 'per_session': {}}
    for m in MICE:
        for s in DAYS:
            features, y = block(m, s, 4)
            for j, mode in enumerate(MODES):
                model, scales = locked[m, mode]
                row, hits[m, s, mode] = score(model, features[j], y, scales[s])
                results['mice'][m][mode]['per_session'][s] = row
        for mode in MODES:
            entry = results['mice'][m][mode]
            entry['macro'] = average(list(entry['per_session'].values()))
    results['macro'] = {mode: average([results['mice'][m][mode]['macro'] for m in MICE]) for mode in MODES}
    boot = np.random.default_rng(31415).integers(K, size=(2000, K), dtype=np.int32)
    paired = {}
    for a, b in [('real', 'total'), ('real', 'shuffled'), ('shuffled', 'total')]:
        delta = {m: np.mean([hits[m,s,a].astype(float) - hits[m,s,b] for s in DAYS], axis=0) for m in MICE}
        delta['macro'] = np.mean(list(delta.values()), axis=0)
        paired[f'{a}-{b}'] = {m: {'delta': d.mean(), 'ci95': np.quantile(d[boot].mean(1), [.025,.975]).tolist()} for m,d in delta.items()}
    results['paired_r32'] = paired
    text = json.dumps(results, indent=2, allow_nan=False)
    (HERE / 'HALF_WINDOW_RESULTS.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
