"""Task2 fixed session4 comparison. Run with the repository venv, python -B."""
import os
import sys
import importlib.util
import json
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e18', OUT.parent / '18_seed_ensemble/run.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
EPOCHS, BATCH = 12, 128
BASE = 'ridge_mse'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(nn.Conv2d(1, 8, k, stride=2, padding=k//2), nn.SiLU(), nn.AvgPool2d(2), nn.Conv2d(8, 12, 3, padding=1), nn.SiLU(), nn.AvgPool2d(4)) for k in (3, 7))
        self.head = nn.Linear(384, len(mean))
        with torch.no_grad():
            self.head.bias.copy_(mean.clamp_min(0.001).expm1().log())

    def forward(self, x):
        return F.softplus(self.head(torch.cat([b(x).flatten(1) for b in self.branches], 1))) + 1e-6


def train(bank, counts):
    torch.manual_seed(42)
    # Per-glyph sufficient statistics: all trials have equal exposure and repeats.
    y = torch.as_tensor(counts, dtype=torch.float32, device=DEVICE)
    model = Encoder(y.mean(0)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    history = []
    for epoch in range(EPOCHS):
        total = 0.
        for idx in torch.randperm(len(y), device=DEVICE).split(BATCH):
            rate = model(bank[idx])
            loss = F.poisson_nll_loss(rate, y[idx], log_input=False)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        history.append(total / len(y))
    with torch.no_grad():
        rates = torch.cat([model(x) for x in bank.split(BATCH)]).cpu().numpy()
    return rates, history


def main():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    images = (e.load_images(e.DATA) >= 128).astype(np.float32)
    assert images.shape == (2748, 64, 64)
    bank = torch.from_numpy(images[:, None]).to(DEVICE)
    coeff = e.dct_encode(images, 16)
    order = np.random.default_rng(123).permutation(len(bank))
    subsets = dict(tune=np.sort(order[:256]), held_out=np.sort(order[256:768]))
    blocks = e.load_feature_blocks(e.DATA, 'train', bin_ms=50)
    split = e.make_split(blocks, 'task2', seed=42)
    result = dict(device=DEVICE, torch=str(torch.__version__), numpy=np.__version__, python=sys.version, config=dict(epochs=EPOCHS, batch=BATCH, cnn_seed=42, train_sessions=[1,2,3], validation_session=4, bin_ms=50, candidates=2748, split_seed=42, subset_seed=123, mlp_seeds=list(e.SEEDS), mlp_weight=.5, ridge_alpha=1, baseline=BASE, selection='highest pooled CNN tune OCF, MSE wins tie; require strict improvement over ridge_mse on tune and held-out to advance', recall='fused score before assignment; ties broken by candidate index', top1='after full-block unique_assignment', ridge='fit DCT16 to log1p(session-glyph mean); Poisson arm converts via expm1 and floors at 1e-6', cnn='Poisson loss on mean counts (equal repeats); log1p rates for MSE; sum log likelihood across repeats for Poisson'), training={}, tune={}, held_out={})
    data = dict(images=images, rows=[], device=DEVICE, training={})
    for mouse in sorted({b.mouse for b in blocks}):
        group = [b for b in blocks if b.mouse == mouse]
        assert sorted(b.session for b in group) == ['session1','session2','session3','session4']
        means, ids, repeats = [], [], []
        for b in group:
            fold = split.blocks[b.key]
            tr = len(fold.train_trials) > 0
            idx = fold.train_trials if tr else fold.validation_trials
            raw, labels = b.features[idx], b.trial_targets[idx]
            targets, counts = e.aggregate_by_target(raw, labels)
            np.testing.assert_array_equal(targets, np.arange(len(bank)))
            rep = np.bincount(labels)
            if tr:
                means.append(counts); ids.append(targets); repeats.extend(np.unique(rep).tolist())
            else:
                row = dict(block=b.key, mouse=mouse, targets=targets, counts=counts, repeats=rep, mlp_counts=e.apply_session_normalizer(counts, e.fit_session_normalizer(raw)), logits={}, assignments={}, ranks={})
        assert len(set(repeats)) == 1
        ridge = e.RidgePath().fit(coeff[np.concatenate(ids)], np.log1p(np.concatenate(means))).predict(coeff, 1.)
        cnn, history = train(bank, np.mean(means, axis=0))
        row['prototypes'] = {'ridge':ridge,'cnn':cnn}
        data['rows'].append(row); data['training'][mouse] = True
        result['training'][mouse] = dict(loss=history, units=b.n_units, outputs=cnn.shape[1], repeats_per_session_glyph=repeats[0])
        print(mouse, history, flush=True)
    del blocks
    for seed in e.SEEDS:
        e.predict_seen_seed('task2', data, seed)
    for row in data['rows']:
        mlp = e.row_zscore(np.mean(np.stack([row['logits'][s] for s in e.SEEDS]), axis=0, dtype=np.float32)); row['logits'].clear()
        obs = row['counts']; target = row['targets']
        for name, proto in row['prototypes'].items():
            logproto = proto if name == 'ridge' else np.log1p(proto)
            rate = np.maximum(np.expm1(proto), 1e-6) if name == 'ridge' else proto
            for metric in ('mse','poisson'):
                arm = name+'_'+metric
                if metric == 'mse':
                    scores = e.response_scores(np.log1p(obs), logproto, metric='mse', device=DEVICE)
                else:
                    scores = ((obs.astype(np.float32) @ np.log(rate).T - rate.sum(1)) * row['repeats'][:,None]).astype(np.float32)
                fused = .5*mlp + .5*e.row_zscore(scores)
                row['assignments'][arm] = e.unique_assignment(fused)
                row['ranks'][arm] = fused  # rank only labeled subsets allowed by gate

    def evaluate(arm, subset):
        by = {}; pics = subsets[subset]
        for r in data['rows']:
            info, _ = e.evaluate_seen(dict(data, rows=[r]), arm, pics)
            mask = np.isin(r['targets'], pics); t = r['targets'][mask]; f = r['ranks'][arm][mask]
            truth = f[np.arange(len(t)),t,None]
            ranks = 1+(f>truth).sum(1)+((f==truth)&(np.arange(len(bank))<t[:,None])).sum(1)
            by[r['mouse']] = dict(n=len(t), ocf=info['ocf_mean'], top1=info['top1_accuracy'], recall1=float(np.mean(ranks==1)), recall32=float(np.mean(ranks<=32)))
        agg = {k:float(np.mean([v[k] for v in by.values()])) for k in ('ocf','top1','recall1','recall32')}
        result[subset][arm] = dict(pooled=agg, mice=by)
        print(subset, arm, by, agg, flush=True)
        return agg['ocf']

    for arm in ('ridge_mse','ridge_poisson','cnn_mse','cnn_poisson'):
        evaluate(arm,'tune')
    score = lambda a: result['tune'][a]['pooled']['ocf']
    selected = max(('cnn_mse','cnn_poisson'), key=score)
    baseline = evaluate(BASE,'held_out')
    errors = [abs(score(BASE)-25.831461860596946), abs(baseline-24.33936155396989)]
    passed = max(errors) <= 1e-5
    gate = passed and score(selected)>score(BASE)
    result.update(reproduction=dict(abs_errors_tune_held_out=errors, passed=passed), selected=selected, tune_gate=gate, advance=bool(gate and evaluate(selected,'held_out')>baseline))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False), encoding='utf-8')
    print('gate', gate, 'advance',result['advance'], flush=True)


if __name__ == '__main__':
    main()
