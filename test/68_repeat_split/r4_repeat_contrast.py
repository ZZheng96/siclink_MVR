"""Fixed paired-repeat experiment; R4 ranks are aligned by target only for evaluation."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import r4_benchmark as B
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED, EPOCHS, PAIRS, WIDTH = 42, 12, 64, 256
TEMP, LAMBDA, MIX = .1, .1, .2
Z = B.row_zscore


def training_data(mouse):
    blocks, scales = [], {}
    for s in B.SESSIONS:
        parts = []
        for r in (1, 2, 3):
            raw, y = B.count_features(mouse, s, r, 50)
            parts.append(np.log1p(raw[np.argsort(y)].astype(np.float32)))
        x = np.stack(parts)
        mu, sd = x.mean((0, 1)), x.std((0, 1)).clip(1e-5)
        scales[s] = mu, sd
        blocks.append((x-mu)/sd)
    return torch.from_numpy(np.stack(blocks)), scales


def fit(x, weight):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = nn.Sequential(nn.Linear(x.shape[-1], WIDTH), nn.ReLU(), nn.Linear(WIDTH, B.K)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    for _ in range(EPOCHS):
        # Each same-session pair has a distinct label within its minibatch.
        batches = [(s, ids[i:i+PAIRS]) for s in range(len(B.SESSIONS))
                   for ids in [rng.permutation(B.K)] for i in range(0, B.K, PAIRS)]
        for i in rng.permutation(len(batches)):
            s, ids = batches[i]
            r = rng.integers(3, size=len(ids)); q = (r+rng.integers(1, 3, size=len(ids))) % 3
            y = torch.from_numpy(np.tile(ids, 2)).long().cuda()
            a = torch.cat((x[s, r, ids], x[s, q, ids])).cuda()
            h = model[1](model[0](a))
            loss = F.cross_entropy(model[2](h), y)
            if weight:
                v = F.normalize(h, dim=1)
                sims = (v @ v.T) / TEMP
                sims.fill_diagonal_(-torch.inf)
                partner = torch.arange(len(y), device=y.device).roll(len(ids))
                loss = loss + weight*F.cross_entropy(sims, partner)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


def current(mouse, s, y):
    vs, ls = [], []
    for run, ms in [('e53_10e7', 10), ('e53_25e12', 25), ('e53_50e9', 50), ('v5_250e7', 250)]:
        d = B.session_scores(run, mouse, s, ms)
        assert np.array_equal(y, d['y'])
        if ms != 250:
            vs.append(Z(d['v5_s3'])); ls.append(Z(d['logits_s3']))
        else:
            x = np.mean([Z(np.mean(vs, axis=0)), Z(np.mean(ls, axis=0)), Z(d['logits_s3'])], axis=0)
        del d
    row = np.partition(x, -100, axis=1)[:, -100:].mean(1, keepdims=True)
    col = np.partition(x, -100, axis=0)[-100:].mean(0, keepdims=True)
    return 2*x-row-col


def ranks(a, y):
    assert np.isfinite(a).all()
    t = a[np.arange(len(y)), y, None]
    return (1+(a>t).sum(1)+((a==t)&(np.arange(B.K)<y[:, None])).sum(1))[np.argsort(y)]


def metrics(rr, ref):
    result = {}
    for k in (1, 32):
        a, b = rr <= k, ref <= k
        h, base = a.any(0), b.any(0)
        result[f'top{k}'] = dict(coverage=int(h.sum()), trials=int(a.sum()),
                                  new=int((h & ~base).sum()), lost=int((base & ~h).sum()),
                                  trial_new=int((a & ~b).sum()), trial_lost=int((b & ~a).sum()),
                                  hit_frequency_0_to_8=np.bincount(a.sum(0), minlength=9).tolist())
    result['best_rank_bins_1_2to5_6to32_33plus'] = np.histogram(rr.min(0), [1, 2, 6, 33, B.K+1])[0].tolist()
    result['top1_from_current_never_top32'] = int(((rr.min(0)==1)&(ref.min(0)>32)).sum())
    return result


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    B.CENTERED = B.dct_encode(B.IMAGES, 16)
    B.CENTERED -= B.CENTERED.mean(0, keepdims=True)
    table = {k: [] for k in ('current', 'ce', 'supcon', 'fusion_ce', 'fusion_supcon')}
    for mouse in B.MICE:
        x, scales = training_data(mouse)
        models = {k: fit(x, w) for k, w in [('ce', 0), ('supcon', LAMBDA)]}
        del x
        for s in B.SESSIONS:
            raw, y = B.count_features(mouse, s, 4, 50)
            mu, sd = scales[s]
            xt = torch.from_numpy((np.log1p(raw.astype(np.float32))-mu)/sd)
            base = current(mouse, s, y)
            table['current'].append(ranks(base, y)); base = Z(base)
            with torch.inference_mode():
                for k, net in models.items():
                    logits = np.concatenate([net(b.cuda()).cpu().numpy() for b in xt.split(128)])
                    table[k].append(ranks(logits, y))
                    table['fusion_'+k].append(ranks((1-MIX)*base + MIX*Z(logits), y))
            print(mouse, s, flush=True)
    ref = np.stack(table['current'])
    assert ((ref==1).sum(), (ref.min(0)==1).sum(), (ref.min(0)>32).sum()) == (572, 462, 531)
    result = dict(seed=SEED, epochs=EPOCHS, pairs_per_batch=PAIRS, hidden=WIDTH, bin_ms=50,
                  optimizer='AdamW(lr=.001, weight_decay=.01), constant LR, last epoch',
                  train='R1-3, one model per mouse, session-specific log1p z-score. Each epoch: one random distinct-repeat pair per session and glyph. Matched initialization and minibatches.',
                  eval='R4, independent MLP logits; historical current includes session CSLS100. Ties: lower class ID first.',
                  supcon_weight=LAMBDA, temperature=TEMP, fusion='0.8*row_z(current)+0.2*row_z(branch_logits), fixed',
                  classes=B.K, trials=int(ref.size), models={k: metrics(np.stack(v), ref) for k, v in table.items()})
    print(json.dumps(result['models']), flush=True)
    Path(__file__).with_name('REPEAT_CONTRAST.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
