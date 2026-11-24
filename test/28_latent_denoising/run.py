"""Route 4: real public training trials only. Run this file with the project Python.

Four matched arms share a trial classifier, encoder and inference rule. D enables
masked-count gradients into the encoder; A adds an episodic same-glyph loss.
The D-off count head is a detached diagnostic probe, not an untrained baseline.
Task2 uses labels ONLY to construct labeled-training pseudo-samples (the public
training catalog stores repeats separately), select evaluation IDs and score.
No target-day fit, transductive assignment or across-query normalization.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.dataloader import load_images
from src.evaluation import score_predictions
from src.features import load_feature_blocks
from src.splits import make_split

DATA = ROOT / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset'
DAYS = tuple(f'session{i}' for i in range(1, 5))
MICE = ('mice1', 'mice2')
ARMS = ('base', 'denoise', 'align', 'both')
BATCH = 128
SEED = 42


class Temporal(nn.Module):
    def __init__(self, units, days, classes):
        super().__init__()
        self.input = nn.Linear(units * 2, 32)
        self.state = nn.GRU(32, 32, batch_first=True)
        self.head = nn.ModuleList(nn.Linear(32, units) for _ in days)
        self.classifier = nn.Linear(5 * 32, classes)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros_like(x[:, :1], dtype=torch.bool)
        mask = mask.expand_as(x)
        obs = torch.cat((x.masked_fill(mask, 0).log1p(), mask.to(x.dtype)), dim=-1)
        return self.state(self.input(obs).tanh())[0]


def cube(block, idx, classes, repeats):
    # [glyph, repeat, time, unit]; group only inside the designated split.
    labels = block.trial_targets[idx]
    if not np.array_equal(np.bincount(labels, minlength=classes), np.full(classes, repeats)):
        raise ValueError(f'Unexpected repeat coverage in {block.key}')
    raw = block.features[idx[np.argsort(labels, kind='stable')]]
    return torch.from_numpy(raw.reshape(classes, repeats, block.n_units, 5).transpose(0, 1, 3, 2).copy())


def unit_mask(x, rng):
    # Exactly one fifth of units (rounded down) hidden over the whole trajectory.
    n, _, u = x.shape
    indices = rng.random((n, u)).argsort(axis=1)[:, :max(1, u // 5)]
    mask = torch.zeros((n, 1, u), dtype=torch.bool, device=x.device)
    return mask.scatter_(2, torch.as_tensor(indices[:, None], device=x.device), True)


def rates(head, z):
    return F.softplus(head(z)) + 1e-6


def fit(source, arm, args):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    classes, repeats, _, units = source[0].shape
    model = Temporal(units, source, classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(args.epochs):
        for day in rng.permutation(len(source)):
            other = (day + rng.integers(1, len(source))) % len(source)
            for r in rng.permutation(repeats):
                order = rng.permutation(classes)
                for start in range(0, classes, BATCH):
                    ids = torch.from_numpy(order[start:start+BATCH]).long()
                    x = source[day][ids, r].to(args.device)
                    z = model(x).flatten(1)
                    loss = F.cross_entropy(model.classifier(z), ids.to(args.device))
                    mask = unit_mask(x, rng)
                    hidden = model(x, mask)
                    if arm not in ('both', 'denoise'):
                        hidden = hidden.detach()
                    rate = rates(model.head[day], hidden)
                    loss = loss + (rate - x * rate.log()).masked_select(mask.expand_as(x)).mean()
                    if arm in ('both', 'align'):
                        # Prototype excludes the query. One different repeat from its
                        # own day and one labeled source-day trial; negatives = other glyphs.
                        supports = torch.cat((source[day][ids, (r+1) % repeats], source[other][ids, r]))
                        a, b = model(supports.to(args.device)).flatten(1).chunk(2)
                        sim = F.normalize(z) @ F.normalize((a+b)/2).T
                        loss = loss + F.cross_entropy(sim / .1, torch.arange(len(ids), device=x.device))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    opt.step()
    return model.eval()


@torch.no_grad()
def prototypes(model, source, device):
    # Equal weight per source day/repeat. Full seen-glyph bank in every arm.
    banks = []
    for day in source:
        parts = []
        for x in day.split(BATCH):
            z = model(x.flatten(0, 1).to(device)).reshape(len(x), x.size(1), -1)
            parts.append(z.mean(1))
        banks.append(torch.cat(parts))
    return F.normalize(torch.stack(banks).mean(0))


@torch.no_grad()
def predict(model, bank, query, seen_day, device, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for x in query.split(BATCH):
        n, rep = x.shape[:2]
        x = x.flatten(0, 1).to(device)
        z = F.normalize(model(x).flatten(1))
        scores = (z @ bank.T).reshape(n, rep, -1).mean(1)
        best = scores.argmax(1)
        first, second = scores.topk(2).values.unbind(1)
        mask = unit_mask(x, rng)
        hidden = model(x, mask)
        heads = model.head if seen_day is None else [model.head[seen_day]]
        # Unseen day: fixed arithmetic mean of source rates, no fitted target head.
        rate = torch.stack([rates(h, hidden) for h in heads]).mean(0)
        ll = (x * rate.log() - rate - (x+1).lgamma()).masked_select(mask.expand_as(x))
        ll = ll.reshape(n, rep, -1).mean((1, 2))  # nats per hidden unit-bin, before repeat pooling
        rows.append(torch.stack((best, first-second, ll), dim=1).cpu().numpy())
    return np.concatenate(rows)


def metrics(images, ids, rows, device, cuts):
    pred, margin, ll = rows.T
    ocf = np.asarray(score_predictions(images[pred.astype(int)], images[ids], device=device))
    if cuts is None:
        cuts = np.quantile(margin, [1/3, 2/3]).tolist()
    bins = np.searchsorted(cuts, margin, side='right')
    return dict(n=len(ids), top1=float((pred == ids).mean()), ocf=float(ocf.mean()),
                masked_loglik_per_unit_bin=float(ll.mean()), margin_cuts=cuts,
                confidence=[dict(n=int((bins == i).sum()), ocf=float(ocf[bins == i].mean())
                                 if np.any(bins == i) else None) for i in range(3)])


def run_fold(task, day, blocks, images, tune, held, args):
    split = make_split(blocks, task, seed=SEED, validation_session=day or DAYS[-1])
    tune_rows = {arm: {} for arm in ARMS}
    pending = {arm: {} for arm in ARMS}
    for mouse in MICE:
        source, target, source_days = [], {}, []
        for block in (b for b in blocks if b.mouse == mouse):
            part = split.blocks[block.key]
            tr, va = part.train_trials, part.validation_trials
            if np.intersect1d(tr, va).size or (task == 'task2' and len(tr) and len(va)):
                raise ValueError(f'Invalid split: {block.key}')
            if len(tr):
                source_days.append(block.session)
                source.append(cube(block, tr, len(images), 3 if task == 'task3' else 4))
            if len(va):
                target[block.session] = cube(block, va, len(images), 1 if task == 'task3' else 4)
        for arm in ARMS:
            model = fit(source, arm, args)
            bank = prototypes(model, source, args.device)
            for session, query in target.items():
                key = f'{mouse}/{session}'
                seen = source_days.index(session) if session in source_days else None
                seed = 8000 + 10*MICE.index(mouse) + DAYS.index(session)
                def infer(ids, offset):
                    return predict(model, bank, query[ids], seen, args.device, seed+offset)
                tune_rows[arm][key] = metrics(images, tune, infer(tune, 0), args.device, None)
                # Independent inference; held-out outputs are not scored or inspected for selection.
                pending[arm][key] = infer(held, 100)
    means = {arm: float(np.mean([v['ocf'] for v in groups.values()])) for arm, groups in tune_rows.items()}
    winner = max(ARMS, key=means.get)  # fixed arm order breaks ties
    comparisons = {}
    for key in tune_rows['base']:
        base, d, a, both = (tune_rows[k][key]['ocf'] for k in ARMS)
        comparisons[key] = dict(denoise_without_align=d-base, denoise_with_align=both-a,
                                align_without_denoise=a-base, align_with_denoise=both-d,
                                interaction=both-a-d+base)
    evaluation = {arm: {key: metrics(images, held, rows, args.device, tune_rows[arm][key]['margin_cuts'])
                        for key, rows in pending[arm].items()} for arm in dict.fromkeys(('base', winner))}
    return dict(split=split.metadata(), tune=tune_rows, tune_macro_ocf=means, tune_ocf_contrasts=comparisons,
                selected=winner, held_out=evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('both', 'task2', 'task3'), default='both')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    blocks = load_feature_blocks(DATA, 'train', bin_ms=50)
    if images.shape != (2748, 64, 64) or sorted(b.key for b in blocks) != [f'{m}/{d}' for m in MICE for d in DAYS]:
        raise ValueError('Expected formal 2748-glyph training bank and eight mouse/session blocks')
    order = np.random.default_rng(123).permutation(len(images))
    tune, held = np.sort(order[:256]), np.sort(order[256:768])
    for task in ('task3', 'task2') if args.task == 'both' else (args.task,):
        for day in DAYS if task == 'task2' else (None,):
            result = run_fold(task, day, blocks, images, tune, held, args)
            result.update(config=vars(args), torch=str(torch.__version__), numpy=np.__version__,
                          seed=SEED, tune_ids=tune.tolist(), held_out_ids=held.tolist(),
                          protocol='Reused local validation, not blind. Fold-specific tune selection; target adaptation off.')
            path = HERE / f'results_{task}_{day or "single_trial"}.json'
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            tmp.replace(path)
            print(path, 'selected:', result['selected'], flush=True)


if __name__ == '__main__':
    main()
