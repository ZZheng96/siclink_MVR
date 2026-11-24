"""Route 2, formal public training data only. From repo root: python -B test/26_temporal_attention/run.py

Fixed epochs; independent full-gallery retrieval (no block assignment). Task2 uses
labels only to reconstruct the four-repeat validation sample and score it. Task3
never combines validation trials. Historical picture subsets are not blind tests.
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from src.dataloader import load_images
from src.evaluation import score_predictions
from src.features import load_feature_blocks
from src.splits import make_split

DATA = ROOT / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED, BIN_MS, BATCH, PRE, EPOCHS = 42, 25, 96, 2, 8
ARMS = ('mlp', 'mix', 'attention', 'shuffled_attention')
MICE = ('mice1', 'mice2')


def glyph_bank(images):
    # Fixed image CNN: ink density plus three oriented edge magnitudes, with layout.
    kernels = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]], [[-1,0,1],[-2,0,2],[-1,0,1]],
                            [[-1,-2,-1],[0,0,0],[1,2,1]], [[-2,-1,0],[-1,0,1],[0,1,2]]],
                           dtype=torch.float32, device=DEVICE)[:, None]
    rows = []
    for x in torch.from_numpy(images).split(BATCH):
        fmap = F.conv2d(x[:, None].to(DEVICE), kernels, padding=1).abs()
        rows.append(F.avg_pool2d(fmap, 8).flatten(1))
    p = torch.cat(rows)
    return F.normalize((p - p.mean(0)) / p.std(0).clamp_min(.01))


class Decoder(nn.Module):
    def __init__(self, arm, units, bins, dim):
        super().__init__()
        self.arm = arm
        width, slots = 24, 4
        if arm == 'mlp':
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(units*bins, 48), nn.GELU())
            size = 48
        else:
            self.time = nn.Sequential(nn.Conv1d(1, 4, 3, padding=1), nn.GELU(), nn.Flatten(),
                                      nn.Linear(4*bins, width))
            self.identity = nn.Parameter(torch.randn(units, width)*.02)
            if arm == 'mix':
                self.mix = nn.Linear(units, slots)
            else:
                self.query = nn.Parameter(torch.randn(slots, width)*.02)
                self.layers = nn.ModuleList(nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(2))
                self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.layers)
            size = slots*width
        # Match inference parameters to ~120k by adjusting the active readout width.
        used = sum(p.numel() for p in self.parameters())
        hidden = (120000-used-dim)//(size+dim+1)
        self.readout = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.count_head = nn.Linear(dim, units*bins)  # same pretraining-only head across arms

    def forward(self, x):
        if self.arm == 'mlp':
            h = self.encoder(x)
        else:
            tokens = self.time(x.reshape(-1, 1, x.size(-1))).reshape(len(x), x.size(1), -1) + self.identity
            if self.arm == 'mix':
                h = self.mix(tokens.transpose(1, 2)).flatten(1)
            else:
                q = self.query.expand(len(x), -1, -1)
                for layer, norm in zip(self.layers, self.norms):
                    q = norm(q + layer(q, tokens, tokens, need_weights=False)[0])
                h = q.flatten(1)
        return self.readout(h)


def train(group, split, arm, proto, hard):
    # A fixed within-source-day permutation breaks spike/label alignment, not marginals.
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED+1)
    xs, ys = [], []
    units = group[0].n_units
    for b in group:
        idx = split.blocks[b.key].train_trials
        if len(idx):
            ys.append(b.trial_targets[idx])
            src = rng.permutation(idx) if arm == 'shuffled_attention' else idx
            xs.append(b.features[src].reshape(len(idx), units, -1))
    x = torch.from_numpy(np.concatenate(xs)).to(DEVICE)
    labels = np.concatenate(ys).astype(np.int64)
    # Each class has equal exposure; source-only lookup enables distinct positive trials.
    counts = np.bincount(labels, minlength=len(proto))
    if counts.min() < 2 or counts.min() != counts.max() or len(counts) != len(proto):
        raise ValueError('Expected equal repeated coverage of the training gallery')
    lookup = torch.as_tensor(np.argsort(labels, kind='stable').reshape(len(proto), -1), device=DEVICE)
    net = Decoder(arm, units, x.size(-1), proto.size(1)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    torch.manual_seed(SEED+2)
    for epoch in range(PRE + EPOCHS):
        indices = torch.randperm(len(x), device=DEVICE)
        for ids in indices.split(BATCH):
            if epoch < PRE:
                raw = x[ids]
                mask = torch.rand_like(raw) < .25
                z = net(raw.log1p().masked_fill(mask, -1))
                log_rate = net.count_head(z).reshape_as(raw)
                loss = F.poisson_nll_loss(log_rate[mask], raw[mask])
            else:
                # Enumerate each glyph/repeat once per epoch, pairing a different repeat.
                y, rep = ids // lookup.size(1), ids % lookup.size(1)
                other = (rep + torch.randint(1, lookup.size(1), rep.shape, device=DEVICE)) % lookup.size(1)
                pair = torch.cat((lookup[y, rep], lookup[y, other]))
                z = F.normalize(net(x[pair].log1p()))
                target = y.repeat(2)
                scores = 16 * (z @ proto.T)
                loss = (F.cross_entropy(scores, target) + .2*(1-(z[:len(y)]*z[len(y):]).sum(1)).mean()
                        + .2*F.softplus(scores.gather(1, hard[target]) - scores.gather(1, target[:,None])).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
    return net.eval()


@torch.no_grad()
def measure(models, blocks, split, pictures, images, proto):
    groups = {}
    repeats = 4 if split.task == 'task2' else 1
    for b in blocks:
        idx = split.blocks[b.key].validation_trials
        labels = b.trial_targets[idx]
        take = np.flatnonzero(np.isin(labels, pictures))
        if not len(idx):
            continue
        order = take[np.argsort(labels[take], kind='stable')]
        np.testing.assert_array_equal(labels[order], np.repeat(pictures, repeats))
        scores = []
        for rows in np.array_split(idx[order], np.arange(BATCH, len(order), BATCH)):
            raw = torch.from_numpy(b.features[rows]).to(DEVICE).reshape(len(rows), b.n_units, -1)
            scores.append((F.normalize(models[b.mouse](raw.log1p())) @ proto.T).cpu().numpy())
        avg = np.concatenate(scores).reshape(len(pictures), repeats, len(proto)).mean(1)
        ranked = np.argsort(-avg, axis=1, kind='stable')
        ocf = score_predictions(images[ranked[:,0]], images[pictures], device=DEVICE)
        groups[b.key] = dict(n=len(pictures), ocf=float(np.mean(ocf)), **{
            f'recall{k}': float((ranked[:,:k] == pictures[:,None]).any(1).mean()) for k in (1,8,32)})
    pooled = {k: float(np.mean([g[k] for g in groups.values()])) for k in ('ocf','recall1','recall8','recall32')}
    return dict(blocks=groups, pooled=pooled)


def main():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (2748,64,64):
        raise ValueError('Expected formal training gallery')
    proto = glyph_bank(images)
    similarity = proto @ proto.T
    similarity.fill_diagonal_(-torch.inf)
    hard = similarity.argsort(dim=1, descending=True, stable=True)[:,:8].contiguous()
    del similarity
    order = np.random.default_rng(123).permutation(len(images))
    tune, held = np.sort(order[:256]), np.sort(order[256:768])
    blocks = load_feature_blocks(DATA, split='train', bin_ms=BIN_MS)
    expected = [f'{m}/session{i}' for m in MICE for i in range(1,5)]
    if sorted(b.key for b in blocks) != expected or any(b.split != 'train' for b in blocks):
        raise ValueError('Expected both mice and all four labeled days')
    results = dict(seed=SEED, bin_ms=BIN_MS, batch=BATCH, pre_epochs=PRE, epochs=EPOCHS,
                   python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=DEVICE,
                   tune_ids=tune.tolist(), heldout_ids=held.tolist(), subset_seed=123,
                   rule='Highest pooled tune OCF among real-input arms; earlier arm wins ties. Require strictly above shuffled control for heldout.',
                   limitation='Reused development subsets; no cross-fold replication or competition test inference.')
    for task in ('task2', 'task3'):
        split = make_split(blocks, task, seed=SEED, validation_session='session4')
        report = dict(split=split.metadata(), tune={}, parameters={}, heldout=None)
        best, best_models = None, None
        for arm in ARMS:
            models = {m: train([b for b in blocks if b.mouse == m], split, arm, proto, hard) for m in MICE}
            report['parameters'][arm] = {m: sum(p.numel() for n,p in net.named_parameters() if not n.startswith('count_head.')) for m,net in models.items()}
            metric = measure(models, blocks, split, tune, images, proto)
            report['tune'][arm] = metric
            if arm != 'shuffled_attention' and (best is None or metric['pooled']['ocf'] > report['tune'][best]['pooled']['ocf']):
                best, best_models = arm, models
            print(task, arm, 'tune', metric['pooled'], flush=True)
        report['selected'] = best
        if report['tune'][best]['pooled']['ocf'] > report['tune']['shuffled_attention']['pooled']['ocf']:
            report['heldout'] = measure(best_models, blocks, split, held, images, proto)
        results[task] = report
        tmp = OUT / 'results.tmp'
        tmp.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(OUT / 'results.json')


if __name__ == '__main__':
    main()
