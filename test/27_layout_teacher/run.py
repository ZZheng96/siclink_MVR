"""Fixed local Task 1 experiment. Run this file with the project's Python environment.

Needs the public training data, experiment 21's prediction archives, and the
experiment 15 font. Writes only under this experiment. No external labels.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.dataloader import load_images
from src.evaluation import score_predictions, summarize_scores
from src.features import aggregate_by_target, load_feature_blocks
from src.neural import SpikeRegressor
from src.retrieval import render_gb2312_bank
from src.splits import make_split

DATA = ROOT / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset'
BASE = ROOT / 'test/21_task1_directional_loss/artifacts'
FONT = Path(r'C:\Windows\Fonts\NotoSansSC-VF.ttf')
SEED = 42


def prepare():
    blocks = load_feature_blocks(DATA, 'train', bin_ms=25)
    split = make_split(blocks, 'task1', seed=SEED)
    val = np.random.default_rng(123).permutation(split.validation_pictures)
    train_ids = np.setdiff1d(np.arange(2748), val)
    if len(np.unique(val)) != 275 or len(train_ids) != 2473 or len(blocks) != 8:
        raise ValueError('Not the fixed Task 1 partition')
    training, rows = {}, []
    for b in sorted(blocks, key=lambda b: b.key):
        s = split.blocks[b.key]
        x, y = b.features[s.train_trials], b.trial_targets[s.train_trials]
        ids, means = aggregate_by_target(x, y)
        if not np.array_equal(ids, train_ids):
            raise ValueError(f'Training IDs differ in {b.key}')
        parts = training.setdefault(b.mouse, ([], []))
        parts[0].extend([means.astype(np.float32), x])
        parts[1].extend([ids, y])
        vy = b.trial_targets[s.validation_trials]
        ids, means = aggregate_by_target(b.features[s.validation_trials], vy)
        if not np.array_equal(ids, np.sort(val)) or not np.all(np.unique(vy, return_counts=True)[1] == 4):
            raise ValueError(f'Unexpected validation rows in {b.key}')
        rows.append((b.mouse, b.key, ids, means.astype(np.float32)))
    return train_ids, val, {m: tuple(map(np.concatenate, parts)) for m, parts in training.items()}, rows


def layout_target(x):
    # For each cell: ink occupancy and three-pixel centered runs along four
    # axes (horizontal, vertical, two diagonals). These are proxies, not IDS.
    p = F.pad(x, (1, 1, 1, 1))
    channels = [x]
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = p[:, :, 1+dy:65+dy, 1+dx:65+dx]
        b = p[:, :, 1-dy:65-dy, 1-dx:65-dx]
        channels.append(a * x * b)
    return F.avg_pool2d(torch.cat(channels, dim=1), 8)


class LayoutTeacher(nn.Sequential):
    def __init__(self):
        # Shared local stroke detector; 8x8 ordered cells = four 4x4 quadrants.
        super().__init__(nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(24, 5, 1), nn.Sigmoid(), nn.AvgPool2d(8))


def fit_teacher(images, device):
    net = LayoutTeacher().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    x = torch.from_numpy(images[:, None]).to(device)
    y = layout_target(x)
    for _ in range(24):
        for i in torch.randperm(len(x)).split(64):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(net(x[i]), y[i]).backward()
            opt.step()
    return net.eval().requires_grad_(False)


@torch.no_grad()
def encode(net, images, device):
    return torch.cat([net(x.to(device)).flatten(1).cpu() for x in
                      torch.from_numpy(images[:, None]).split(64)])


def fit_students(training, train_ids, target, device):
    models = {}
    for mouse, (x, y) in training.items():
        torch.manual_seed(SEED)
        net = SpikeRegressor(x.shape[1], target.shape[1], hidden=512, dropout=0.4)
        net.set_normalization(x)
        net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.01)
        features = torch.from_numpy(x)
        positions = torch.from_numpy(np.searchsorted(train_ids, y))
        for _ in range({'mice1': 8, 'mice2': 11}[mouse]):
            for i in torch.randperm(len(x)).split(256):
                opt.zero_grad(set_to_none=True)
                F.mse_loss(net(features[i].to(device)), target[positions[i]].to(device)).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                opt.step()
        models[mouse] = net.eval()
    return models


def rank(query, bank_vectors, k):
    sim = F.normalize(query.cpu(), dim=1) @ bank_vectors.T
    if not torch.isfinite(sim).all():
        raise ValueError('Nonfinite retrieval score')
    return torch.argsort(sim, descending=True, stable=True)[:, :k].numpy()


def score_records(out, name, pictures, groups, indices, images, bank, device):
    scores = np.asarray(score_predictions(bank[indices], images[pictures], device=device))
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite official OCF')
    np.savez_compressed(out / f'{name}.npz', pictures=pictures, groups=groups,
                        bank_rows=indices, ocf=scores)
    return summarize_scores(scores, groups), scores


@torch.no_grad()
def spike_rows(rows, pictures, models, vectors, device):
    ids, groups, matches = [], [], []
    for mouse, key, targets, counts in rows:
        mask = np.isin(targets, pictures)
        query = models[mouse](torch.from_numpy(counts[mask]).to(device))
        ids.extend(targets[mask]); groups.extend([key] * int(mask.sum()))
        matches.extend(rank(query, vectors, 1)[:, 0])
    return tuple(map(np.asarray, (ids, groups, matches)))


def archived_baseline(part, ids, groups, bank, images):
    # Only the DCT-MSE baseline fields, never experiment 21's candidate scores.
    path = BASE / f'task1_directional_loss_{part}_predictions.npz'
    with np.load(path, allow_pickle=False) as a:
        lookup = dict(zip(zip(a['group_ids'], a['picture_indices']), range(len(ids))))
        if len(lookup) != len(ids) or len(a['picture_indices']) != len(ids):
            raise ValueError('Wrong baseline sample count or duplicate keys')
        order = [lookup[key] for key in zip(groups, ids)]
        idx = a['baseline_candidate_indices'][order]
        if (idx.min() < 0 or idx.max() >= len(bank) or
                not np.array_equal(a['targets'][order], images[ids] * 255) or
                not np.array_equal(a['baseline_predictions'][order], bank[idx] * 255)):
            raise ValueError('Baseline targets/font pixels differ from current run')
    return idx


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    images = (load_images(DATA) >= 128).astype(np.float32)
    if images.shape != (2748, 64, 64):
        raise ValueError('Unexpected training image set')
    train, val, training, rows = prepare()
    tune, diagnostic, held = map(np.sort, (val[:16], val[16:64], val[64:]))
    bank = render_gb2312_bank(FONT, font_size=52, variation='Regular')
    if bank.shape != (6763, 64, 64):
        raise ValueError('Unexpected independent font bank')
    # Fresh directory per invocation: a losing run cannot inherit held-out files.
    from tempfile import mkdtemp
    out = Path(mkdtemp(prefix='run_', dir=HERE))
    print(f'Experiment artifacts: {out}', flush=True)
    teacher = fit_teacher(images[train], device)
    z = encode(teacher, images[train], device)
    center, scale = z.mean(0), z.std(0).clamp_min(0.01)
    vectors = F.normalize((encode(teacher, bank, device) - center) / scale, dim=1)
    # Disjoint image-only diagnostic, one query per character, no spike inputs.
    nearest = rank((encode(teacher, images[diagnostic], device) - center) / scale, vectors, 8)
    diag_scores = np.stack([score_predictions(bank[nearest[:, i]], images[diagnostic], device=device)
                            for i in range(nearest.shape[1])], axis=1)
    np.savez_compressed(out / 'image_diagnostic.npz', pictures=diagnostic,
                        teacher_bank_rows=nearest, ocf=diag_scores)
    models = fit_students(training, train, (z - center) / scale, device)
    torch.save({'teacher': teacher.state_dict(), 'center': center, 'scale': scale,
                'students': {m: net.state_dict() for m, net in models.items()}}, out / 'models.pt')
    ids, groups, idx = spike_rows(rows, tune, models, vectors, device)
    base = archived_baseline('tune', ids, groups, bank, images)
    baseline, bs = score_records(out, 'baseline_tune', ids, groups, base, images, bank, device)
    if abs(baseline['ocf_mean'] - 16.811784552406262) > 1e-5:
        raise ValueError('Archived baseline tune OCF did not reproduce')
    candidate, cs = score_records(out, 'student_tune', ids, groups, idx, images, bank, device)
    wins = candidate['ocf_mean'] > baseline['ocf_mean']
    selected = 'student' if wins else 'baseline'
    report = {'experiment': '27_layout_teacher', 'seed': SEED, 'device': device,
              'torch': str(torch.__version__), 'numpy': np.__version__,
              'font': str(FONT), 'font_sha256': hashlib.sha256(FONT.read_bytes()).hexdigest(),
              'train_ids': train.tolist(), 'tune_ids': tune.tolist(), 'held_out_ids': held.tolist(),
              'diagnostic_ids': diagnostic.tolist(), 'image_teacher_top1_ocf': float(diag_scores[:, 0].mean()),
              'image_teacher_oracle_at_8': float(diag_scores.max(1).mean()),
              'baseline_tune': baseline, 'student_tune': candidate,
              'paired_tune_delta_by_picture': {str(p): float((cs-bs)[ids == p].mean()) for p in tune},
              'selection': 'strictly greater tune spike OCF; ties retain baseline', 'selected': selected,
              'scope': 'Local reused Task1 split; image oracle is limited to eight teacher candidates, not a full-bank OCF ceiling. No identity labels or identity recall.'}
    def persist():
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    persist()  # Lock selection before either branch touches held-out predictions.
    if wins:
        ids, groups, idx = spike_rows(rows, held, models, vectors, device)
    else:
        ids = np.concatenate([targets[np.isin(targets, held)] for _, _, targets, _ in rows])
        groups = np.repeat([key for _, key, _, _ in rows], len(held))
        idx = archived_baseline('held_out', ids, groups, bank, images)
    report['selected_held_out'], _ = score_records(out, f'{selected}_held_out', ids, groups,
                                                    idx, images, bank, device)
    report['student_held_out_evaluated'] = wins
    persist()
    print(f"Selected {selected}; local held-out OCF {report['selected_held_out']['ocf_mean']:.5f}")


if __name__ == '__main__':
    main()
