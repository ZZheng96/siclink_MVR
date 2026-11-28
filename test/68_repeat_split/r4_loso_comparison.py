"""Fixed-recipe LOSO comparison. Run with python -B to avoid bytecode files."""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
import r4_temporal_code as ref
from src.neural import SpikeClassifier, fit_classifier_fixed_epochs

METRICS = ('top1', 'recall32', 'nll')
LINEAR = dict(epochs=20, batch=512, lr=0.001, wd=0.1, seed=42)
MLP = dict(epochs=7, batch_size=256, learning_rate=0.0003, weight_decay=0.001, seed=42)


def score(model, x, y, device):
    # Linear/LayerNorm/SiLU and disabled Dropout act independently on each row.
    scores = ref.metrics(ref.predict(model, x, device), y)
    return {k: scores[k] for k in METRICS}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old = json.loads((ref.HERE / 'TEMPORAL_RESULTS_FORWARD.json').read_text())
    rows, forward, sensitivity = [], [], []
    for mouse in ref.MICE:
        for day in ref.SESSIONS:
            days = [s for s in ref.SESSIONS if s != day]
            for ms in (250, 50):
                feature = f'rate{ms}'
                blocks = [ref.load_counts(mouse, s, r, ms) for s in days for r in (1, 2, 3)]
                train = ref.features(np.concatenate([b[0] for b in blocks]), feature)
                labels = np.concatenate([b[1] for b in blocks])
                counts, y = ref.load_counts(mouse, day, 4, ms)
                train, test = ref.standardize(train, ref.features(counts, feature))
                for kind in (('linear',) if ms == 250 else ('linear', 'mlp')):
                    if kind == 'linear':
                        model = ref.train_linear(train, labels, device=device, **LINEAR)
                    else:
                        torch.manual_seed(MLP['seed'])
                        net = SpikeClassifier(train.shape[1], ref.K, 512, 0.2, apply_log=False)
                        # Train only the network: inputs already have the shared fold normalization.
                        net.set_normalization = lambda _: None
                        model, _ = fit_classifier_fixed_epochs(net, (train, labels), device=device, **MLP)
                    m = score(model, test, y, device)
                    row = dict(mouse=mouse, held_day=day, train_days=days, model=f'{ms}ms_{kind}',
                               train_n=len(labels), test_n=len(y), **m)
                    rows.append(row)
                    print(row, flush=True)
                    if day == 4 and kind == 'linear':
                        prev = old[f'{mouse}/forward/{feature}']
                        forward.append(dict(mouse=mouse, model=row['model'], original={k: prev[k] for k in METRICS},
                                            loso_minus_original={k: m[k]-prev[k] for k in METRICS}))
                    if mouse == 'mice2' and ms == 250:
                        masked = test.copy()
                        masked[:, 56] = 0
                        alt = score(model, masked, y, device)
                        sensitivity.append(dict(held_day=day, masked=alt, masked_minus_main={k: alt[k]-m[k] for k in METRICS}))
                        print('sensitivity:', sensitivity[-1], flush=True)
        ref.CACHE.clear()
    macro = {name: {k: float(np.mean([r[k] for r in rows if r['model'] == name])) for k in METRICS}
             for name in ('250ms_linear', '50ms_linear', '50ms_mlp')}
    report = dict(protocol='Other three days R1-3; log1p then pooled per-feature training z-score, zero SD -> 1. Held day R4 row-independent logits, no adaptation or assignment.',
                  torch_version=torch.__version__, device=str(device), linear=dict(optimizer='AdamW', **LINEAR),
                  mlp=dict(architecture='512 LayerNorm SiLU Dropout -> 512 SiLU Dropout -> 2748', dropout=0.2,
                           label_smoothing=0.05, grad_clip=5, optimizer='AdamW', **MLP),
                  differences_from_v5='Single seed, shared pooled training normalization rather than session normalization; forward-compatible bin-major features, raw standalone MLP logits in FP32 at inference, no encoding/DCT fusion or assignment.',
                  results=rows, eight_fold_macro=macro, day4_forward_reference='TEMPORAL_RESULTS_FORWARD.json; linear only, no original MLP',
                  day4_forward_comparison=forward, exploratory_mice2_250ms_col56=dict(
                      intervention='Zero standardized feature index 56 only at inference, same trained weights; prompted by R1-3 day4 mean shift. Excluded from model comparison/selection.', results=sensitivity))
    print('macro:', macro, '\nforward:', forward, flush=True)
    (ref.HERE / 'LOSO_COMPARISON.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
