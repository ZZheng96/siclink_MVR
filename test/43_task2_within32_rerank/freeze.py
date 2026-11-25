import hashlib
import json
from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parent
old = json.loads((out.parent/'42_task2_production_paired/ids.json').read_text())['ids']
prior = np.random.default_rng(123).permutation(2748)[:768]
pool = np.setdiff1d(np.arange(2748), np.r_[prior, old])
assert len(pool) == 1468
ids = np.sort(np.random.default_rng(271828).permutation(pool)[:512]).astype('<i8')
plan = dict(base='0add2d4', scope='Public train only; both mice session4. No hidden data.',
    subset='Exclude experiment 41 tune256 and heldout512, plus experiment 42 ids512. Permute sorted remainder without replacement with numpy.default_rng(271828), take first 512 then sort. Identical IDs for both mice; evaluation only.',
    ids=ids.tolist(), ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
    control='Exact experiment 42 production v5 config and checkpoints. Historical 23.824141 tolerance 1e-5; stop on mismatch.',
    cnn='Identical exp42 per-mouse training: session1-3 pooled per-glyph means; CNN seeds 42,202,777; 12 epochs, batch128, AdamW lr .002 weight_decay .001. Mean predicted rates before Poisson.',
    rerank='Full-row z-scores as in 42. Rank original production top32 by descending 0.8*v5_z + 0.2*Poisson_z; mixed-score ties by ascending glyph ID. Map original descending 32 v5 values onto this order, without epsilon. Other scores untouched. Actual post-remap v5-score ties follow production stable order (ascending glyph ID). Report and stop on set or intended order mismatch; do not repair ties.',
    assignment='Same production unique_assignment on full 2748x2748 per mouse for both arms; no subset assignment.',
    primary='Paired mean final OCF difference (rerank-v5) on frozen IDs by mouse, equal mouse macro. Check all rows for exact candidate membership and record recall32. Only both strictly positive and invariants passing warrants cross-day replication, not production. No further weight trials on this held day.')
with (out/'preregistration.json').open('x', encoding='utf8') as f:
    json.dump(plan, f, indent=2)
print('Frozen ID hash:', plan['ids_sha256'])
