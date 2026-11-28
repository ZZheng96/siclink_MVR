"""Fresh R1-2 training; score same-trial nominations on R3 only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from r4_selective_switch import B, F, scores_and_votes


def collect(out, mouse, centered):
    F.train_dev(out, mouse)
    sessions = []
    for s in B.SESSIONS:
        scores, votes = scores_and_votes(out, mouse, s, 3, centered)
        top = np.argsort(-scores, axis=1, kind='stable')[:, :5].copy()
        sessions.append((top, votes.T, F.labels(mouse, s, 3)))
    return tuple(map(np.concatenate, zip(*sessions)))


def summarize(top, votes, y):
    truth = top == y[:, None]
    covered = np.unique(y[truth[:, 0]])
    vote = votes[:, :, None] == top[:, None, 1:]
    n = vote.sum(1)
    assert [w for _, w, _ in F.RUNS if w != 250] == [10, 25, 50]
    rules = dict(zip(('10ms', '25ms', '50ms'), vote.transpose(1, 0, 2)))
    rules.update(any=n >= 1, ge2=n >= 2)

    def coverage(mask):
        chars = np.unique(y[mask])
        return {'trials': int(mask.sum()), 'unique': len(chars),
                'new_unique': len(np.setdiff1d(chars, covered))}

    result = {'trials': len(y), 'baseline': {'hits': int(truth[:, 0].sum()), 'unique': len(covered)}}
    for rank, sl in [('2-5', slice(None))] + [(str(i+2), slice(i, i+1)) for i in range(4)]:
        t = truth[:, 1:][:, sl]
        row = {'truth': coverage(t.any(1))}
        for key, mask in rules.items():
            observable = mask[:, sl]
            hits = coverage((t & observable).any(1))
            opportunities = int(observable.sum())
            row[key] = {**hits, 'opportunities': opportunities,
                        'nominated_trials': int(observable.any(1).sum()),
                        'precision': hits['trials']/opportunities if opportunities else None}
        result[rank] = row
    return result


def main():
    centered = B.dct_encode(B.IMAGES, 16)
    centered -= centered.mean(0, keepdims=True)
    with TemporaryDirectory() as tmp:
        blocks = {m: collect(Path(tmp), m, centered) for m in B.MICE}
    overall = summarize(*map(np.concatenate, zip(*blocks.values())))
    assert overall['baseline'] == {'hits': 429, 'unique': 353}, overall['baseline']
    result = {'protocol': 'F.train_dev on repeats 1,2. R3 only. Per-session current CSLS100; stable descending score, ties by ascending class ID. Each MLP vote is argmax of mean logits over seeds 42,202,777.',
              'units': 'trials are distinct mouse/session/row hits. unique counts target IDs; new_unique excludes baseline top1 coverage within the same reporting scope. Opportunities count (trial,candidate) pairs over ALL trials, with the named branch, any branch, or >=2 branches nominating that candidate. Multiple nominators of one candidate count once per rule. nominated_trials collapses different candidates on the same trial. precision = correct pairs / opportunities; each trial has one truth.',
              'overall': overall, 'by_mouse': {m: summarize(*v) for m, v in blocks.items()}}
    dest = F.HERE / 'R3_SAME_TRIAL_ORACLE.json'
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for m, stats in [('overall', overall), *result['by_mouse'].items()]:
        print(m, stats['baseline'], json.dumps(stats['2-5']))
    for rank in '2345':
        print(rank, json.dumps(overall[rank]))
    print(dest)


if __name__ == '__main__':
    main()
