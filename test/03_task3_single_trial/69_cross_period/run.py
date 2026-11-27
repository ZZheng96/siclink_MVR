"""Run with the repository CUDA interpreter, python -B run.py."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.functional import cross_entropy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from src.features import trial_count_features
from src.neural import SpikeClassifier

TRAIN = ROOT / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset/train'
DER = ROOT / 'data/derived/repeat_split_v1'
HASH = '7c264d473f10a1b49ea7392f794f9f0f7bf7e19d9e330d378163375b0720b630'
VIEWS = (10, 25, 50)
MICE = ('mice1', 'mice2')
ARMS = ('25ms', '50ms', 'fusion')
C = 2748


def check(value, msg):
    if not value:
        raise ValueError(msg)


def load():
    data, seen = {}, set()
    for mouse, units in zip(MICE, (134, 78)):
        data[mouse] = []
        for s in range(1, 5):
            with np.load(TRAIN / mouse / f'session{s}.npz') as f:
                y, ids = f['target'], f['sample_ids']
                np.testing.assert_array_equal(f['group_ptr'], np.arange(4*C+1))
                b = SimpleNamespace(**{k:f[k] for k in ('trial_ptr','spike_samples','unit_index')}, n_units=units, sample_rate_hz=30000, window_ms=250)
            check(len(y)==len(ids)==4*C and len(set(ids))==4*C and not seen.intersection(ids), 'duplicate or missing IDs')
            seen.update(ids)
            check(np.all((0<=y)&(y<C)), 'target range')
            counts = np.zeros(C, int)
            r = np.empty(len(y), int)
            for i,t in enumerate(y):
                counts[t] += 1
                r[i] = counts[t]
            check((counts==4).all(), 'target multiplicity')
            for k in range(1,5):
                with np.load(DER / mouse / f'session{s}_repeat{k}.npz') as f:
                    ix = f['source_trial_index']
                    np.testing.assert_array_equal(np.sort(ix), np.flatnonzero(r==k))
                    np.testing.assert_array_equal(f['target'], y[ix])
                    np.testing.assert_array_equal(f['sample_ids'], ids[ix])
                    np.testing.assert_array_equal(f['repeat_index'], np.full(C,k))
            check(b.unit_index.min()>=0 and b.unit_index.max()<units and len(b.trial_ptr)==len(y)+1, 'source spike dimensions')
            feats = {v:np.log1p(trial_count_features(b, v)) for v in VIEWS}
            data[mouse].append((y,r,ids,feats))
    return data


def prepared(blocks, v, k):
    parts = []
    for _, r, _, feats in blocks:
        x = feats[v]
        train = x[r!=k]
        mu = train.mean(0,dtype=np.float64).astype('f')
        sd = train.std(0,dtype=np.float64).astype('f').clip(1e-5)
        parts.append((x-mu)/sd)
    return np.concatenate(parts)


@torch.no_grad()
def scores(model, x):
    model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return torch.cat([model(a) for a in x.split(256)])


def fit(x, y, train):
    torch.manual_seed(42)
    model = SpikeClassifier(x.shape[1], C, 1024, 0., False)
    model.set_normalization(x[train])
    model.cuda()
    x = torch.from_numpy(x).cuda()
    xt, yt = x[train], y[train]
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.,fused=True)
    for epoch in range(1,101):
        model.train()
        order = torch.from_numpy(np.random.default_rng(42+epoch).permutation(len(yt))).cuda()
        for ix in order.split(256):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = cross_entropy(model(xt[ix]).float(),yt[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
        n = int((scores(model,xt).argmax(1)==yt).sum())
        if n==len(yt):
            break
    return scores(model,x).float(), epoch, n


def metric(logits,y,ix):
    return dict(correct=int((logits[ix].argmax(1)==y[ix]).sum()),total=len(ix))


def pooled(rows, split):
    return {a:{m:sum(row['arms'][a][split][m] for row in rows) for m in ('correct','total')} for a in ARMS}


def main():
    digest = hashlib.sha256((OUT/'preregistration.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    check(digest==HASH or hashlib.sha256((OUT/'preregistration.json').read_bytes()).hexdigest()==HASH,'preregistration changed')
    data = load()
    result = dict(preregistration_sha256=HASH, torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(), audit='Verified 87936 globally unique source IDs and all 32 derived target/ID/chronological mappings',rows=[],folds={})
    for k in range(5):
        rows = []
        for mouse in MICE:
            blocks = data[mouse]
            r = np.concatenate([b[1] for b in blocks]); ids=np.concatenate([b[2] for b in blocks])
            tr, te = np.flatnonzero(r!=k),np.flatnonzero(r==k)
            check(not set(ids[tr]).intersection(ids[te]) and len(te)==(4*C if k else 0),'split')
            y = torch.from_numpy(np.concatenate([b[0] for b in blocks])).long().cuda()
            row = dict(fold=k or 'all',mouse=mouse,branches={},arms={}); fusion = None
            for v in VIEWS:
                x = prepared(blocks,v,k)
                logits,epoch,n = fit(x,y,tr)
                del x
                check(torch.isfinite(logits).all().item(),'nonfinite scores')
                row['branches'][str(v)]=dict(epoch=epoch,train_correct=n,train_total=len(tr))
                parts = {'train':tr, **({'held':te} if k else {})}
                if v in (25,50):
                    row['arms'][f'{v}ms']={s:metric(logits,y,i) for s,i in parts.items()}
                z = (logits-logits.mean(1,keepdim=True))/logits.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)
                fusion = z if fusion is None else fusion.add_(z)
                del logits,z
                print(k,mouse,v,epoch,n,len(tr),flush=True)
            fusion.div_(3)
            row['arms']['fusion']={s:metric(fusion,y,i) for s,i in parts.items()}
            del fusion
            rows.append(row)
        result['rows'].extend(rows)
        p = pooled(rows,'held' if k else 'train'); result['folds'][k or 'all']=p
        check(all(m['total']==(21984 if k else 87936) for m in p.values()),'fold denominator')
        print(k,p,flush=True)
        if not k and not all(m['correct']==87936 for m in p.values()):
            result['status']='capacity_gate_failed'; break
    else:
        result['status']='completed'
        result['held_pooled']=pooled(result['rows'][2:],'held')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['status'],result.get('held_pooled'),flush=True)


if __name__=='__main__':
    main()
