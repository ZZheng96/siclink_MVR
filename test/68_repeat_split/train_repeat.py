"""Repeat-holdout v1. CLI: train | final --epoch-B 4 [--epoch-C 8].
Scoring: load_data -> (log1p_features, target, session, source_trial_index);
load_scaler -> (mean, scale); load_model -> eval Decoder(logits, sigmoid_pixels_or_None).
All loaders use integer mouse IDs. TRAIN_DONE.json / FINAL_DONE.json seal a run.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / 'data/derived/repeat_split_v1'
RUN = ROOT / 'data/derived/repeat_run_v1'
PICTURES = ROOT / 'data/siclink-mouse-visual-reconstruction-bci-challenge/siclink_chinese_character_decoding_dataset/train/pictures.npz'
REPEATS = {('F1','train'):(1,), ('F1','val'):(2,), ('F2','train'):(1,2), ('F2','val'):(3,), ('FINAL','train'):(1,2,3)}
UNITS = {1:134, 2:78}
SEEDS = (42,202,777)
EPOCHS = (4,8,12)
K = 2748


def sha(path):
    with path.open('rb') as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
        return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_file(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def features(mouse, trial_ptr, spike_samples, unit_index):
    """Blind event-only featurizer, retaining empty trials."""
    p,t,u = trial_ptr,spike_samples,unit_index
    n = UNITS[mouse]
    if not all(a.ndim == 1 and a.dtype.kind in 'iu' for a in (p,t,u)) or not len(p) or p[0] != 0 or p[-1] != len(t) or len(t) != len(u) or np.any(p[1:] < p[:-1]):
        raise ValueError('invalid event pointers/arrays')
    if np.any((t<0)|(t>=7500)) or np.any((u<0)|(u>=n)):
        raise ValueError('event outside frozen time/unit bounds')
    row = np.repeat(np.arange(len(p)-1, dtype=np.int64), np.diff(p.astype(np.int64)))
    counts = np.bincount((row*10+t//750)*n+u, minlength=(len(p)-1)*10*n)
    return np.log1p(counts.astype(np.float32)).reshape(len(p)-1,10*n)


def paths(fold,mouse,phase):
    if mouse not in UNITS:
        raise ValueError(mouse)
    return [(s,r,SPLIT/f'mice{mouse}'/f'session{s}_repeat{r}.npz') for s in range(1,5) for r in REPEATS[fold,phase]]


def load_data(fold,mouse,phase):
    chunks = []
    for s,r,path in paths(fold,mouse,phase):
        with np.load(path,allow_pickle=False) as z:
            y,src,rep,g,p = (z[k] for k in ('target','source_trial_index','repeat_index','group_ptr','trial_ptr'))
            if not all(a.dtype.kind in 'iu' for a in (y,src,rep,g)) or not all(a.shape==(K,) for a in (y,src,rep)) or not np.array_equal(np.sort(y),np.arange(K)) or not np.all(rep==r) or not np.array_equal(g,np.arange(K+1)) or p.shape != (K+1,) or np.any((src<0)|(src>=4*K)):
                raise ValueError(f'invalid sample metadata: {path}')
            x = features(mouse,p,z['spike_samples'],z['unit_index'])
        chunks.append((x,y,np.full(K,s,dtype=np.int64),src))
    x,y,s,src = map(np.concatenate,zip(*chunks))
    if len(np.unique(s*(4*K)+src)) != len(src):
        raise ValueError('duplicate source trial')
    return x,y,s,src


class Decoder(nn.Module):
    def __init__(self,mouse,model):
        super().__init__()
        if model not in ('B','C'):
            raise ValueError(model)
        self.hidden = nn.Sequential(nn.Linear(10*UNITS[mouse],512),nn.ReLU(),nn.Dropout(0.2))
        self.cls = nn.Linear(512,K)
        self.reg = nn.Linear(512,4096) if model=='C' else None

    def forward(self,x):
        h = self.hidden(x)
        return self.cls(h), (self.reg(h).sigmoid() if self.reg is not None else None)


def directory(fold,mouse):
    return RUN/fold/f'mice{mouse}'


def load_scaler(fold,mouse):
    with np.load(directory(fold,mouse)/'scaler.npz',allow_pickle=False) as z:
        return z['mean'],z['scale']


def checkpoint(fold,mouse,model,seed,epoch):
    return directory(fold,mouse)/model/f'seed{seed}'/f'epoch{epoch}.pt'


def load_model(fold,mouse,model,seed,epoch,device='cpu'):
    net = Decoder(mouse,model)
    ck = torch.load(checkpoint(fold,mouse,model,seed,epoch),map_location='cpu',weights_only=True)
    net.load_state_dict(ck['state_dict'])
    return net.to(device).eval()


def fit(fold,mouse,model,seed,epochs,x,y,library,scaler_hash):
    start = time.monotonic()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = torch.Generator().manual_seed(seed)
    net = Decoder(mouse,model).to(x.device)
    opt = torch.optim.AdamW(net.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    records = []
    for epoch in range(1,max(epochs)+1):
        losses = []
        for idx in torch.randperm(len(x),generator=rng).to(x.device).split(256):
            logits,p = net(x[idx]); labels = y[idx]
            loss = F.cross_entropy(logits,labels)
            if p is not None:
                loss = loss + F.mse_loss(p,library[labels])
            losses.append(loss.detach()); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if not torch.stack(losses).isfinite().all().item():
            raise FloatingPointError(f'{fold}/{mouse}/{model}/{seed}/{epoch} loss')
        if epoch in epochs:
            dest = checkpoint(fold,mouse,model,seed,epoch)
            dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(),fold=fold,mouse=mouse,model=model,seed=seed,epoch=epoch,scaler_sha256=scaler_hash),dest)
            records.append(dict(path=str(dest.relative_to(RUN)),sha256=sha(dest),elapsed=time.monotonic()-start,utc=now()))
    print(f'{fold} mice{mouse} {model} seed{seed} epoch{max(epochs)} saved',flush=True)
    return records


def run(folds,recipes,stage):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUN.mkdir(parents=True,exist_ok=True)
    for f in folds:
        dest = RUN/f
        dest.mkdir(exist_ok=True)
        if any(dest.iterdir()):
            raise FileExistsError(f'refusing to overwrite {dest}')
    meta = dict(protocol='repeat-holdout v1',start=now(),device=torch.cuda.get_device_name() if device=='cuda' else device,
                python=sys.version,os=platform.platform(),torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),cublas_workspace=':4096:8',deterministic=True,tf32=False,benchmark=False,
                script_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('training_protocol.md')),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                optimizer='AdamW(lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.01)',batch=256,workers=0,
                pictures_sha256=sha(PICTURES),seeds=SEEDS,recipes=recipes,groups=[])
    with np.load(PICTURES,allow_pickle=False) as z:
        images=z['images']
    if images.shape!=(K,64,64) or images.dtype!=np.uint8:
        raise ValueError('invalid library')
    library=torch.from_numpy((images>=128).astype(np.float32).reshape(K,-1)).to(device)
    for fold in folds:
        for mouse in UNITS:
            x,y,s,src = load_data(fold,mouse,'train')
            val_rows=0
            if fold!='FINAL':
                vx,vy,vs,vi=load_data(fold,mouse,'val'); val_rows=len(vy); del vx
                if np.intersect1d(s*(4*K)+src,vs*(4*K)+vi).size:
                    raise ValueError(f'{fold} mice{mouse}: source leakage')
            mean=x.mean(axis=0,dtype=np.float64).astype(np.float32)
            scale=x.std(axis=0,dtype=np.float64).astype(np.float32); scale[scale==0]=1
            d=directory(fold,mouse); d.mkdir(); np.savez(d/'scaler.npz',mean=mean,scale=scale)
            sc=sha(d/'scaler.npz'); x-=mean; x/=scale
            xt=torch.from_numpy(x).to(device); yt=torch.from_numpy(y).to(device); del x
            group=dict(fold=fold,mouse=mouse,fit_rows=len(y),val_rows=val_rows,scaler_sha256=sc,
                       input_files={str(p.relative_to(ROOT)):sha(p) for phase in ('train','val') if (fold,phase) in REPEATS for _,_,p in paths(fold,mouse,phase)},checkpoints=[])
            meta['groups'].append(group)
            for model,epochs in recipes.items():
                for seed in SEEDS:
                    group['checkpoints'].extend(fit(fold,mouse,model,seed,epochs,xt,yt,library,sc))
                    json_file(RUN/f'{stage}_progress.json',meta)
            del xt,yt
    meta['completed']=now()
    json_file(RUN/f'{stage}_DONE.json',meta)
    print(f'{stage}_DONE',flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__); sub=cli.add_subparsers(dest='stage',required=True)
    sub.add_parser('train'); final=sub.add_parser('final')
    final.add_argument('--epoch-B',type=int,choices=EPOCHS,required=True); final.add_argument('--epoch-C',type=int,choices=EPOCHS)
    args=cli.parse_args()
    if args.stage=='train':
        run(('F1','F2'),{'B':EPOCHS,'C':EPOCHS},'TRAIN')
    else:
        recipe={'B':(args.epoch_B,)}
        if args.epoch_C: recipe['C']=(args.epoch_C,)
        run(('FINAL',),recipe,'FINAL')
