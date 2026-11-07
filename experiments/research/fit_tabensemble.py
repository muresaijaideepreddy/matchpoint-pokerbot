"""Train one poker-specific TabM-style batch ensemble. No external architecture code or serving dependency."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/"tools")]
from fit_neural import action_support
from game import Action, Observation
from neural_policy import NeuralPolicy
from tabensemble_policy import TabEnsemble, public_features, encode

NUMERIC = (12,13,14,16,17,18,19,20,21,22,27)
MODEL_SEED = 11883


def load_rows(source, independent=False):
    buckets = [(list(),list(),list()),(list(),list(),list())]
    gids = [{},{}]
    for path in sorted(source.rglob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                item = json.loads(line)
                deal = item["deal_id"]
                split = int(independent or zlib.crc32(f"7721/{deal}".encode()) < .2*2**32)
                obs = Observation.from_dict(item["features"])
                base = Action(item["base"])
                obs_input = public_features(obs,base,item["equity"],item["aggression"],item["fold_rate"],item["key"])
                rewards = {Action(int(k)):v for k,v in item["returns"].items()}
                assert set(rewards)==set(obs.legal_actions) and base in rewards
                gains = [rewards[a]-rewards[base] if a in rewards else np.nan for a in Action]
                x,y,g = buckets[split]
                x.append(obs_input)
                y.append(gains)
                g.append(gids[split].setdefault(deal,len(gids[split])))
    return [(np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),np.asarray(g,dtype=np.int64)) for x,y,g in buckets],gids


def train_knots(x):
    index, lower, inverse = [], [], []
    for col in NUMERIC:
        levels = np.unique(np.quantile(x[:,col],np.linspace(0.,1.,9)).astype(np.float32))
        for a,b in zip(levels,levels[1:]):
            if b-a > 1.e-6:
                index.append(col)
                lower.append(a)
                inverse.append(1./(b-a))
    return np.asarray(index,np.int64),np.asarray(lower,np.float32),np.asarray(inverse,np.float32)


class MemberLinear(nn.Module):
    def __init__(self, inp, out, first):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(inp,out))
        nn.init.uniform_(self.weight,-1/math.sqrt(inp),1/math.sqrt(inp))
        self.input_scale=nn.Parameter(torch.ones(4,inp))
        self.output_scale=nn.Parameter(torch.ones(4,out))
        self.bias=nn.Parameter(torch.zeros(4,out))
        with torch.no_grad():
            if first:
                self.input_scale.bernoulli_(.5).mul_(2).sub_(1)
            else:
                self.input_scale.normal_(1.,.02)
            self.output_scale.normal_(1.,.02)

    def forward(self, h):
        return ((h*self.input_scale) @ self.weight)*self.output_scale + self.bias


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l0=MemberLinear(width,64,True)
        self.l1=MemberLinear(64,64,False)
        self.w=nn.Parameter(torch.empty(4,64,4))
        self.b=nn.Parameter(torch.zeros(4,4))
        self.drop=nn.Dropout(.05)
        nn.init.uniform_(self.w,-.125,.125)

    def forward(self,x):
        h=self.drop(torch.relu(self.l0(x[:,None,:])))
        h=self.drop(torch.relu(self.l1(h)))
        return torch.einsum("nhk,hka->nha",h,self.w)+self.b


def as_device(x,device):
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def per_head_gains(scores, base):
    return scores - scores.gather(2,base[:,None,None].expand(-1,4,1))


@torch.inference_mode()
def rmse_cuda(net,x,y,base):
    net.eval()
    total = torch.zeros((),device=x.device,dtype=torch.float64)
    count = torch.zeros((),device=x.device,dtype=torch.int64)
    for i in range(0,len(x),4096):
        preds=per_head_gains(net(x[i:i+4096]),base[i:i+4096]).mean(1)*40
        labels=y[i:i+4096]
        valid=labels.isfinite()
        total+=torch.where(valid,preds-labels,0).double().square().sum()
        count+=valid.sum()
    return (total/count).sqrt().item()


def export(net,x_test,knots,supported,out):
    def cpu(a):
        return a.detach().cpu().numpy().copy()
    params=dict(tabensemble_format=1,index=knots[0],lower=knots[1],inverse_width=knots[2],supported=supported,margin=np.float64(2.))
    for i,layer in enumerate((net.l0,net.l1)):
        params[f"w{i}"]=cpu(layer.input_scale[:,:,None]*layer.weight[None,:,:]*layer.output_scale[:,None,:])
        params[f"b{i}"]=cpu(layer.bias)
    params.update(w2=cpu(net.w)*40,b2=cpu(net.b)*40)
    path=out/"policy.npz"
    temp=out/"policy.tmp"
    with temp.open("wb") as f:
        np.savez_compressed(f,**params)
    policy=TabEnsemble.load(temp)
    sample=x_test[:2048]
    with torch.no_grad():
        target=cpu(net(as_device(encode(sample,*knots),next(net.parameters()).device)).mean(dim=1))*40
    got=policy.predict(sample)
    np.testing.assert_allclose(got,target,atol=2.e-4,rtol=1.e-4)
    e=float(np.abs(got-target).max())
    temp.replace(path)
    return policy,e


def compare(policy,x,y,g):
    base=x[:,4:8].argmax(1)
    mask=np.isfinite(y)
    def error(p,xx):
        scores=np.vstack([p.predict(xx[i:i+2048]) for i in range(0,len(x),2048)])
        gains=scores-scores[np.arange(len(x)),base,None]
        sq=np.square(np.where(mask,gains-y,0.).astype(np.float64))
        return np.bincount(g,weights=sq.sum(axis=1)),scores
    candidate_squares,scores=error(policy,x)
    reference=NeuralPolicy.load(PROJECT/"models/swirl_compact96.npz")
    baseline_squares,_=error(reference,np.ascontiguousarray(x[:,:28]))
    denominator=np.bincount(g,weights=mask.sum(axis=1))
    mse_ref=np.sqrt(baseline_squares.sum()/denominator.sum())
    mse_cand=np.sqrt(candidate_squares.sum()/denominator.sum())
    chosen=policy.actions(x,scores)
    payoff=y[np.arange(len(x)),chosen]
    assert np.isfinite(payoff).all()
    rng=np.random.default_rng(50907721)
    samples=[]
    for _ in range(125):
        ids=rng.integers(0,len(denominator),size=(8,len(denominator)))
        n=denominator[ids].sum(axis=1)
        samples.extend(np.sqrt(candidate_squares[ids].sum(axis=1)/n)-np.sqrt(baseline_squares[ids].sum(axis=1)/n))
    return dict(rows=len(x),legal_targets=int(mask.sum()),deals=len(denominator),compact96_rmse=float(mse_ref),
                tabensemble_rmse=float(mse_cand),delta=float(mse_cand-mse_ref),
                paired_deal_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),
                changes=int((chosen!=base).sum()),replay_gain_chips=float(payoff.mean(dtype=np.float64)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--test-replay",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=45)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    device=args.device
    if args.epochs < 1:
        p.error("epochs should be positive")
    out=args.output_dir
    if (out/"policy.npz").exists():
        p.error("Output checkpoint already exists")
    out.mkdir(parents=True,exist_ok=True)
    if device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu")
    torch.manual_seed(MODEL_SEED)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    (train,hold),seen=load_rows(args.replay)
    x,y,_=train
    vx,vy,vg=hold
    if min(len(x),len(vx)) == 0:
        p.error("Supply replay shards containing distinct training and held-out deals")
    knots=train_knots(x)
    print(f"Training on {device}: {len(x)} training rows, {len(vx)} held-out rows. Fitted {len(knots[0])} numeric intervals on training rows.",flush=True)
    xg,yg,bg=as_device(encode(x,*knots),device),as_device(np.nan_to_num(y/40),device),as_device(x[:,4:8].argmax(1),device)
    legal=as_device(np.isfinite(y),device)
    vxg,vyg,vbg=as_device(encode(vx,*knots),device),as_device(vy,device),as_device(vx[:,4:8].argmax(1),device)
    net=Network(xg.shape[1]).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    best=math.inf
    chosen_epoch=0
    best_state=None
    for epoch in range(1,args.epochs+1):
        net.train()
        for ids in torch.randperm(len(x),device=device).split(2048):
            target=yg[ids,None,:]
            allowed=legal[ids,None,:]
            predicted=per_head_gains(net(xg[ids]),bg[ids])
            loss=((predicted-target).square()*allowed).sum()/(4*allowed.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        value=rmse_cuda(net,vxg,vyg,vbg)
        print(f"Epoch {epoch}/{args.epochs} validation {value:.7f} chips",flush=True)
        if value<best:
            best=value
            chosen_epoch=epoch
            best_state=copy.deepcopy(net.state_dict())
        if epoch-chosen_epoch>=6:
            break
    net.load_state_dict(best_state)
    net.eval()
    policy,err=export(net,vx,knots,action_support(x,y,64),out)
    print(f"Frozen epoch {chosen_epoch}, {min(2048,len(vx))} NumPy parity rows, max error {err:.8f} chips.",flush=True)
    valid=compare(policy,vx,vy,vg)
    assert abs(valid["tabensemble_rmse"]-best)<2.e-4
    (out/"validation.json").write_text(json.dumps(valid,indent=2))
    print("Validation",json.dumps(valid),flush=True)
    test=None
    if args.test_replay:
        locked,independent=load_rows(args.test_replay,True)
        if not independent[1].keys().isdisjoint(seen[0]) or not independent[1].keys().isdisjoint(seen[1]):
            raise ValueError("Independent replay overlaps training or validation deal IDs")
        tx,ty,tg=locked[1]
        if not len(tx):
            raise ValueError("No independent replay data")
        test=compare(policy,tx,ty,tg)
        (out/"independent.json").write_text(json.dumps(test,indent=2))
    summary=dict(architecture="four related two-layer 64-wide MLPs, independent outputs",
                 original_action_margin=2,model_seed=MODEL_SEED,split_seed=7721,epochs_ran=epoch,chosen_epoch=chosen_epoch,
                 training_rows=len(x),train_only_quantile_intervals=len(knots[0]),numpy_parity_rows=min(2048,len(vx)),
                 max_export_error_chips=err,checkpoint_bytes=(out/"policy.npz").stat().st_size,
                 sha256=hashlib.sha256((out/"policy.npz").read_bytes()).hexdigest(),
                 validation=valid,untouched=test,
                 references=["https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html","https://github.com/yandex-research/tabm"])
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("Independent",json.dumps(test),flush=True)


if __name__=="__main__":
    main()
