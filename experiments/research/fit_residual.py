"""Fit the MatchPoint chip-valued residual network on official-format JSONL replays."""
import argparse
from array import array
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from game import Action, Observation
from tools.fit_neural import action_support
from residual_value import CHIPS, INPUTS, ResidualValue, encode


def read_replay(ordinary, train_only):
    buffers = [(array("f"),array("f")) for _ in range(2)]
    original_deals = set()
    files = set()
    for destinations, always_train in ((ordinary,False),(train_only,True)):
        current = sorted({p.resolve() for folder in destinations for p in folder.rglob("*.jsonl")})
        if files.intersection(current):
            raise ValueError("A source shard was supplied twice")
        files.update(current)
        for p in current:
            with p.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    deal = row["deal_id"]
                    if always_train and deal in original_deals:
                        raise ValueError("Train-only replay overlaps the original deal IDs")
                    if not always_train:
                        original_deals.add(deal)
                    held = not always_train and zlib.crc32(f"7721/{deal}".encode()) < .2*2**32
                    obs = Observation.from_dict(row["features"])
                    base = Action(row["base"])
                    returns = {Action(int(k)):float(v) for k,v in row["returns"].items()}
                    if set(returns) != set(obs.legal_actions) or base not in returns or not np.isfinite(list(returns.values())).all():
                        raise ValueError(f"Invalid legal action returns near {deal}")
                    data, labels = buffers[held]
                    data.extend(encode(obs,base,row["equity"],row["aggression"],row["fold_rate"]))
                    labels.extend(returns[a]-returns[base] if a in returns else np.nan for a in Action)
    if not all(len(x) for x,_ in buffers):
        raise ValueError("Supply populated replay shards with distinct training and validation deals")
    return [(np.frombuffer(x,np.float32).reshape(-1,INPUTS), np.frombuffer(y,np.float32).reshape(-1,4))
            for x,y in buffers]


class Net(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(INPUTS,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,4))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self,x):
        return self.body(x)+x[:,-4:]


def squared(net,x,y):
    values=net(x)
    gains=values-(values*x[:,4:8]).sum(1,keepdim=True)
    legal=y.isfinite()
    return ((gains-y.nan_to_num()/CHIPS).square()*legal).sum(),legal.sum()


def export(net, x, support, width, path):
    weights={}
    for i,layer in enumerate((net.body[0],net.body[2],net.body[4])):
        scale=CHIPS if i==2 else 1.
        weights[f"w{2*i}"]=np.ascontiguousarray(layer.weight.detach().cpu().numpy().T*scale)
        weights[f"w{2*i+1}"]=np.ascontiguousarray(layer.bias.detach().cpu().numpy()*scale)
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,residual_format=1,inputs=INPUTS,width=width,supported=support,margin=2.0,**weights)
    policy=ResidualValue.load(tmp)
    with torch.inference_mode():
        original=net(x[:2048]).cpu().numpy()*CHIPS
    exported=policy.predict(x[:2048].cpu().numpy())
    np.testing.assert_allclose(original,exported,atol=2e-4,rtol=1e-4)
    error=float(np.abs(original-exported).max())
    print(f"Verified NumPy export on {len(exported)} rows; max difference {error:.6g} chips.",flush=True)
    tmp.replace(path)
    return error


def train(train_arrays, validation_arrays, dest, width, device, epochs):
    torch.manual_seed(7721)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    tx,ty=train_arrays
    vx,vy=validation_arrays
    support=action_support(tx,ty,64)
    train_x,train_y,val_x,val_y=[torch.from_numpy(a).to(device) for a in (tx,ty,vx,vy)]
    print(f"Training on {device}: {len(tx)} training rows, {len(vx)} held-out rows.",flush=True)
    net=Net(width).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.02)
    best=float("inf")
    for epoch in range(1, epochs + 1):
        sums=torch.zeros((),device=device); counts=torch.zeros_like(sums)
        for ids in torch.randperm(len(tx),device=device).split(1024):
            error,legal=squared(net,train_x[ids],train_y[ids])
            optim.zero_grad(set_to_none=True)
            (error/legal).backward()
            optim.step()
            sums+=error.detach();counts+=legal
        training=float((sums/counts).sqrt())*CHIPS
        with torch.inference_mode():
            metric,legal=squared(net,val_x,val_y)
            validation=float((metric/legal).sqrt())*CHIPS
        print(f"Epoch {epoch}/{epochs}  train action gain RMSE {training:.3f} chips  validation {validation:.3f}",flush=True)
        if validation<best:
            best,best_epoch=validation,epoch
            weights={k:v.detach().clone() for k,v in net.state_dict().items()}
        if epoch-best_epoch>=6:break
    net.load_state_dict(weights)
    print(f"Exporting epoch {best_epoch} on the grouped validation split.",flush=True)
    parity=export(net,val_x,support,width,dest/"policy.npz")
    return {"width":width,"inputs":INPUTS,"training_rows":len(tx),"validation_rows":len(vx),
            "best_epoch":best_epoch,"epochs_run":epoch,"validation_rmse_chips":best,
            "numpy_export_max_error_chips":parity}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",type=Path,action="append",required=True)
    p.add_argument("--train-only",type=Path,action="append",default=[])
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--width",type=int,choices=(64,96),default=96)
    p.add_argument("--epochs",type=int,default=48)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    if args.epochs < 1:
        p.error("epochs should be positive")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu")
    if (args.output_dir/"policy.npz").exists():
        p.error("Output checkpoint already exists")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    train_set,validation=read_replay(args.replay,args.train_only)
    report=train(train_set,validation,args.output_dir,args.width,args.device,args.epochs)
    (args.output_dir/"training.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
