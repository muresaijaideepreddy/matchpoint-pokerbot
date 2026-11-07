"""Learn action values above a known fold payoff using absolute and relative targets."""
import argparse
from array import array
import json
from pathlib import Path
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import Action, Observation, STARTING_STACK
from neural_policy import NeuralPolicy, features
from rl_policy import state_key

STYLES = ("passive", "aggressive", "folding", "mixed")


def encode(obs, base, equity, aggression, fold, key):
    raw = features(obs, base, equity, aggression, fold, detailed=True)
    style = key.split("/", 1)[0]
    return np.concatenate((raw[:28], raw[100:], np.array([s == style for s in STYLES], dtype=np.float32)))


class AnchoredPolicy(NeuralPolicy):
    def __init__(self, path, response_rates=None):
        with np.load(path, allow_pickle=False) as f:
            if str(f["architecture"]) != "fold-anchor-v1":
                raise ValueError("Unknown architecture")
            self.params = [f[f"p{i}"] for i in range(6)]
            self.supported = f["supported"]
            self.margin = float(f["margin"])
        self.response_rates = response_rates

    def predict(self, x):
        w0, b0, w1, b1, w2, b2 = self.params
        h = np.maximum(0, x@w0+b0)
        h = np.maximum(0, h@w1+b1)
        active = (h@w2+b2)*40
        return np.column_stack((np.zeros(len(x), dtype=np.float32), active))

    def select(self, obs, memory, base, equity):
        key = state_key(obs, memory, base, equity)
        if key is None or not self.supported[obs.street, base].any():
            return base
        rates = self.response_rates
        if rates is None:
            from rivercount import _response_rates as rates
        _, aggression = rates(memory, obs.street, None)
        fold, _ = rates(memory, obs.street, Action.RAISE_POT)
        x = encode(obs, base, equity, aggression, fold, key)
        return Action(int(self.actions(x[None, :])[0]))


def load_rows(folder, split):
    batches = [[array("f"), array("f"), array("I")] for _ in range(2 if split else 1)]
    for path in sorted(folder.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                deal = zlib.crc32(("7721/"+r["deal_id"]).encode())
                x, y, ids = batches[int(split and deal < 0.2*2**32)]
                obs = Observation.from_dict(r["features"])
                vals = np.full(4, np.nan, dtype=np.float32)
                committed = STARTING_STACK - obs.my_stack
                for a, payoff in r["returns"].items():
                    vals[int(a)] = float(payoff) + committed
                if np.isfinite(vals[0]) and abs(vals[0]) > 1e-5:
                    raise ValueError("Stored fold payoff disagrees with committed chips")
                x.frombytes(encode(obs, Action(r["base"]), r["equity"], r["aggression"], r["fold_rate"], r["key"]).tobytes())
                y.frombytes(vals.tobytes())
                ids.append(deal)
    return [(np.frombuffer(x, dtype=np.float32).reshape(-1,56), np.frombuffer(y,dtype=np.float32).reshape(-1,4),
             np.frombuffer(ids,dtype=np.uint32)) for x,y,ids in batches]


def compare(policy, baseline, rows):
    x, absolute, deal_ids = rows
    counts = np.isfinite(absolute).sum(1)
    errors = []
    for model, inputs in ((policy,x), (baseline,x[:,:28])):
        per_row = []
        for start in range(0,len(x),4096):
            end = start+4096
            scores = model.predict(inputs[start:end])
            raw_error = scores-absolute[start:end]
            base = x[start:end,4:8].argmax(1)
            errors_squared = (raw_error - raw_error[np.arange(len(base)),base,None]).astype(float)**2
            per_row.extend(np.nansum(errors_squared,axis=1))
        errors.append(np.array(per_row))
    n = counts.sum()
    candidate, reference = [float(np.sqrt(e.sum()/n)) for e in errors]
    _, groups = np.unique(deal_ids, return_inverse=True)
    # Influence of each complete deal on the difference of the two ratios' square roots.
    delta_influence = (errors[0]-candidate**2*counts)/(2*candidate*n) - (errors[1]-reference**2*counts)/(2*reference*n)
    clustered = np.bincount(groups,weights=delta_influence)
    se = np.sqrt(len(clustered)/(len(clustered)-1) * np.square(clustered).sum())
    return dict(rows=len(x), legal_targets=int(n), reference=reference, candidate=candidate,
                delta=candidate-reference, ci95=[candidate-reference-1.96*se,candidate-reference+1.96*se])


def main(args):
    import torch
    from torch import nn
    from tools.fit_neural import action_support
    torch.manual_seed(7721)
    torch.set_num_threads(1)
    train, val = load_rows(args.train, True)
    x, absolute, _ = train
    print(f"Training on cuda: {len(x)} training rows, {len(val[0])} held-out rows.",flush=True)
    dev = torch.device("cuda")
    net = nn.Sequential(nn.Linear(56,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(),lr=0.0006,weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=3,min_lr=0.00004)

    def losses(xb, yb):
        active = net(xb)
        scores = torch.cat((torch.zeros_like(active[:,:1]),active),dim=1)
        valid = yb.isfinite()
        err = scores-yb.nan_to_num()/40
        rel = err-err.gather(1,xb[:,4:8].argmax(1,keepdim=True))
        return (rel.square()*valid).sum(), (err.square()*valid).sum(), valid.sum()

    tx, ty, vx, vy = [torch.from_numpy(a).to(dev) for a in (x,absolute,val[0],val[1])]
    best, epoch_best, saved = float("inf"), 0, None
    for epoch in range(1,65):
        ss = torch.zeros((),device=dev)
        count = torch.zeros((),device=dev)
        for batch in torch.randperm(len(tx),device=dev).split(1024):
            rel, anchored, num = losses(tx[batch],ty[batch])
            opt.zero_grad(set_to_none=True)
            ((rel+0.35*anchored)/num).backward()
            opt.step()
            ss += rel.detach()
            count += num
        with torch.inference_mode():
            metrics = [losses(vx[i:i+4096],vy[i:i+4096]) for i in range(0,len(vx),4096)]
            measured = 40*torch.sqrt(sum(m[0] for m in metrics)/sum(m[2] for m in metrics)).item()
        print(f"Epoch {epoch}/64  train action gain RMSE {40*torch.sqrt(ss/count).item():.6f} chips  validation {measured:.6f}",flush=True)
        scheduler.step(measured)
        if measured < best:
            best,epoch_best = measured,epoch
            saved = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-epoch_best>=9:
            break
    net.load_state_dict(saved)
    print(f"Exporting epoch {epoch_best} on the grouped validation split.")
    params = {}
    for i,layer in enumerate([net[0],net[2],net[4]]):
        params[f"p{2*i}"] = layer.weight.detach().cpu().numpy().T.copy()
        params[f"p{2*i+1}"] = layer.bias.detach().cpu().numpy()
    staged = args.output.with_suffix(".staged.npz")
    np.savez_compressed(staged,architecture="fold-anchor-v1",margin=2.,supported=action_support(x,absolute,64),**params)
    model = AnchoredPolicy(staged)
    expected = net(vx[:1024]).detach().cpu().numpy()*40
    np.testing.assert_allclose(model.predict(val[0][:1024])[:,1:],expected,rtol=1e-4,atol=2e-4)
    staged.replace(args.output)
    base_model = NeuralPolicy.load(args.baseline)
    summary = {"validation":compare(model,base_model,val)}
    if args.test:
        summary["test"] = compare(model,base_model,load_rows(args.test,False)[0])
    args.output.with_suffix(".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Held-out action gain RMSE {summary['validation']['candidate']:.6f} chips.")
    print(json.dumps(summary))
    print(f"Saved {args.output}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--test",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    main(p.parse_args())
