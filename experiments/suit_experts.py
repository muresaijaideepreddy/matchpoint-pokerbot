"""Value model with shared suit encoders and street-specific action experts.

The inference interface uses NumPy only. The training command imports PyTorch
inside train() and checks the exported implementation on the held-out split.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import Action
from neural_policy import NeuralPolicy, features
from rl_policy import state_key


class SuitExpertPolicy(NeuralPolicy):
    def __init__(self, path, response_rates=None):
        with np.load(path, allow_pickle=False) as f:
            if str(f["architecture"]) != "suit-street-v1":
                raise ValueError("Unknown architecture")
            self.params = {k: f[k] for k in f.files if k.endswith(("_w", "_b"))}
            self.supported, self.margin = f["supported"], float(f["margin"])
        self.response_rates = response_rates

    def linear(self, x, name):
        return x @ self.params[name+"_w"] + self.params[name+"_b"]

    def predict(self, x):
        suits = x[:, 28:100].reshape(-1, 4, 18)
        suit_hidden = np.maximum(0, self.linear(suits, "suit1"))
        suit_values = np.maximum(0, self.linear(suit_hidden, "suit2"))
        public = np.concatenate((x[:, :28], x[:, 100:], suit_values.sum(1), suit_values.max(1)), axis=1)
        h = np.maximum(0, self.linear(public, "trunk1"))
        h = h + np.maximum(0, self.linear(h, "trunk2"))
        index, street = np.arange(len(x)), x[:, :4].argmax(1)
        styles = self.linear(x[:, :28], "gate").reshape(-1, 4, 3)[index, street]
        prob = np.exp(styles - styles.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        scores = self.linear(h, "experts").reshape(-1, 4, 3, 4)[index, street]
        return (scores * prob[:, :, None]).sum(1) * 40

    def select(self, obs, memory, base, equity):
        if state_key(obs, memory, base, equity) is None or not self.supported[obs.street, base].any():
            return base
        rates = self.response_rates
        if rates is None:
            from rivercount import _response_rates as rates
        _, aggression = rates(memory, obs.street, None)
        fold, _ = rates(memory, obs.street, Action.RAISE_POT)
        x = features(obs, base, equity, aggression, fold, detailed=True)
        return Action(int(self.actions(x[None, :])[0]))


def train(args):
    import torch
    from torch import nn
    from tools.fit_neural import CHIPS, action_support, evaluate, load_replays, squared_gains, validation_rmse

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.suit1, self.suit2 = nn.Linear(18, 24), nn.Linear(24, 16)
            self.trunk1, self.trunk2 = nn.Linear(84, 96), nn.Linear(96, 96)
            self.gate, self.experts = nn.Linear(28, 12), nn.Linear(96, 48)
            self.dropout = nn.Dropout(0.06)

        def forward(self, x):
            suits = x[:, 28:100].reshape(-1, 4, 18)
            sv = self.suit2(self.suit1(suits).relu()).relu()
            public = torch.cat((x[:, :28], x[:, 100:], sv.sum(1), sv.amax(1)), dim=1)
            h = self.trunk1(public).relu()
            h = h + self.dropout(self.trunk2(h).relu())
            index, street = torch.arange(len(x), device=x.device), x[:, :4].argmax(1)
            gate = self.gate(x[:, :28]).reshape(-1, 4, 3)[index, street].softmax(1)
            values = self.experts(h).reshape(-1, 4, 3, 4)[index, street]
            return (values * gate[:, :, None]).sum(1)

    torch.manual_seed(7721)
    torch.set_num_threads(1)
    dev = torch.device(args.device)
    (x, y), (vx, vy) = load_replays(args.dataset_dir, 7721, 0.2, detailed=True)
    print(f"Training on {dev}: {len(x)} training rows, {len(vx)} held-out rows.", flush=True)
    net = Network().to(dev)
    tx, ty = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=0.00004)
    best, epoch_best, chosen = float("inf"), 0, None
    for epoch in range(1, 65):
        net.train()
        ss = torch.zeros((), device=dev)
        n = torch.zeros((), device=dev)
        for ids in torch.randperm(len(x), device=dev).split(2048):
            loss, count = squared_gains(net, tx[ids], ty[ids])
            opt.zero_grad(set_to_none=True)
            (loss/count).backward()
            opt.step()
            ss += loss.detach()
            n += count
        net.eval()
        val = validation_rmse(net, vx, vy, dev)
        scheduler.step(val)
        print(f"Epoch {epoch}/64  train action gain RMSE {CHIPS*(ss/n).sqrt().item():.6f} chips  validation {val:.6f}", flush=True)
        if val < best:
            best, epoch_best = val, epoch
            chosen = {key: t.detach().cpu().clone() for key, t in net.state_dict().items()}
        if epoch - epoch_best >= 10:
            break
    net.load_state_dict(chosen)
    net.eval()
    print(f"Exporting epoch {epoch_best} on the grouped validation split.")
    array = {}
    for name in ("suit1", "suit2", "trunk1", "trunk2", "gate", "experts"):
        layer = getattr(net, name)
        array[name+"_w"] = layer.weight.detach().cpu().numpy().T.copy()
        array[name+"_b"] = layer.bias.detach().cpu().numpy()
    target = args.output
    target.parent.mkdir(exist_ok=True, parents=True)
    staged = target.with_suffix(".staged.npz")
    np.savez_compressed(staged, architecture="suit-street-v1", supported=action_support(x, y, 64), margin=2., **array)
    policy = SuitExpertPolicy(staged)
    with torch.inference_mode():
        reference = CHIPS*net(torch.from_numpy(vx[:1024]).to(dev)).cpu().numpy()
    actual = policy.predict(vx[:1024])
    np.testing.assert_allclose(actual, reference, atol=0.0002, rtol=0.0001)
    permuted = vx[:1024].copy()
    permuted[:, 28:100] = vx[:1024, 28:100].reshape(-1, 4, 18)[:, [2,0,3,1]].reshape(-1,72)
    np.testing.assert_allclose(policy.predict(permuted), actual, atol=0.0002, rtol=0.0001)
    print(f"Verified NumPy export and suit invariance; max Torch difference {np.abs(actual-reference).max():.6g} chips.")
    evaluate(policy, vx, vy)
    staged.replace(target)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda","cpu"))
    train(parser.parse_args())
