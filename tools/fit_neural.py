"""Fit legal action advantages from train_rl.py JSONL replays and export a NumPy policy."""

import argparse
from array import array
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys
import tempfile
import zlib

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game import Action, Observation
from neural_policy import DETAILED_FEATURES, FORMAT, HIDDEN, N_FEATURES, NeuralPolicy, features


# Training in units of half a starting stack keeps gradients well-scaled.
# The exported final layer converts the prediction back to chips.
CHIPS = 40.0


def load_replays(directories, seed, holdout_fraction, detailed=False, disk=None):
    if isinstance(directories, Path):
        directories = [directories]
    files = sorted({path.resolve() for directory in directories for path in directory.rglob("*.jsonl")})
    if not files:
        raise ValueError(f"No JSONL replay shards under {directories}")
    columns = DETAILED_FEATURES if detailed else N_FEATURES
    buffers = [(array("f"), array("f")), (array("f"), array("f"))]
    with ExitStack() as stack:
        streams = []
        if disk:
            streams = [(stack.enter_context((disk / f"x{i}").open("wb")),
                        stack.enter_context((disk / f"y{i}").open("wb"))) for i in range(2)]
        _encode_replays(files, seed, holdout_fraction, detailed, buffers, streams)
    if disk:
        splits = []
        for i in range(2):
            if not (disk / f"x{i}").stat().st_size:
                raise ValueError("An empty deal split was produced. Generate more replay deals.")
            splits.append((np.memmap(disk / f"x{i}", dtype=np.float32, mode="c").reshape(-1, columns),
                           np.memmap(disk / f"y{i}", dtype=np.float32, mode="c").reshape(-1, len(Action))))
        return splits
    splits = []
    for inputs, targets in buffers:
        if not len(inputs):
            raise ValueError("An empty deal split was produced. Generate more replay deals.")
        splits.append((np.frombuffer(inputs, dtype=np.float32).reshape(-1, columns),
                       np.frombuffer(targets, dtype=np.float32).reshape(-1, len(Action))))
    return splits


def _encode_replays(files, seed, holdout_fraction, detailed, buffers, streams):
    # Hash the underlying deal, so both seats and all its street decisions stay together.
    for path in files:
        with path.open(encoding="utf-8") as rows:
            for line_no, line in enumerate(rows, 1):
                try:
                    row = json.loads(line)
                    deal_id = row["deal_id"]
                    if not isinstance(deal_id, str) or not deal_id:
                        raise ValueError("missing deal_id")
                    held_out = zlib.crc32(f"{seed}/{deal_id}".encode()) < holdout_fraction * 2**32
                    obs = Observation.from_dict(row["features"])
                    base = Action(row["base"])
                    x = features(obs, base, row["equity"], row["aggression"], row["fold_rate"], detailed)
                    returns = {Action(int(k)): float(v) for k, v in row["returns"].items()}
                    if (set(returns) != set(obs.legal_actions) or base not in returns
                            or not np.isfinite(x).all() or not np.isfinite(list(returns.values())).all()):
                        raise ValueError("invalid legal actions, base action, features or returns")
                    # NaN distinguishes an unavailable action from a zero-chip gain.
                    y = [returns[a] - returns[base] if a in returns else np.nan for a in Action]
                    if streams:
                        inputs, targets = streams[held_out]
                        inputs.write(x.tobytes())
                        targets.write(np.asarray(y, dtype=np.float32).tobytes())
                    else:
                        inputs, targets = buffers[held_out]
                        inputs.frombytes(x.tobytes())
                        targets.extend(y)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def action_support(x, y, min_samples):
    counts = np.zeros((4, 4, 4), dtype=np.int64)
    street, base = x[:, :4].argmax(axis=1), x[:, 4:8].argmax(axis=1)
    np.add.at(counts, (street, base), np.isfinite(y))
    return counts >= min_samples


def squared_gains(net, xb, yb):
    legal = yb.isfinite()
    base = xb[:, 4:8].argmax(dim=1, keepdim=True)
    scores = net(xb)
    gains = scores - scores.gather(1, base)
    return ((gains - yb.nan_to_num() / CHIPS).square() * legal).sum(), legal.sum()


@torch.inference_mode()
def validation_rmse(net, x, y, device):
    error = count = 0.0
    for i in range(0, len(x), 4096):
        xb = torch.from_numpy(x[i:i+4096]).to(device)
        yb = torch.from_numpy(y[i:i+4096]).to(device)
        squared, valid = squared_gains(net, xb, yb)
        error += squared.item()
        count += valid.item()
    return CHIPS * (error / count) ** 0.5


def fit(x, y, device, args, validation):
    width = args.width
    net = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(),
                        nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4)).to(device)
    optimizer = (torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                 if args.weight_decay else torch.optim.Adam(net.parameters(), lr=args.learning_rate))
    inputs = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    net.train()
    best_rmse, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(args.epochs):
        loss_sum = torch.zeros((), device=device)
        legal_count = torch.zeros((), device=device)
        for batch in torch.randperm(len(x)).split(args.batch_size):
            xb, yb = inputs[batch].to(device), targets[batch].to(device)
            squared, valid = squared_gains(net, xb, yb)
            optimizer.zero_grad(set_to_none=True)
            (squared / valid).backward()
            optimizer.step()
            loss_sum += squared.detach()
            legal_count += valid
        rmse = (loss_sum / legal_count).sqrt().item() * CHIPS
        message = f"Epoch {epoch + 1}/{args.epochs}  train action gain RMSE {rmse:.3f} chips"
        if args.patience:
            loss = validation_rmse(net, *validation, device)
            message += f"  validation {loss:.3f}"
            if loss < best_rmse:
                best_rmse, best_epoch = loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(message, flush=True)
        if args.patience and epoch + 1 - best_epoch >= args.patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"Exporting epoch {best_epoch} on the grouped validation split.", flush=True)
    return net.eval()


def export(net, sample, supported, margin, path):
    params = {}
    for i, layer in enumerate((net[0], net[2], net[4])):
        scale = CHIPS if i == 2 else 1.0
        params[f"w{2*i}"] = np.ascontiguousarray(layer.weight.detach().cpu().numpy().T * scale)
        params[f"w{2*i+1}"] = layer.bias.detach().cpu().numpy() * scale
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
        temp = Path(fp.name)
    try:
        with temp.open("wb") as fp:
            inputs, width = params["w0"].shape
            version = FORMAT if (inputs, width) == (N_FEATURES, HIDDEN) else 2
            np.savez_compressed(fp, format=version, inputs=inputs, width=width,
                                supported=supported, margin=margin, **params)
        policy = NeuralPolicy.load(temp)
        with torch.inference_mode():
            reference = net(torch.from_numpy(sample).to(next(net.parameters()).device)).cpu().numpy() * CHIPS
        actual = policy.predict(sample)
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-4)
        error = float(np.abs(actual - reference).max())
        print(f"Verified NumPy export on {len(sample)} rows; max difference {error:.6g} chips.")
        temp.replace(path)
        return policy
    finally:
        temp.unlink(missing_ok=True)


def evaluate(policy, x, y):
    squared = gain = 0.0
    legal_count = overrides = 0
    for start in range(0, len(x), 4096):
        xb, yb = x[start:start+4096], y[start:start+4096]
        rows = np.arange(len(xb))
        base = xb[:, 4:8].argmax(axis=1)
        scores = policy.predict(xb)
        gains = scores - scores[rows, base][:, None]
        squared += float(np.nansum((gains - yb) ** 2, dtype=np.float64))
        legal_count += int(np.isfinite(yb).sum())
        selected = policy.actions(xb, scores)
        realized = yb[rows, selected]
        if not np.isfinite(realized).all():
            raise RuntimeError("The exported policy selected an unavailable action")
        overrides += int((selected != base).sum())
        gain += realized.sum(dtype=np.float64)
    print(f"Held-out action gain RMSE {(squared/legal_count)**0.5:.3f} chips. "
          f"{overrides}/{len(x)} decisions changed, mean replay gain {gain/len(x):+.3f} chips/decision.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, action="append", required=True,
                   help="replay directory containing JSONL shards; repeat to combine sources")
    p.add_argument("--disk-backed", action="store_true", help="stage encoded tensors in temporary memory-mapped files")
    p.add_argument("--output", type=Path, required=True, help="destination .npz candidate checkpoint")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=0, help="stop after this many epochs without lower validation RMSE")
    p.add_argument("--weight-decay", type=float, default=0., help="weight decay for AdamW; zero uses Adam")
    p.add_argument("--width", type=int, default=HIDDEN, help="units per hidden layer")
    p.add_argument("--detailed", action="store_true", help="encode rank/suit blockers and street betting lines")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--min-samples", type=int, default=64, help="minimum legal samples per street/base/action")
    p.add_argument("--margin", type=float, default=2.0, help="minimum predicted gain over incumbent, in chips")
    p.add_argument("--holdout", type=float, default=0.2, help="fraction of deals held out")
    p.add_argument("--seed", type=int, default=7721)
    args = p.parse_args()
    if (min(args.epochs, args.batch_size, args.min_samples) < 1 or args.patience < 0 or not 0 <= args.weight_decay < float("inf")
            or not 1 <= args.width <= 1024 or not 0 < args.holdout < 1
            or not 0 < args.learning_rate < float("inf") or not 0 < args.margin < float("inf")):
        p.error("supply positive training settings and a holdout strictly between zero and one")
    if args.output.suffix.lower() != ".npz":
        p.error("--output must be a .npz file")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable to this PyTorch installation")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    with (tempfile.TemporaryDirectory(prefix="plo-replay-") if args.disk_backed else nullcontext()) as scratch:
        (x, y), (hold_x, hold_y) = load_replays(args.dataset_dir, args.seed, args.holdout, args.detailed,
                                               Path(scratch) if scratch else None)
        print(f"Training on {device}: {len(x)} training rows, {len(hold_x)} held-out rows.", flush=True)
        net = fit(x, y, device, args, (hold_x, hold_y))
        policy = export(net, hold_x[:1024], action_support(x, y, args.min_samples), args.margin, args.output)
        evaluate(policy, hold_x, hold_y)
        del x, y, hold_x, hold_y
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
