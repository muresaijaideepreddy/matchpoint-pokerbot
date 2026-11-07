"""Fit shallow histogram-boosted action-value regressors and export NumPy trees."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boosted_policy import BoostedPolicy
from tools.fit_neural import action_support, load_replays
from game import Action


def fit_action(x, y, hx, hy, action, args):
    # An incumbent action has identically zero advantage. Fit only the rows
    # where this action is a genuine alternative, and anchor the base at export.
    train = np.isfinite(y[:, action]) & (x[:, 4+action] == 0)
    valid = np.isfinite(hy[:, action]) & (hx[:, 4+action] == 0)
    tx, ty = x[train], y[train, action]
    vx, vy = hx[valid], hy[valid, action]
    if len(tx) < 2 or len(vx) < 1:
        raise ValueError(f"Insufficient grouped replay for legal alternative {action.name}")
    start = time.monotonic()
    model = HistGradientBoostingRegressor(
        max_iter=args.rounds, max_depth=args.depth, max_leaf_nodes=2**args.depth,
        min_samples_leaf=args.min_leaf, max_bins=127, learning_rate=args.rate, l2_regularization=24.0,
        early_stopping=False, categorical_features=None, random_state=args.seed,
    ).fit(tx, ty)
    errors = []
    for inputs, target in ((tx, ty), (vx, vy)):
        errors.append(np.array([np.sum((p-target)**2, dtype=np.float64) for p in model.staged_predict(inputs)]))
    print(f"Fitted {action.name}: {len(tx):,} training alternatives, {len(vx):,} held-out alternatives, "
          f"{time.monotonic()-start:.1f} s", flush=True)
    return model, errors


def export(models, best, depth, sample, support, margin, path):
    trees = [level[0].nodes for model in models for level in model._predictors[:best]]
    starts = np.cumsum([0] + [len(nodes) for nodes in trees[:-1]], dtype=np.int32)
    feature, threshold, left, right, value = [], [], [], [], []
    for start, nodes in zip(starts, trees):
        if np.any(nodes["is_categorical"]):
            raise ValueError("Only numeric tree splits are supported")
        leaf = nodes["is_leaf"].astype(bool)
        positions = np.arange(len(nodes))
        feature.extend(np.where(leaf, 0, nodes["feature_idx"]))
        threshold.extend(np.where(leaf, 0.0, nodes["num_threshold"]))
        left.extend(start + np.where(leaf, positions, nodes["left"]))
        right.extend(start + np.where(leaf, positions, nodes["right"]))
        value.extend(nodes["value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, boosted_format=1, inputs=sample.shape[1], trees=best, depth=depth,
                                starts=starts, feature=np.array(feature, dtype=np.uint16),
                                threshold=np.asarray(threshold, dtype=np.float64), left=np.array(left, dtype=np.int32),
                                right=np.array(right, dtype=np.int32), value=np.asarray(value, dtype=np.float64),
                                bias=np.array([m._baseline_prediction.item() for m in models]),
                                supported=support, margin=margin)
        policy = BoostedPolicy.load(temp)
        truth = np.column_stack([next(p for i, p in enumerate(model.staged_predict(sample), 1) if i == best)
                                 for model in models])
        base = sample[:, 4:8].argmax(axis=1)
        truth[np.arange(len(sample)), base] = 0
        actual = policy.predict(sample)
        single = np.stack([policy.predict_one(row, b) for row, b in zip(sample, base)])
        np.testing.assert_allclose(actual, truth, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(single, truth, rtol=1e-10, atol=1e-9)
        max_error = float(np.max(np.abs(truth-single)))
        temp.replace(path)
        return policy, max_error
    finally:
        temp.unlink(missing_ok=True)


def measure(policy, hx, hy):
    squared = gain = 0.0
    legal = changes = 0
    for i in range(0, len(hx), 1024):
        x, y = hx[i:i+1024], hy[i:i+1024]
        values = policy.predict(x)
        action = policy.actions(x, values)
        realized = y[np.arange(len(x)), action]
        if not np.isfinite(realized).all():
            raise RuntimeError("Export selected an illegal action")
        squared += np.nansum((y-values)**2)
        legal += int(np.isfinite(y).sum())
        changes += int(np.count_nonzero(action != x[:, 4:8].argmax(axis=1)))
        gain += float(realized.sum(dtype=np.float64))
    return (squared/legal)**0.5, gain/len(hx), changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--min-leaf", type=int, default=192)
    p.add_argument("--rate", type=float, default=0.08)
    p.add_argument("--margin", type=float, default=2)
    p.add_argument("--min-samples", type=int, default=64)
    p.add_argument("--threads", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--seed", type=int, default=7721)
    p.add_argument("--holdout", type=float, default=0.2)
    args = p.parse_args()
    if (not 1 <= args.depth <= 8 or not 1 <= args.rounds <= 512 or args.min_leaf < 2 or args.min_samples < 1
            or not 0 < args.rate <= 1 or not 0 < args.holdout < 1 or args.margin <= 0):
        p.error("Invalid tree, split, or action-support setting")
    with threadpool_limits(limits=args.threads):
        (x, y), (hx, hy) = load_replays(args.dataset_dir, args.seed, args.holdout, detailed=True)
        print(f"Training on CPU: {len(x)} training rows, {len(hx)} held-out rows.", flush=True)
        models, errors = [], np.zeros((2, args.rounds))
        for action in Action:
            model, loss = fit_action(x, y, hx, hy, action, args)
            models.append(model)
            errors += loss
        curve = np.sqrt(errors / np.array([np.isfinite(y).sum(), np.isfinite(hy).sum()])[:, None])
        best = int(curve[1].argmin()) + 1
        print("Staged evaluation: one boosting tree per action per epoch.", flush=True)
        for i, (tr, va) in enumerate(curve.T, 1):
            print(f"Epoch {i}/{args.rounds}  train action gain RMSE {tr:.3f} chips  validation {va:.3f}")
        print(f"Exporting epoch {best} on the grouped validation split.", flush=True)
        sample = hx[np.linspace(0, len(hx)-1, min(len(hx), 2048), dtype=int)]
        policy, error = export(models, best, args.depth, sample, action_support(x, y, args.min_samples),
                               args.margin, args.output)
        print(f"Verified batched and single-row NumPy export on {len(sample)} rows; max error {error:.6g} chips.")
        rmse, gain, changes = measure(policy, hx, hy)
        np.testing.assert_allclose(rmse, curve[1, best-1], rtol=1e-8, atol=1e-7)
        print(f"Held-out action gain RMSE {rmse:.3f} chips. {changes}/{len(hx)} decisions changed, "
              f"mean replay gain {gain:+.3f} chips/decision.", flush=True)
        times = []
        for row in sample:
            t = time.perf_counter_ns()
            policy.predict_one(row, row[4:8].argmax())
            times.append((time.perf_counter_ns()-t)/1e6)
        summary = dict(training_rows=len(x), held_out_rows=len(hx), detailed_features=x.shape[1],
                       grouped_deal_split_seed=args.seed, holdout_fraction=args.holdout, best_epoch=best,
                       rounds=args.rounds, depth=args.depth, learning_rate=args.rate, min_leaf=args.min_leaf,
                       min_action_support=args.min_samples, margin_chips=args.margin,
                       held_out_rmse=rmse, replay_gain_chips=gain, overrides=changes,
                       max_export_error_chips=error, exported_bytes=args.output.stat().st_size,
                       single_score_p50_ms=np.percentile(times,50), single_score_p95_ms=np.percentile(times,95),
                       single_score_p99_ms=np.percentile(times,99))
        args.output.with_name("summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
