"""Fit preflop heads-up matchup equities for the tournament's 36-card deck."""

import argparse
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import rivercount as agent
import equity


def build(deals, bins, seed):
    generator = np.random.default_rng(seed)
    # Four representative equity levels per opponent bin resolve bet cutoffs.
    centers = np.quantile(agent._PREFLOP, (np.arange(4 * bins) + .5) / (4 * bins)).reshape(bins, 4) / 65535
    boundaries = np.quantile(agent._PREFLOP, np.arange(1, bins) / bins) / 65535
    mass = np.zeros((bins, bins))
    wins = np.zeros_like(mass)
    deck = np.arange(36, dtype=np.intp)
    for start in range(0, deals, 8192):
        n = min(8192, deals - start)
        rows = generator.permuted(np.broadcast_to(deck, (n, 36)), axis=1)
        board = np.sort(rows[:, 10:15], axis=1)
        board_id = equity._BOARD_CLASS[equity._BINOMIAL[board, equity._BOARD_POSITIONS].sum(axis=1)]
        categories, ranks = [], []
        for hand in (rows[:, :5], rows[:, 5:10]):
            ids = equity._BINOMIAL[np.sort(hand, axis=1), equity._BOARD_POSITIONS].sum(axis=1)
            categories.append(np.searchsorted(boundaries, agent._PREFLOP[ids] / 65535))
            pairs = equity._PAIR_TYPE[tuple(hand[:, i] for i in equity._HOLE_PAIRS)]
            ranks.append(equity._PAIR_SCORE[board_id[:, None], pairs].max(axis=1))
        cells = categories[0] * bins + categories[1]
        values = (ranks[0] > ranks[1]) + .5 * (ranks[0] == ranks[1])
        count = np.bincount(cells, minlength=bins * bins).reshape(bins, bins)
        points = np.bincount(cells, weights=values, minlength=bins * bins).reshape(bins, bins)
        mass += count + count.T
        wins += points + (count - points).T
    return dict(centers=centers, boundaries=boundaries, prior=mass / mass.sum(axis=1, keepdims=True),
                equity=wins / mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", type=int, default=4_000_000)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7846221)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.deals < args.bins ** 2 * 20 or not 8 <= args.bins <= 128:
        parser.error("Use 8–128 bins and at least 20 times their squared count in deals.")
    data = build(args.deals, args.bins, args.seed)
    if not np.isfinite(data["equity"]).all() or not np.allclose(data["equity"] + data["equity"].T, 1):
        raise AssertionError("Incomplete or asymmetric matchup table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Wrote {args.bins}x{args.bins} matchups from {args.deals:,} disjoint deals to {args.output}")


if __name__ == "__main__":
    main()
