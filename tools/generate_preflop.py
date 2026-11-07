"""Regenerate the preflop showdown equity table with a fixed training seed.

For each random board, deal six disjoint five-card hands. Each hand has
five legal heads-up opponents. Pool results across equivalent suit names.
Run from any directory after installing the bot's requirements.
"""

import argparse
import sys
import zlib
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equity import _BOARD_TRIPLES, _FIVE_SCORE, _HOLE_PAIRS, _PAIR_TYPE, _TRIPLE_TYPE


BINOMIAL = np.array([[comb(card, n) for card in range(36)] for n in range(1, 6)])


def hand_indices(sorted_hands):
    return sum(row[sorted_hands[..., i]] for i, row in enumerate(BINOMIAL))


def suit_classes():
    hands = np.array(list(combinations(range(36), 5)), dtype=np.uint8)
    rank_bits = np.left_shift(1, hands.astype(np.int64) // 4)
    suits = np.stack([((hands % 4 == suit) * rank_bits).sum(axis=1)
                     for suit in range(4)], axis=1)
    suits.sort(axis=1)
    keys = (suits << np.arange(0, 36, 9)).sum(axis=1)
    _, classes = np.unique(keys, return_inverse=True)
    by_index = np.empty(len(hands), dtype=np.int32)
    by_index[hand_indices(hands)] = classes
    return by_index


def generate(num_boards, groups, seed):
    rng = np.random.default_rng(seed)
    classes = suit_classes()
    nclasses = int(classes.max()) + 1
    observations = np.zeros(nclasses, dtype=np.int64)
    half_points = np.zeros(nclasses, dtype=np.int64)
    deck = np.arange(36, dtype=np.uint8)
    for start in range(0, num_boards, 16):
        n = min(16, num_boards - start)
        decks = rng.permuted(np.broadcast_to(deck, (n, 36)), axis=1)
        boards = decks[:, :5]
        triples = _TRIPLE_TYPE[tuple(boards[:, i] for i in _BOARD_TRIPLES)]
        # For a fixed board, value every hole pair once and reuse it.
        pair_values = _FIVE_SCORE[:, triples].max(axis=2).T
        cards = rng.permuted(np.broadcast_to(decks[:, None, 5:], (n, groups, 31)), axis=2)
        hands = cards[:, :, :30].reshape(n, groups, 6, 5)
        pairs = _PAIR_TYPE[tuple(hands[..., i] for i in _HOLE_PAIRS)]
        scores = pair_values[np.arange(n)[:, None, None, None], pairs].max(axis=-1)
        opponents = scores[..., None, :]
        hero = scores[..., :, None]
        # A win earns two half-points, a tie one. Exclude the tie against self.
        earned = (hero > opponents).sum(axis=-1) * 2 + (hero == opponents).sum(axis=-1) - 1
        ids = classes[hand_indices(np.sort(hands, axis=-1))].ravel()
        observations += np.bincount(ids, minlength=nclasses)
        half_points += np.bincount(ids, weights=earned.ravel(), minlength=nclasses).astype(np.int64)
    if np.any(observations == 0):
        raise ValueError("Increase --boards to cover every starting hand class.")
    values = half_points / (10 * observations)
    return np.rint(65535 * values[classes]).astype("<u2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=40000)
    parser.add_argument("--groups", type=int, default=192)
    parser.add_argument("--seed", type=int, default=83571)
    args = parser.parse_args()
    table = generate(args.boards, args.groups, args.seed)
    output = ROOT / "preflop.bin.z"
    output.write_bytes(zlib.compress(table.tobytes(), 9))
    print(f"{len(table):,} hands, mean equity {table.mean() / 65535:.5f}, {output.stat().st_size:,} bytes.")
