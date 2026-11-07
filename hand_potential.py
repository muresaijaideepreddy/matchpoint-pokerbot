"""How strongly the remaining board can change a hand's showdown equity."""

from functools import lru_cache
from zlib import crc32

import numpy as np

from equity import _BINOMIAL, _BOARD_CLASS, _BOARD_POSITIONS, _HOLE_PAIRS, _PAIR_SCORE, _PAIR_TYPE


@lru_cache(maxsize=2)
def _templates(n):
    """Each of 16 runouts has 24 legal opposing hands drawn conditional on it."""
    rng = np.random.default_rng(21861 + n)
    missing = 5 - n
    size = 31 - n
    permutations = rng.permuted(np.broadcast_to(np.arange(size, dtype=np.uint8), (32, 16, size)), axis=2)
    live = permutations[:, :, missing:]
    picks = rng.permuted(np.broadcast_to(np.arange(26, dtype=np.uint8), (32, 16, 24, 26)), axis=3)
    hands = np.take_along_axis(live[:, :, None, :], picks[:, :, :, :5], axis=3)
    return permutations[:, :, :missing].copy(), hands


def _canonical(hole, board):
    # Sort equivalent suit names identically so a suit relabeling produces the
    # same estimate as well as the same exact distribution.
    signatures = [[0, 0] for _ in range(4)]
    for group, cards in enumerate((board, hole)):
        for card in cards:
            signatures[card % 4][group] |= 1 << (card // 4)
    suits = sorted(range(4), key=signatures.__getitem__)
    labels = {suit: i for i, suit in enumerate(suits)}
    def encode(cards):
        return tuple(sorted((card // 4) * 4 + labels[card % 4] for card in cards))
    return encode(hole), encode(board)


def future_equity_variation(hole, board):
    """Estimated fraction of showdown variance explained by turn/river runouts."""
    n = len(board)
    if n not in (3, 4):
        return 0.0
    hole, board = _canonical(hole, board)
    known = set(hole + board)
    deck = np.asarray([c for c in range(36) if c not in known])
    runouts, opponents = _templates(n)
    block = crc32(bytes(hole + board)) % len(runouts)
    added = deck[runouts[block]]
    boards = np.sort(np.column_stack((np.broadcast_to(board, (len(added), n)), added)), axis=1)
    ids = _BOARD_CLASS[_BINOMIAL[boards, _BOARD_POSITIONS].sum(axis=1)]
    hero_pairs = _PAIR_TYPE[tuple(np.asarray(hole)[_HOLE_PAIRS])]
    hero = _PAIR_SCORE[ids[:, None], hero_pairs].max(axis=1)
    other = deck[opponents[block]]
    pairs = _PAIR_TYPE[other[..., _HOLE_PAIRS[0]], other[..., _HOLE_PAIRS[1]]]
    opponent = _PAIR_SCORE[ids[:, None, None], pairs].max(axis=2)
    shares = (hero[:, None] > opponent) + 0.5 * (hero[:, None] == opponent)
    conditional = shares.mean(axis=1)
    noise = shares.var(axis=1, ddof=1).mean() / shares.shape[1]
    between = max(0.0, conditional.var(ddof=1) - noise)
    return float(min(1.0, between / max(0.04, shares.var())))
