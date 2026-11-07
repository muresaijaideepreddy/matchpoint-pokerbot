"""Check the bot's equity shortcuts against the tournament's hand evaluator."""

from itertools import combinations, permutations
from math import comb
from random import Random

import pytest

from rivercount import _PREFLOP, preflop_equity
from equity import estimate_equity
from evaluator import cards_from_str, omaha_score, score_five


def _river_outcomes(hole, board):
    unseen = sorted(set(range(36)) - set(hole) - set(board))
    hero = omaha_score(hole, board)
    triples = list(combinations(board, 3))
    pair_scores = {
        pair: max(score_five(pair + triple) for triple in triples)
        for pair in combinations(unseen, 2)
    }
    wins = ties = 0
    for opponent in combinations(unseen, 5):
        score = max(pair_scores[pair] for pair in combinations(opponent, 2))
        wins += hero > score
        ties += hero == score
    return wins, ties


def test_river_ties_and_flush_blocker():
    board = cards_from_str("6c 7c 8c Qd Kh")
    shares = []
    for ace in ("Ad", "Ac"):
        hole = cards_from_str(f"9d Ts {ace} Jd Qh")
        wins, ties = _river_outcomes(hole, board)
        total = comb(26, 5)
        assert 0 < ties < total - wins
        actual = estimate_equity(hole, board, Random(1), samples=1)
        assert actual == pytest.approx((wins + ties / 2) / total, abs=1e-12)
        shares.append(actual)
    # Both hands make the same ten-high straight. Ac removes opposing flushes.
    assert shares[1] > shares[0]


@pytest.mark.parametrize("public", ["Tc Jc Qc", "Tc Jc Qc 6d"])
def test_flop_and_turn_rollouts_preserve_a_royal_flush(public):
    hole = cards_from_str("Ac Kc Ah Jd Ts")
    assert estimate_equity(hole, cards_from_str(public), Random(7), samples=64) == 1


def test_preflop_index_and_suit_symmetry():
    size = comb(36, 5)
    assert len(_PREFLOP) == size
    # Descending-card combinations enumerate the file from its last entry to
    # its first, independently of the production combinatorial-index formula.
    offsets = {0, 7231, 128025, 320044, size - 1}
    for offset, hand in enumerate(combinations(range(35, -1, -1), 5)):
        if offset not in offsets:
            continue
        expected = int(_PREFLOP[size - 1 - offset]) / 65535
        for suits in permutations(range(4)):
            renamed = [4 * (c // 4) + suits[c % 4] for c in hand]
            assert preflop_equity(renamed) == expected
            assert preflop_equity(renamed[::-1]) == expected


def test_preflop_table_distinguishes_aces_from_dead_quads():
    aces = cards_from_str("Ac Ad Kc Qd Jh")
    quads = cards_from_str("6c 6d 6h 6s 7d")
    # Omaha uses just two hole cards. Four sixes leave no six for the board.
    assert preflop_equity(aces) > 0.6
    assert preflop_equity(quads) < 0.4
