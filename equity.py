from functools import reduce
from itertools import combinations, combinations_with_replacement, permutations
from math import comb
from operator import or_
from pathlib import Path

import numpy as np

from evaluator import score_five


_HOLE_PAIRS = np.array(list(combinations(range(5), 2))).T
_BOARD_TRIPLES = np.array(list(combinations(range(5), 3))).T


def _build_hand_tables():
    """Cards with equal ranks and the same flush possibilities rank identically."""
    rank_pairs = list(combinations_with_replacement(range(9), 2))
    rank_triples = list(combinations_with_replacement(range(9), 3))
    rank_pair_id = {ranks: i for i, ranks in enumerate(rank_pairs)}
    rank_triple_id = {ranks: i for i, ranks in enumerate(rank_triples)}
    pair_type = np.zeros((36, 36), dtype=np.int16)
    triple_type = np.zeros((36, 36, 36), dtype=np.int16)
    for cards in combinations(range(36), 2):
        a, b = cards
        suit = a % 4 if a % 4 == b % 4 else 4
        pair_type[a, b] = pair_type[b, a] = 5 * rank_pair_id[(a // 4, b // 4)] + suit
    for cards in combinations(range(36), 3):
        a, b, c = cards
        suit = a % 4 if a % 4 == b % 4 == c % 4 else 4
        value = 5 * rank_triple_id[(a // 4, b // 4, c // 4)] + suit
        for order in permutations(cards):
            triple_type[order] = value

    scores = np.zeros((5 * len(rank_pairs), 5 * len(rank_triples)), dtype=np.int32)
    rank_scores = {}
    for i, pair in enumerate(rank_pairs):
        for j, triple in enumerate(rank_triples):
            ranks = tuple(sorted(pair + triple))
            if ranks[0] == ranks[-1]:
                continue  # five of one rank cannot be dealt
            if ranks not in rank_scores:
                rainbow = [4 * rank + k % 4 for k, rank in enumerate(ranks)]
                ordinary = score_five(rainbow)
                suited = score_five([4 * r for r in ranks]) if len(set(ranks)) == 5 else ordinary
                rank_scores[ranks] = ordinary, suited
            ordinary, suited = rank_scores[ranks]
            block = scores[5 * i:5 * i + 5, 5 * j:5 * j + 5]
            block[:] = ordinary
            for suit in range(4):
                block[suit, suit] = suited
    return pair_type, triple_type, scores


_RANK_KEYS = np.array([5 ** (card // 4) for card in range(36)])
_BINOMIAL = np.array([[comb(card, n) for n in range(1, 6)] for card in range(36)])


def _build_board_scores():
    """Factor the best board triple for a pair into rank-only and flush scores."""
    rank_triples = {ranks: i for i, ranks in enumerate(combinations_with_replacement(range(9), 3))}
    boards = list(combinations_with_replacement(range(9), 5))
    board_index = np.zeros(5 * 5**8 + 1, dtype=np.int16)
    triple_indices = []
    for i, ranks in enumerate(boards):
        board_index[sum(5**rank for rank in ranks)] = i
        triple_indices.append([rank_triples[t] for t in combinations(ranks, 3)])
    # Type 4 is unsuited. Type 0 allows a flush if both cards and all
    # three board cards are clubs. The resulting score is suit-independent.
    ordinary = _FIVE_SCORE[4::5, 4::5][:, triple_indices].max(axis=2).T.copy()
    flush = np.zeros((512, 45), dtype=np.int32)
    for mask in range(512):
        if 3 <= mask.bit_count() <= 5:
            ranks = [rank for rank in range(9) if mask & (1 << rank)]
            triples = [rank_triples[t] for t in combinations(ranks, 3)]
            flush[mask] = _FIVE_SCORE[0::5, 0::5][:, triples].max(axis=1)
    # Reverse descending-card combinations to put physical boards in their
    # combinatorial-index order. Most share a row of scores in the final table.
    cards = np.array(list(combinations(range(35, -1, -1), 5)), dtype=np.uint8)[::-1]
    rank_ids = board_index[_RANK_KEYS[cards].sum(axis=1)].astype(np.int32)
    suited_ranks = 1 << (cards.astype(np.int16) // 4)
    flush_ids = np.zeros(len(cards), dtype=np.int16)
    flush_by_pair = np.zeros((1 + 4 * 512, 225), dtype=np.int32)
    for suit in range(4):
        matching = cards % 4 == suit
        eligible = matching.sum(axis=1) >= 3
        rank_masks = (suited_ranks[eligible] * matching[eligible]).sum(axis=1)
        start = 1 + 512 * suit
        flush_ids[eligible] = start + rank_masks
        flush_by_pair[start:start + 512, suit::5] = flush
    keys, by_board = np.unique(rank_ids * len(flush_by_pair) + flush_ids, return_inverse=True)
    ordinary_scores = ordinary[keys // len(flush_by_pair)].repeat(5, axis=1)
    scores = np.maximum(ordinary_scores, flush_by_pair[keys % len(flush_by_pair)])
    return by_board.astype(np.uint16), scores


_BOARD_POSITIONS = np.arange(5)


def _build_draw_templates():
    generator = np.random.default_rng(712421)
    templates = {}
    for board_size in (0, 3, 4):
        positions = np.arange(31 - board_size, dtype=np.uint8)
        rows = generator.permuted(np.broadcast_to(positions, (4096, len(positions))), axis=1)
        templates[board_size] = rows[:, :10 - board_size].copy()
    return templates


def _build_river_masks():
    """A bit for each possible five-card opponent hand among 26 unseen cards."""
    hole_sets = np.array(list(combinations(range(26), 5)), dtype=np.int8)
    card_masks = []
    for card in range(26):
        contains = (hole_sets == card).any(axis=1)
        card_masks.append(int.from_bytes(np.packbits(contains, bitorder="little").tobytes(), "little"))
    pairs = list(combinations(range(26), 2))
    return np.array(pairs).T, tuple(card_masks[a] & card_masks[b] for a, b in pairs)


_RIVER_HAND_COUNT = comb(26, 5)
_RIVER_MASK_BYTES = (_RIVER_HAND_COUNT + 7) // 8
_BOARD_FILE = Path(__file__).with_name("board_scores.npz")
_HELPER_FILE = Path(__file__).with_name("equity_helpers.npz")


if __name__ == "__main__":
    _PAIR_TYPE, _TRIPLE_TYPE, _FIVE_SCORE = _build_hand_tables()
    _BOARD_CLASS, _PAIR_SCORE = _build_board_scores()
    np.savez_compressed(_BOARD_FILE, classes=_BOARD_CLASS, scores=_PAIR_SCORE)
    _DRAW_TEMPLATES = _build_draw_templates()
    _RIVER_PAIRS, _RIVER_HAND_MASKS = _build_river_masks()
    packed = b"".join(mask.to_bytes(_RIVER_MASK_BYTES, "little") for mask in _RIVER_HAND_MASKS)
    np.savez_compressed(
        _HELPER_FILE, pairs=_PAIR_TYPE, triples=_TRIPLE_TYPE, five=_FIVE_SCORE,
        river_pairs=_RIVER_PAIRS, river_masks=np.frombuffer(packed, dtype=np.uint8),
        **{f"draw_{k}": v for k, v in _DRAW_TEMPLATES.items()},
    )
else:
    # Build tables once offline. The referee restarts the process for each half.
    with np.load(_BOARD_FILE, allow_pickle=False) as tables:
        _BOARD_CLASS, _PAIR_SCORE = tables["classes"], tables["scores"]
    with np.load(_HELPER_FILE, allow_pickle=False) as tables:
        _PAIR_TYPE, _TRIPLE_TYPE, _FIVE_SCORE = tables["pairs"], tables["triples"], tables["five"]
        _DRAW_TEMPLATES = {k: tables[f"draw_{k}"] for k in (0, 3, 4)}
        _RIVER_PAIRS = tables["river_pairs"]
        _RIVER_HAND_MASKS = tuple(int.from_bytes(row, "little") for row in
                                  tables["river_masks"].reshape(len(_RIVER_PAIRS[0]), _RIVER_MASK_BYTES))


def _river_equity(hero_pairs, board, unseen):
    """Exact result against all 65,780 opponent hands, ties worth half."""
    index = _BINOMIAL[np.sort(board), _BOARD_POSITIONS].sum()
    scores = _PAIR_SCORE[_BOARD_CLASS[index]]
    hero_score = scores[hero_pairs].max()
    opp_pairs = _PAIR_TYPE[tuple(unseen[_RIVER_PAIRS])]
    opp_scores = scores[opp_pairs]
    losing = reduce(or_, map(_RIVER_HAND_MASKS.__getitem__, np.flatnonzero(opp_scores > hero_score)), 0)
    tying = reduce(or_, map(_RIVER_HAND_MASKS.__getitem__, np.flatnonzero(opp_scores == hero_score)), 0)
    return 1.0 - (losing.bit_count() + 0.5 * (tying & ~losing).bit_count()) / _RIVER_HAND_COUNT


def estimate_equity(hole, board, rng, samples=128):
    """Expected share of the pot at showdown against a uniformly random opponent."""
    known = set(hole) | set(board)
    unseen = [c for c in range(36) if c not in known]
    hole = np.asarray(hole)
    board = np.asarray(board, dtype=np.intp)
    hero_pairs = _PAIR_TYPE[tuple(hole[_HOLE_PAIRS])]
    if len(board) == 5:
        return _river_equity(hero_pairs, board, np.asarray(unseen))

    # A fresh permutation of the live deck gives every template row a uniform
    # legal deal. Select a different block of independently generated rows.
    rng.shuffle(unseen)
    templates = _DRAW_TEMPLATES[len(board)]
    if samples <= len(templates):
        start = rng.randrange(len(templates) - samples + 1)
        deals = np.asarray(unseen)[templates[start:start + samples]]
    else:
        generator = np.random.default_rng(rng.getrandbits(64))
        deals = generator.permuted(np.broadcast_to(unseen, (samples, len(unseen))), axis=1)
    boards = np.empty((samples, 5), dtype=np.intp)
    boards[:, :len(board)] = board
    boards[:, len(board):] = deals[:, 5:10 - len(board)]
    boards.sort(axis=1)
    board_ids = _BOARD_CLASS[_BINOMIAL[boards, _BOARD_POSITIONS].sum(axis=1), None]
    opp_pairs = _PAIR_TYPE[tuple(deals[:, i] for i in _HOLE_PAIRS)]
    hero = _PAIR_SCORE[board_ids, hero_pairs].max(axis=1)
    opponent = _PAIR_SCORE[board_ids, opp_pairs].max(axis=1)
    return (np.count_nonzero(hero > opponent) + 0.5 * np.count_nonzero(hero == opponent)) / samples
