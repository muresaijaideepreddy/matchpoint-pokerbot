"""Average training labels over possible boards when a call ends all betting.

The known opponent hand is used only to generate offline targets. Features and all
held-out labels remain byte-for-byte unchanged. Also measure unavoidable variance
from the remaining board alone on these particular held-out decisions.
"""
import argparse
from itertools import combinations
import json
from math import sqrt
from pathlib import Path
import sys
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from equity import _BINOMIAL, _BOARD_CLASS, _BOARD_POSITIONS, _HOLE_PAIRS, _PAIR_SCORE, _PAIR_TYPE
from game import make_deal


def score_shares(boards, hero, other):
    boards = np.sort(boards, axis=1)
    classes = _BOARD_CLASS[_BINOMIAL[boards, _BOARD_POSITIONS].sum(axis=1), None]
    ranks = []
    for hand in (hero, other):
        pairs = _PAIR_TYPE[tuple(np.asarray(hand)[_HOLE_PAIRS])]
        ranks.append(_PAIR_SCORE[classes, pairs].max(axis=1))
    return (ranks[0] > ranks[1]).astype(np.float64) + 0.5 * (ranks[0] == ranks[1])


def average_call(row, indices, samples):
    obs = row["features"]
    prefix, index = row["deal_id"].rsplit("/", 1)
    deal = make_deal(prefix, int(index))
    hole, board = tuple(obs["hole_cards"]), tuple(obs["board"])
    seat = deal.hole_cards.index(hole)
    if board != deal.board[:len(board)]:
        raise ValueError("Replay board disagrees with reconstructed deal")
    other = deal.hole_cards[1-seat]
    pot, cost = obs["pot"], min(obs["to_call"], obs["my_stack"])
    returns = row["returns"]
    fold_return = float(returns["0"])
    observed = score_shares(np.asarray([deal.board]), hole, other)[0]
    if not np.isclose(fold_return + (pot + cost) * observed - cost, returns["1"]):
        raise ValueError("Reconstructed all-in payout disagrees with the stored referee payout")
    live = np.array(sorted(set(range(36)).difference(hole, other, board)), dtype=np.intp)
    combinations_by_position = indices[len(board)]
    total = len(combinations_by_position)
    sampled = total > samples
    if sampled:
        seed = zlib.crc32(row["deal_id"].encode())
        choices = np.random.default_rng(seed).choice(total, size=samples, replace=False)
        combinations_by_position = combinations_by_position[choices]
    boards = np.empty((len(combinations_by_position), 5), dtype=np.intp)
    boards[:, :len(board)] = board
    boards[:, len(board):] = live[combinations_by_position]
    shares = score_shares(boards, hole, other)
    # Sampling without replacement: unbiased finite-population variance.
    variance = shares.var(ddof=int(sampled)) * ((total-1)/total if sampled else 1)
    return fold_return + (pot+cost)*shares.mean() - cost, float(variance*(pot+cost)**2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--split-seed", type=int, default=7721)
    p.add_argument("--holdout", type=float, default=0.2)
    args = p.parse_args()
    if args.samples < 2 or not 0 < args.holdout < 1:
        p.error("Use at least two runouts and a valid holdout fraction")
    args.output.mkdir(parents=True, exist_ok=False)
    indices = {n: np.array(list(combinations(range(26-n), 5-n)), dtype=np.int16) for n in (0,3,4)}
    counts = {"training_rows": 0, "heldout_rows": 0, "training_allins": 0, "heldout_allins": 0,
              "heldout_legal_action_entries": 0}
    variance_sum = 0.0
    for source_num, source in enumerate(args.dataset_dir):
        shards = sorted(source.rglob("*.jsonl"))
        if not shards:
            raise ValueError(f"No replay data found under {source}")
        for path in shards:
            dest = args.output / str(source_num) / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with path.open(encoding="utf-8") as src, dest.open("x", encoding="utf-8") as out:
                for raw in src:
                    row = json.loads(raw)
                    heldout = zlib.crc32(f"{args.split_seed}/{row['deal_id']}".encode()) < args.holdout * 2**32
                    key = "heldout" if heldout else "training"
                    counts[key + "_rows"] += 1
                    if heldout:
                        counts["heldout_legal_action_entries"] += len(row["returns"])
                    obs = row["features"]
                    if (len(obs["board"]) < 5 and set(row["returns"]) == {"0", "1"} and obs["to_call"] > 0
                            and (obs["opp_stack"] == 0 or obs["my_stack"] <= obs["to_call"])):
                        call, variance = average_call(row, indices, args.samples)
                        counts[key + "_allins"] += 1
                        if heldout:
                            variance_sum += variance
                        else:
                            row["returns"]["1"] = call
                            raw = json.dumps(row, separators=(",", ":")) + "\n"
                    out.write(raw)
            print(f"Processed {source.name}/{path.name}; averaged {counts['training_allins']} training labels.", flush=True)
    counts["holdout_board_only_rmse_lower_bound_estimate_all_entries"] = sqrt(variance_sum/counts["heldout_legal_action_entries"])
    counts["holdout_board_only_sd_for_allin_alternative"] = sqrt(variance_sum/counts["heldout_allins"]) if counts["heldout_allins"] else None
    (args.output/"diagnostics.json").write_text(json.dumps(counts, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
