"""Search calling and raising margins with actual deployed callback randomness.

Run with one CPU worker. Each trial starts both strategies from a fresh half.
Screen settings on training seeds, freeze the candidate, then run untouched
development deals. Use compare_agents.py separately for final confirmation.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, replace
from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rivercount as agent
import equity
from evaluator import warmup
from game import AgentConfig, BIG_BLIND, Game, Street, make_deal
from judge_engine import AgentStats, play_half
from response_policy import Adjustments, choose_adjusted


BLANK = Adjustments()
DEALS = 1000
SEED_PREFIX = "response-search-v1"


class Player:
    """Match the published judge's delayed result delivery and per-half reset."""

    def __init__(self, options=None):
        self.policy = agent.choose_action if options is None else partial(choose_adjusted, options=options)
        self.name = "RiverCount" if options is None else "ResponseStudy"
        self.stats = AgentStats()

    def describe(self):
        return self.name

    def start_half(self):
        self.memory = {}
        self.pending = []
        agent.new_half(AgentConfig(), self.memory)

    def act(self, observation):
        self.flush_hand_results()
        action = self.policy(observation, self.memory)
        if action not in observation.legal_actions:
            raise ValueError(f"Illegal action {action} at {observation}")
        return action

    def hand_end(self, result):
        self.pending.append(result)

    def flush_hand_results(self):
        for result in self.pending:
            agent.on_hand_end(result, self.memory)
        self.pending.clear()


@lru_cache(maxsize=24000)
def _allin_equity(hole0, hole1, board):
    """Conditional seat-zero share after the last chip goes in.

    This is a training score. Hidden cards never enter the policy callback.
    Flop and turn use every remaining runout; preflop uses 512 sampled boards.
    """
    remaining = sorted(set(range(36)).difference(hole0, hole1, board))
    if board:
        draws = np.asarray(list(itertools.combinations(remaining, 5 - len(board))))
    else:
        # Independent of the actual remaining board and of policy RNG streams.
        seed = int.from_bytes(bytes(hole0 + hole1), "big")
        generator = np.random.default_rng(seed)
        draws = generator.permuted(np.broadcast_to(remaining, (512, len(remaining))), axis=1)[:, :5]
    boards = np.sort(np.column_stack((np.broadcast_to(board, (len(draws), len(board))), draws))
                     .astype(np.intp), axis=1)
    indices = equity._BINOMIAL[boards, equity._BOARD_POSITIONS].sum(axis=1)
    board_classes = equity._BOARD_CLASS[indices][:, None]
    scores = []
    for hand in (hole0, hole1):
        pair_types = equity._PAIR_TYPE[tuple(np.asarray(hand)[equity._HOLE_PAIRS])]
        scores.append(equity._PAIR_SCORE[board_classes, pair_types].max(axis=1))
    return float(np.mean((scores[0] > scores[1]) + 0.5 * (scores[0] == scores[1])))


def play(options, seed, deals=DEALS):
    raw = np.zeros((2, deals), dtype=int)
    adjusted = np.zeros((2, deals), dtype=float)
    for seat in (0, 1):
        players = [Player(options), Player()]
        if seat:
            players.reverse()
        for player in players:
            player.start_half()
        for hand in range(deals):
            deal = make_deal(seed, hand)
            game = Game(deal, hand % 2, hand_index=hand)
            while not game.is_terminal:
                actor = game.to_act
                obs = game.observation(actor)
                game.apply(players[actor].act(obs))
            result = game.result()
            raw[seat, hand] = result.nets[seat]
            adjusted[seat, hand] = result.nets[seat]
            if result.went_to_showdown and obs.street != Street.RIVER:
                share = _allin_equity(*deal.hole_cards, obs.board)
                if seat:
                    share = 1 - share
                adjusted[seat, hand] = share * result.pot - game.contributed[seat]
            for s in (0, 1):
                players[s].hand_end(game.hand_result(s))
        for player in players:
            player.flush_hand_results()
    return raw, adjusted


def verify():
    options = replace(BLANK, postflop_call=.14, preflop_reraise=-.04)
    seed = SEED_PREFIX + "-parity"
    observed, _ = play(options, seed, deals=64)
    reference = [play_half(Player(options), Player(), seed, half, 64).nets_agent1 for half in (1, 2)]
    if not np.array_equal(observed, reference):
        raise AssertionError("Trial outcomes differ from the official referee")
    print("Verified both halves against the official referee on 64 duplicate deals.", flush=True)


def summary(samples):
    return {"mean": float(samples.mean()), "se": float(samples.std(ddof=1) / math.sqrt(len(samples)))}


def key(options):
    return tuple(asdict(options).values())


def compact(options):
    return {k: v for k, v in asdict(options).items() if v}


def file_id(options):
    return hashlib.sha256(json.dumps(compact(options), sort_keys=True).encode()).hexdigest()[:12]


def evaluate(options, seeds, directory):
    rows = []
    data_path = directory / file_id(options)
    data_path.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        output = data_path / (seed + ".npz")
        if not output.exists():
            raw, adj = play(options, seed)
            with output.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(f, raw=raw, adjusted=adj, params=np.array(key(options)))
            output.with_suffix(".tmp").replace(output)
        with np.load(output, allow_pickle=False) as f:
            if not np.array_equal(f["params"], key(options)):
                raise ValueError("Mismatched search checkpoint")
            rows.append(np.stack([f["raw"].sum(axis=0), f["adjusted"].sum(axis=0)]) / (2 * BIG_BLIND))
    values = np.column_stack(rows)
    # Half/half blend hedges finite preflop rollout error. Only raw chip
    # outcomes on separate deals determine whether the candidate is stronger.
    blend = values.mean(axis=0)
    per_match = {seed: float(row[0].mean()) for seed, row in zip(seeds, rows)}
    result = {"settings": compact(options), "deals": int(values.shape[1]), "matches": per_match,
              "raw": summary(values[0]), "adjusted": summary(values[1]), "blend": summary(blend)}
    return result


def screen(candidates, stage, matches, directory):
    seeds = [f"{SEED_PREFIX}-{stage}-{i:02d}" for i in range(matches)]
    scored = []
    for i, candidate in enumerate(dict.fromkeys(candidates), 1):
        started = perf_counter()
        record = evaluate(candidate, seeds, directory)
        record["seconds"] = round(perf_counter() - started, 2)
        print(json.dumps({"stage": stage, "candidate": i, **record}), flush=True)
        scored.append((record["blend"]["mean"], candidate, record))
    scored.sort(key=lambda x: x[0], reverse=True)
    (directory / (stage + ".json")).write_text(json.dumps([t[2] for t in scored], indent=2))
    return [t[1] for t in scored]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "response-search")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), ROOT / "response_policy.py", ROOT / "rivercount.py", ROOT / "game.py",
                    ROOT / "equity.py", ROOT / "rl_policy.json", ROOT / "board_scores.npz",
                    ROOT / "preflop.bin.z"]
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest = args.output / "sources.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("Search inputs changed; use a fresh --output directory.")
    manifest.write_text(json.dumps(fingerprint, indent=2))
    warmup()
    verify()
    if args.verify_only:
        return
    dimensions = {
        "preflop_open": (-0.04, 0.04),
        "preflop_reraise": (-0.075, -0.035, 0.045),
        "postflop_open": (-0.085, -0.035, 0.055),
        "postflop_reraise": (-0.12, -0.05, 0.07),
        "preflop_call": (-0.07, 0.05, 0.12),
        "postflop_call": (-0.10, -0.04, 0.06, 0.14),
        "postflop_call_reraise": (-0.13, 0.08, 0.20),
    }
    starters = [replace(BLANK, **{k: v}) for k, vs in dimensions.items() for v in vs]
    stage1 = screen(starters, "train-a", 2, args.output)
    # Recheck every winner on entirely new training seeds; combine only
    # adjustments on different dimensions for the next small search round.
    distinct = {}
    for candidate in stage1:
        field = next(iter(compact(candidate)))
        distinct.setdefault(field, candidate)
    axes = list(distinct.values())[:4]
    combos = [Adjustments(**(compact(a) | compact(b))) for a, b in itertools.combinations(axes, 2)]
    combos.append(Adjustments(**{k: v for c in axes[:3] for k, v in compact(c).items()}))
    stage2 = screen(stage1[:5] + combos, "train-b", 4, args.output)
    stage3 = screen(stage2[:4], "train-c", 6, args.output)
    champion = stage3[0]
    (args.output / "frozen.json").write_text(json.dumps(asdict(champion), indent=2) + "\n")
    screen([champion], "development", 6, args.output)


if __name__ == "__main__":
    main()
