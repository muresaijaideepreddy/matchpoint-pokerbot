"""Evaluate the preflop range-response planner with production observations."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from statistics import mean, stdev

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import rivercount as agent
import preflop_solver as solver
from evaluator import warmup
from game import Action, AgentConfig, Game, make_deal
from judge_engine import play_half
from tools.search_responses import Player


PREFIX = "preflop-infoset-v1"


class PlannerBot(Player):
    def __init__(self, table, settings):
        super().__init__()
        self.policy = partial(solver.choose_action, table=table, settings=settings)
        self.name = "RangeResponse"

    def start_half(self):
        self.memory = {}
        self.pending = []
        solver.new_half(AgentConfig(), self.memory)

    def flush_hand_results(self):
        for result in self.pending:
            solver.on_hand_end(result, self.memory)
        self.pending.clear()


def validate(table):
    """A reference opponent's public-frequency counts should match the mirror."""
    _, nodes = solver.betting_tree()
    levels = np.linspace(.22, .76, 32)
    checked = 0
    for seat in (0, 1):
        players = [PlannerBot(table, solver.Settings()), Player()]
        if seat:
            players.reverse()
        for bot in players:
            bot.start_half()
        for i in range(180):
            game = Game(make_deal("preflop-reference-parity", i), i % 2, i)
            while not game.is_terminal:
                actor = game.to_act
                game.apply(players[actor].act(game.observation(actor)))
            for s, player in enumerate(players):
                player.hand_end(game.hand_result(s))
                player.flush_hand_results()
            m = players[seat].memory
            reverse = m["mirror"]
            if reverse["counts"] != players[1-seat].memory["counts"]:
                diff = [(k, v, players[1-seat].memory["counts"].get(k))
                        for k, v in reverse["counts"].items() if v != players[1-seat].memory["counts"].get(k)]
                raise AssertionError(f"Opponent frequency mirror diverged on hand {i}: {diff}")
            if i in (0, 50, 150):
                for node in nodes.values():
                    expected = [agent._POLICY.select(node.obs, reverse, agent._pick_action(node.obs, reverse, q), q)
                                for q in levels]
                    actual = solver.modeled_actions(node.obs, reverse, levels)
                    if not np.array_equal(expected, actual):
                        raise AssertionError(f"Modeled actions differ at {node.history}")
                    checked += len(levels)
        if m["model_hits"] != m["model_trials"]:
            raise AssertionError("Failed to reproduce a reference opponent's showdown action")
    print(f"Verified {checked:,} action predictions and all public count changes in 360 hands.", flush=True)


def trials(table, settings, seed, n=1000):
    nets = []
    fits = []
    for half in (1, 2):
        candidate = PlannerBot(table, settings)
        result = play_half(candidate, Player(), seed, half, n)
        nets.append(result.nets_agent1)
        fits.append([candidate.memory["model_hits"], candidate.memory["model_trials"]])
    return (np.sum(nets, axis=0) / 4).tolist(), fits


def assess(table, configs, phase, seeds, output):
    ranking = []
    for i, config in enumerate(configs):
        values = []
        matches = {}
        opts = asdict(config)
        key = "-".join(str(x) for x in opts.values())
        for n in range(seeds):
            seed = f"{PREFIX}-{phase}-{n}"
            cache = output / f"{key}_{seed}.json"
            if cache.exists():
                sample = json.loads(cache.read_text())
            else:
                bb, hits = trials(table, config, seed)
                sample = dict(values=bb, fit=hits, settings=opts)
                cache.with_suffix(".tmp").write_text(json.dumps(sample))
                cache.with_suffix(".tmp").replace(cache)
            values.extend(sample["values"])
            matches[seed] = mean(sample["values"])
        record = dict(phase=phase, candidate=i+1, settings=opts, matches=matches,
                      mean=mean(values), se=stdev(values) / len(values) ** .5)
        print(json.dumps(record), flush=True)
        ranking.append((record["mean"], config))
    return [x[1] for x in sorted(ranking, key=lambda x: -x[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/preflop-study"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.table, allow_pickle=False) as f:
        table = dict(f)
    warmup()
    validate(table)
    if args.check_only:
        return
    configs = [
        solver.Settings(),
        solver.Settings(future_bets=0., position=0.),
        solver.Settings(future_bets=2., position=.06),
        solver.Settings(min_raises=1),
        solver.Settings(min_raises=2),
        solver.Settings(future_bets=.5, position=.04, margin=.08),
    ]
    best = assess(table, configs, "train-a", 2, args.output)
    finalists = assess(table, best[:3], "train-b", 5, args.output)
    winner = finalists[0]
    (args.output / "frozen.json").write_text(json.dumps(asdict(winner), indent=2))
    assess(table, [winner], "development", 6, args.output)


if __name__ == "__main__":
    main()
