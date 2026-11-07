"""Compare two trusted local policies on identical, seat-swapped deals.

    python tools/compare_agents.py --candidate challenger.py --seeds audit1 audit2
    python tools/compare_agents.py --candidate models/neural_candidate.npz --head-to-head-only

Scores are big blinds per hand, half the referee's big blinds per duplicate
deal. This measures strategy with direct callbacks. Use judge_engine.py with
separate servers to test HTTP and the match time bank.
"""

import argparse
import json
import math
import os
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import fmean, stdev
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import warmup
from game import BIG_BLIND, NUM_DEALS, Action, AgentConfig
from judge_engine import AgentStats, coerce_action, combine_halves, play_half


def polarized_action(obs, memory):
    """A held-out style that mixes nut-blocker bluffs with slowplayed value."""
    from rivercount import preflop_equity
    from equity import estimate_equity

    key = (obs.hand_index, obs.hole_cards, obs.board)
    if memory.get("key") != key:
        memory["key"] = key
        memory["equity"] = (estimate_equity(obs.hole_cards, obs.board, memory["equity_rng"], 96)
                            if obs.board else preflop_equity(obs.hole_cards))
    equity = memory["equity"]
    rng = memory["action_rng"]
    price = obs.to_call / (obs.pot + obs.to_call)
    value = equity >= (0.55, 0.69, 0.72, 0.75)[obs.street] + 0.035 * obs.raises_this_street
    if obs.raise_to:
        # The ace of a live flush suit makes the nut flush unavailable to the opponent.
        suits = Counter(c % 4 for c in obs.board)
        blocker = any(c // 4 == 8 and suits[c % 4] >= 2 for c in obs.hole_cards)
        weak = equity < (0.45 if not obs.board else 0.36)
        bluff_rate = 0.12 + 0.12 * obs.is_button + 0.24 * blocker
        if obs.to_call:
            bluff_rate *= 0.45 / max(1, obs.raises_this_street)
        if value:
            trap = not obs.is_button and not obs.to_call and rng.random() < 0.25
            if not trap:
                return Action.RAISE_POT if rng.random() < 0.77 else Action.RAISE_HALF
        elif weak and rng.random() < bluff_rate:
            return Action.RAISE_POT if rng.random() < 0.77 else Action.RAISE_HALF
    if not obs.to_call:
        return Action.CHECK_CALL
    # A bigger bet or another raise on this street tightens the calling range.
    required = price + (0.15 + 0.04 * obs.raises_this_street) * (1 - price)
    return Action.CHECK_CALL if equity >= required else Action.FOLD


def callbacks(spec):
    if spec == "polarized":
        def start(config, memory):
            memory.update(equity_rng=Random(31811), action_rng=Random(72161))
        return spec, start, polarized_action, lambda result, memory: None
    if spec in ("value", "tight"):
        # Reuse the public-observation value bettors from the training harness.
        from tools.train_rl import Episode

        def act(obs, memory):
            if memory.get("hand") != obs.hand_index:
                memory["hand"] = obs.hand_index
                memory["episode"] = Episode("audit-value-rollouts", obs.hand_index, 0, spec, (), None)
            return memory["episode"].opponent(obs, 0)

        return spec, lambda config, memory: None, act, lambda result, memory: None

    path = Path(spec).resolve()
    if path.suffix == ".npz":
        # Both neural and CFR checkpoints are safe NumPy archives; the latter
        # has an average strategy, regret accumulators, and a distinct loader.
        with ZipFile(path) as archive:
            is_cfr = {"regrets.npy", "strategy_sum.npy"}.issubset(archive.namelist())
        if is_cfr:
            from cfr_policy import CFRPolicy, choose_action, on_hand_end
            from rivercount import new_half

            model = CFRPolicy.load(path, int(os.environ.get("CFR_MIN_VISITS", "40")))

            def start(config, memory):
                new_half(config, memory)
                memory["cfr"] = model

            return path.name, start, choose_action, on_hand_end
    if path.suffix in (".json", ".npz"):
        with patch.dict(os.environ, RIVERCOUNT_MODEL=str(path)):
            module = runpy.run_path(str(ROOT / "rivercount.py"))
        name = path.name
    else:
        module = runpy.run_path(str(path))
        name = module["AGENT_NAME"]
    return name, module["new_half"], module["choose_action"], module["on_hand_end"]


class StrategyBot:
    """The judge delivers pending hand results before each decision or at half end."""

    def __init__(self, spec, config, counts=None):
        self.name, self.new_half, self.choose_action, self.on_hand_end = callbacks(spec)
        self.config = config
        self.memory = {}
        self.pending = []
        self.stats = AgentStats()
        self.counts = counts

    def start_half(self):
        self.new_half(self.config, self.memory)

    def describe(self):
        return self.name

    def act(self, obs):
        self.flush_hand_results()
        raw = self.choose_action(obs, self.memory)
        action = coerce_action(raw)
        if action not in obs.legal_actions:
            raise ValueError(f"{self.name} returned {raw!r} on hand {obs.hand_index} {obs.street.name}")
        if self.counts is not None:
            self.counts["decisions"] += 1
            self.counts["raises"] += action >= Action.RAISE_HALF
            self.counts["bets_faced"] += obs.to_call > 0
            self.counts["folds"] += action == Action.FOLD
        return action

    def hand_end(self, result):
        self.pending.append(result)
        if self.counts is not None:
            stage = "showdown" if result.went_to_showdown else "without_showdown"
            self.counts[stage + "_hands"] += 1
            self.counts[stage + "_net_chips"] += result.my_net
            if not result.went_to_showdown:
                street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(result.board)]
                outcome = "won" if result.winner == "me" else "lost"
                self.counts[f"{street}_fold_{outcome}_hands"] += 1
                self.counts[f"{street}_fold_{outcome}_net_chips"] += result.my_net

    def flush_hand_results(self):
        for result in self.pending:
            self.on_hand_end(result, self.memory)
        self.pending.clear()


def scores(first, second, seed, deals, counts=(None, None)):
    config = AgentConfig(num_hands=deals)
    halves = [
        play_half(StrategyBot(first, config, counts[0]), StrategyBot(second, config, counts[1]), seed, half, deals)
        for half in (1, 2)
    ]
    result = combine_halves(*halves)
    return [pair / 2 for pair in result.paired_bb], result.verdict


def summary(values):
    mean = fmean(values)
    se = stdev(values) / math.sqrt(len(values))
    return dict(deals=len(values), bb_per_hand=mean, se=se, ci95=[mean - 1.96 * se, mean + 1.96 * se])


def display(row):
    low, high = row["ci95"]
    return f"{row['bb_per_hand']:+.3f} [{low:+.3f}, {high:+.3f}]"


def compare(candidate, incumbent, opponents, seeds, deals):
    head = []
    deltas = defaultdict(list)
    challenger_scores = defaultdict(list)
    incumbent_scores = defaultdict(list)
    field = []
    seed_rows = []
    counts = Counter(), Counter()
    for seed in seeds:
        h2h, verdict = scores(candidate, incumbent, seed, deals, counts)
        head.extend(h2h)
        row = dict(seed=seed, head_to_head=summary(h2h), verdict=verdict, opponents={})
        print(f"{seed}: candidate vs incumbent {display(row['head_to_head'])} ({verdict})", flush=True)
        by_opponent = []
        for opponent in opponents:
            new, new_verdict = scores(candidate, opponent, seed, deals)
            old, old_verdict = scores(incumbent, opponent, seed, deals)
            diff = [a - b for a, b in zip(new, old)]
            by_opponent.append(diff)
            deltas[opponent].extend(diff)
            challenger_scores[opponent].extend(new)
            incumbent_scores[opponent].extend(old)
            row["opponents"][opponent] = dict(candidate=summary(new), incumbent=summary(old),
                                               difference=summary(diff), candidate_verdict=new_verdict,
                                               incumbent_verdict=old_verdict)
            print(f"  vs {Path(opponent).stem}: candidate {fmean(new):+.3f}, incumbent {fmean(old):+.3f}, "
                  f"paired difference {display(row['opponents'][opponent]['difference'])}", flush=True)
        # Each opponent saw these same deals. Pair across opponents before
        # estimating uncertainty on their equal-weight average.
        if by_opponent:
            field.extend(fmean(per_deal) for per_deal in zip(*by_opponent))
        seed_rows.append(row)
    overall = dict(head_to_head=summary(head), opponents={},
                   head_to_head_counts=dict(candidate=dict(counts[0]), incumbent=dict(counts[1])))
    for opponent in opponents:
        overall["opponents"][opponent] = dict(candidate=summary(challenger_scores[opponent]),
                                              incumbent=summary(incumbent_scores[opponent]),
                                              difference=summary(deltas[opponent]))
    if field:
        overall["field_difference"] = summary(field)
    return dict(candidate=candidate, incumbent=incumbent, deals_per_seed=deals,
                units="big blinds per hand", transport="in-process, unclocked",
                seeds=seed_rows, overall=overall)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, help="bot .py or RiverCount .json/.npz checkpoint")
    parser.add_argument("--incumbent", default=str(ROOT / "rivercount.py"), help="reference bot or checkpoint")
    parser.add_argument("--seeds", nargs="+", default=["comparison-1", "comparison-2"])
    parser.add_argument("--deals", type=int, default=NUM_DEALS, help="duplicate deals per match, per seed")
    field = parser.add_mutually_exclusive_group()
    field.add_argument("--head-to-head-only", action="store_true")
    field.add_argument("--opponents", nargs="+", help="bot paths or profiles value, tight and polarized")
    parser.add_argument("--json", type=Path, help="save per-seed and pooled scores")
    args = parser.parse_args()
    if not 2 <= args.deals <= NUM_DEALS or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"use 2..{NUM_DEALS} duplicate deals per seed and distinct seeds")
    opponents = args.opponents
    if args.head_to_head_only:
        opponents = []
    elif opponents is None:
        opponents = ([str(p) for p in sorted((ROOT / "baseline-bots").glob("baseline_*.py"))]
                     + ["value", "tight", "polarized"])
    if len(set(opponents)) != len(opponents):
        parser.error("specify each opponent once")
    warmup()
    print("Strategy callbacks, no HTTP or time bank. Scores in bb/hand, approximate paired 95% intervals.",
          flush=True)
    results = compare(args.candidate, args.incumbent, opponents, args.seeds, args.deals)
    overall = results["overall"]
    print("Pooled head-to-head:", display(overall["head_to_head"]))
    for name, counts in overall["head_to_head_counts"].items():
        hands = 2 * overall["head_to_head"]["deals"]
        print(f"  {name}: raised {counts['raises'] / max(1, counts['decisions']):.1%} of decisions, "
              f"folded to {counts['folds'] / max(1, counts['bets_faced']):.1%} of bets; net bb/hand "
              f"from showdowns {counts.get('showdown_net_chips', 0) / (hands * BIG_BLIND):+.3f}, "
              f"from folds {counts.get('without_showdown_net_chips', 0) / (hands * BIG_BLIND):+.3f}")
    for opponent, row in overall["opponents"].items():
        print(f"  vs {Path(opponent).stem}: candidate {display(row['candidate'])}, "
              f"incumbent {display(row['incumbent'])}, difference {display(row['difference'])}")
    if opponents:
        print("Equal-weight field difference:", display(overall["field_difference"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
