"""Offline Monte Carlo policy improvement in the official short-deck PLO5 simulator.

For each sampled decision, play every alternative on the same hidden deal.
Train on differences in terminal chip rewards relative to the incumbent action.
The optional JSONL dataset contains public decision features and legal action
returns. Only the simulator uses unrevealed cards to complete the hand.
"""

import argparse
import json
import math
import random
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rivercount as agent
from game import Action, AgentConfig, Game, NUM_DEALS, make_deal
from rl_policy import ResidualPolicy, state_key

PROFILES = ("station", "pot", "half", "random", "value", "tight", "self")

@lru_cache(maxsize=128)
def population_style(half_seed):
    """Draw one opponent per simulated match, fixed across the two seats."""
    rng = random.Random(f"population/{half_seed}")
    return (rng.uniform(0.47, 0.62), rng.uniform(0.55, 0.84), rng.uniform(-0.03, 0.23),
            rng.uniform(0.02, 0.36), rng.uniform(0.0, 0.34), rng.uniform(0.15, 0.9))


class Episode:
    def __init__(self, deal_seed, hand, hero, profile, memories, policy):
        self.hero = hero
        self.profile = profile
        self.memories = memories
        self.policy = policy
        self.style = population_style(deal_seed) if profile == "population" else None
        self.seed = zlib.crc32(deal_seed.encode()) + 8191 * hand
        self.equities = {}
        self.overrides = 0

    def equity(self, obs, seat):
        key = seat, obs.street
        if key not in self.equities:
            if not obs.board:
                value = agent.preflop_equity(obs.hole_cards)
            else:
                # A deal has the same street cards in every counterfactual branch.
                rng = random.Random(self.seed + 17 * seat + 67 * obs.street)
                value = agent.estimate_equity(obs.hole_cards, obs.board, rng, 128)
            self.equities[key] = value
        return self.equities[key]

    def opponent(self, obs, seat):
        profile = self.profile
        if profile == "random":
            rng = random.Random(self.seed + 271 * len(obs.action_history))
            return rng.choice(obs.legal_actions)
        if profile in ("station", "half", "pot"):
            action = {"station": Action.CHECK_CALL, "pot": Action.RAISE_POT,
                      "half": Action.RAISE_HALF}[profile]
            return action if action in obs.legal_actions else Action.CHECK_CALL
        equity = self.equity(obs, seat)
        if profile == "population":
            pre, post, caution, bluff, trap, pot_size = self.style
            rng = random.Random(self.seed + 271 * len(obs.action_history))
            price = obs.to_call / (obs.pot + obs.to_call)
            value = equity > (pre if not obs.board else post) + 0.045 * obs.raises_this_street
            can_bluff = equity < 0.42 and rng.random() < bluff / (1 + obs.raises_this_street)
            if obs.raise_to and ((value and rng.random() >= trap) or can_bluff):
                return Action.RAISE_POT if rng.random() < pot_size else Action.RAISE_HALF
            if obs.to_call and equity < price + caution + 0.035 * obs.raises_this_street:
                return Action.FOLD
            return Action.CHECK_CALL
        if profile in ("self", "incumbent"):
            base = agent._pick_action(obs, self.memories[seat], equity)
            if profile == "incumbent":
                return agent._POLICY.select(obs, self.memories[seat], base, equity)
            return base
        tight = profile == "tight"
        threshold = (0.535, 0.62, 0.65, 0.67)[obs.street] + 0.07 * tight
        if Action.RAISE_POT in obs.legal_actions and equity > threshold + 0.055 * obs.raises_this_street:
            return Action.RAISE_POT
        odds = obs.to_call / max(1, obs.pot + obs.to_call)
        if obs.to_call and equity < odds + 0.06 + 0.045 * obs.raises_this_street + 0.11 * tight:
            return Action.FOLD
        return Action.CHECK_CALL

    def play(self, game, *, real, snapshots=None, streets=(), sample_later=False):
        seen = {}
        sampling_rng = random.Random(self.seed + self.hero + 721) if snapshots is not None and sample_later else None
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            memory = self.memories[seat]
            if seat == self.hero:
                equity = self.equity(obs, seat)
                base = agent._pick_action(obs, memory, equity)
                action = self.policy.select(obs, memory, base, equity)
                if real:
                    self.overrides += action != base
                if snapshots is not None and obs.street in streets and (sample_later or obs.street not in seen):
                    key = state_key(obs, memory, base, equity)
                    if key:
                        count, index = seen.get(obs.street, (0, len(snapshots)))
                        seen[obs.street] = count + 1, index
                        if count == 0:
                            snapshots.append((game.clone(), obs, key, equity, base, action))
                        elif sampling_rng.randrange(count + 1) == 0:
                            snapshots[index] = (game.clone(), obs, key, equity, base, action)
            else:
                action = self.opponent(obs, seat)
            if real:
                memory["last"] = (obs, action)
            game.apply(action)
        return game.hand_result(self.hero).my_net


def fresh_memories():
    memories = [{}, {}]
    for memory in memories:
        agent.new_half(AgentConfig(), memory)
    return memories


def train_profile(job):
    profile, deals, seed, states, streets, dataset_dir, shard, sample_later = job
    policy = ResidualPolicy({key: Action(row["action"]) for key, row in states.items()})
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    seed = f"{seed}/{profile}" + (f"/{shard}" if shard else "")
    samples = 0
    dataset = None
    if dataset_dir:
        name = f"{profile}-{shard}.jsonl" if shard else f"{profile}.jsonl"
        dataset = (Path(dataset_dir) / name).open("w", encoding="utf-8")
    try:
        for hero in (0, 1):
            memories = fresh_memories()
            for hand in range(deals):
                game = Game(make_deal(seed, hand), hand % 2, hand_index=hand)
                episode = Episode(seed, hand, hero, profile, memories, policy)
                snapshots = []
                actual = episode.play(game, real=True, snapshots=snapshots, streets=streets, sample_later=sample_later)
                for saved, obs, key, equity, base, chosen in snapshots:
                    returns = {chosen: actual}
                    for action in obs.legal_actions:
                        if action != chosen:
                            branch = saved.clone()
                            branch.apply(action)
                            returns[action] = episode.play(branch, real=False)
                    for action, reward in returns.items():
                        gain = reward - returns[base]
                        row = totals[key][int(action)]
                        row[0] += 1
                        row[1] += gain
                        row[2] += gain * gain
                    if dataset:
                        _, aggression = agent._response_rates(memories[hero], obs.street, None)
                        bet_fold, _ = agent._response_rates(memories[hero], obs.street, Action.RAISE_POT)
                        dataset.write(json.dumps(dict(key=key, deal_id=f"{seed}/{hand}", features=obs.to_dict(), equity=equity,
                                                      aggression=aggression, fold_rate=bet_fold, base=int(base),
                                                      returns=returns), separators=(",", ":")) + "\n")
                    samples += 1
                for seat, memory in enumerate(memories):
                    agent.on_hand_end(game.hand_result(seat), memory)
    finally:
        if dataset:
            dataset.close()
    return profile, samples, {key: dict(rows) for key, rows in totals.items()}


def load_policy(checkpoint):
    if checkpoint is None:
        return ResidualPolicy()
    if checkpoint.suffix == ".npz":
        from neural_policy import NeuralPolicy
        return NeuralPolicy.load(checkpoint, response_rates=agent._response_rates)
    return ResidualPolicy.load(checkpoint)


def compare_profile(job):
    profile, deals, seed, checkpoint, reference = job
    return compare_policy(profile, deals, seed, load_policy(checkpoint), load_policy(reference))


def compare_policy(profile, deals, seed, candidate, reference=None):
    """Paired out-of-sample chip results for any policy exposing select(obs, memory, base, equity)."""
    seed = f"heldout/{seed}/{profile}"
    policies = [ResidualPolicy() if reference is None else reference, candidate]
    paired = [[0] * deals for _ in policies]
    overrides = 0
    for version, policy in enumerate(policies):
        for hero in (0, 1):
            for i in range(deals):
                if i % NUM_DEALS == 0:
                    memories = fresh_memories()
                game = Game(make_deal(seed, i), i % 2, hand_index=i)
                episode = Episode(seed, i, hero, profile, memories, policy)
                paired[version][i] += episode.play(game, real=True)
                if version == 1:
                    overrides += episode.overrides
                for seat, memory in enumerate(memories):
                    agent.on_hand_end(game.hand_result(seat), memory)
    chips_per_bb_hand = 2 * AgentConfig().big_blind
    differences = [(b - a) / chips_per_bb_hand for a, b in zip(*paired)]
    return dict(profile=profile, baseline=mean(paired[0])/chips_per_bb_hand,
                learned=mean(paired[1])/chips_per_bb_hand, delta=mean(differences),
                se=stdev(differences)/math.sqrt(deals) if deals > 1 else None, overrides=overrides)


def run_jobs(function, jobs, workers):
    if workers == 1:
        yield from map(function, jobs)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            yield from pool.map(function, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "rl_policy.json")
    parser.add_argument("--profiles", nargs="+", choices=(*PROFILES, "incumbent", "population"), default=list(PROFILES))
    parser.add_argument("--sample-later", action="store_true",
                        help="sample any eligible decision on each street, including reactions to reraises")
    parser.add_argument("--initial-checkpoint", type=Path, help="start policy iteration with this frozen JSON table")
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="frozen policy to compare against in the paired evaluation; default: base heuristic")
    parser.add_argument("--deals", type=int, default=1000, help="training deals per profile, played from both seats")
    parser.add_argument("--eval-deals", type=int, default=300, help="held-out duplicate deals per profile")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", default="rivercount-train-1")
    parser.add_argument("--streets", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--iterations", type=int, default=1, help="Monte Carlo policy-improvement passes")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--confidence-z", type=float, default=2.0, help="lower confidence bound on action advantage")
    parser.add_argument("--retain-z", type=float, help="additional confidence threshold on each selected action")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset-dir", type=Path, help="optional per-pass/profile training JSONL shards")
    args = parser.parse_args()
    if min(args.workers, args.iterations, args.min_samples, args.deals) <= 0 or args.eval_deals < 0:
        parser.error("workers, iterations, min-samples and deals must be positive; eval-deals must be nonnegative")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("specify each opponent profile once")
    if not args.eval_only:
        if args.checkpoint.suffix != ".json":
            parser.error("use a .json checkpoint for tabular training; .npz neural models support --eval-only")
        checkpoint = dict(format=1, seed=args.seed, deals_per_profile=args.deals, profiles=args.profiles,
                          iterations=args.iterations, min_samples=args.min_samples, confidence_z=args.confidence_z,
                          retain_z=args.retain_z, states={})
        if args.sample_later:
            checkpoint["sample_later"] = True
        if args.initial_checkpoint:
            initial = json.loads(args.initial_checkpoint.read_text(encoding="utf-8"))
            if initial.get("format") != 1 or not isinstance(initial.get("states"), dict):
                parser.error("the initial checkpoint must be a tabular policy")
            checkpoint.update(initial_checkpoint=args.initial_checkpoint.name, states=initial["states"])
        for iteration in range(args.iterations):
            totals = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
            dataset = args.dataset_dir / f"pass-{iteration}" if args.dataset_dir else None
            if dataset:
                dataset.mkdir(parents=True, exist_ok=True)
            # Train independently across half-match shards so larger runs use all
            # available workers and never carry opponent statistics beyond 1,000 hands.
            jobs = [(profile, min(NUM_DEALS, args.deals - start), f"{args.seed}/{iteration}",
                     checkpoint["states"], args.streets, dataset, start // NUM_DEALS, args.sample_later)
                    for start in range(0, args.deals, NUM_DEALS) for profile in args.profiles]
            for profile, count, rows in run_jobs(train_profile, jobs, args.workers):
                print(f"Pass {iteration + 1}: {profile}, {count} paired decisions", flush=True)
                for key, actions in rows.items():
                    for action, row in actions.items():
                        accumulator = totals[key][action]
                        for k, value in enumerate(row):
                            accumulator[k] += value
            # Only adopt decisions with positive estimated benefit after an uncertainty penalty.
            learned = {}
            for key, actions in totals.items():
                best_bound = 0.0
                retained = False
                for action, (n, total, squared) in actions.items():
                    if n < max(2, args.min_samples):
                        continue
                    avg = total / n
                    se = math.sqrt(max(0, squared - n * avg * avg) / (n * (n - 1)))
                    bound = avg - args.confidence_z * se
                    if bound > best_bound:
                        learned[key] = dict(action=action, samples=n, gain_chips=round(avg, 4), se=round(se, 4))
                        best_bound = bound
                        retained = args.retain_z is None or avg > args.retain_z * se
                if not retained:
                    learned.pop(key, None)
            checkpoint["states"] = learned
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            temp.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp.replace(args.checkpoint)
            print(f"Saved {len(learned)} supported overrides to {args.checkpoint}", flush=True)
    if args.eval_deals:
        jobs = [(profile, args.eval_deals, args.seed, args.checkpoint, args.reference_checkpoint)
                for profile in args.profiles]
        for result in run_jobs(compare_profile, jobs, args.workers):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
