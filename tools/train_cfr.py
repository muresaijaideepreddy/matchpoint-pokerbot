"""Train an external-sampling MCCFR strategy for the full short-deck PLO5 game.

One traversal samples a deal and the non-traversing player's moves, while
enumerating the traverser's legal bets. The finite public betting graph is
built from the referee. Card strengths and observed betting conditions are
bucketed, so this is an imperfect-recall abstraction of the original game.

    python tools/train_cfr.py --iterations 10000 --output runs/cfr.npz
    python tools/train_cfr.py --iterations 100000 --output runs/cfr.npz --resume
    python tools/train_cfr.py --abstraction history-draw --iterations 120000 --output runs/expanded.npz

The history-draw variant records which player last raised on the preceding
street and distinguishes high and low variation in future showdown equity.
Only the acting hand and visible board enter either private abstraction.
"""

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rivercount as agent
import equity as hand_eval
from cfr_policy import FORMAT, HISTORY_DRAW_FORMAT, draw_flag, legal_choices, private_key, public_key
from evaluator import omaha_score, warmup
from game import BOARD_CARDS_VISIBLE, Game, make_deal


@dataclass(slots=True)
class Terminal:
    fold_net: int
    showdown_stake: int
    street: int


@dataclass(slots=True)
class Decision:
    seat: int
    street: int
    key: int
    actions: tuple
    children: list


def betting_graph(history_draw=False):
    """Memoize equivalent public states. Showdown reward uses the sampled deal."""
    cache = {}

    def build(game, last_street, previous_aggressor=-1, current_aggressor=-1):
        if game.is_terminal:
            result = game.result()
            if result.went_to_showdown:
                assert game.contributed[0] == game.contributed[1]
                return Terminal(0, game.contributed[0], last_street)
            return Terminal(result.nets[0], 0, last_street)
        key = (game.street, game.to_act, tuple(game.stacks), tuple(game.bets),
               tuple(game.acted), game.raises_this_street, game.last_increment)
        if history_draw:
            # Current aggressor becomes part of the next street's information.
            key += (previous_aggressor, current_aggressor)
        if key not in cache:
            obs = game.observation(game.to_act)
            actions = legal_choices(obs)
            children = []
            for action in actions:
                branch = game.clone()
                branch.apply(action)
                aggressor = game.to_act if action in obs.raise_to else current_aggressor
                if branch.street == game.street:
                    children.append(build(branch, game.street, previous_aggressor, aggressor))
                else:
                    children.append(build(branch, game.street, aggressor, -1))
            cache[key] = Decision(game.to_act, game.street, public_key(obs, actions, history_draw), actions, children)
        return cache[key]

    return build(Game(make_deal("cfr-graph", 0), button_seat=0), 0)


_FLOP_RUNOUTS = np.array(list(combinations(range(23), 2)))


def allin_share(deal, street, samples, seed):
    """Expected win minus loss for seat 0 after betting stops on this street."""
    board = deal.board[:BOARD_CARDS_VISIBLE[street]]
    blocked = set(deal.hole_cards[0] + deal.hole_cards[1] + board)
    remaining = np.array([card for card in range(36) if card not in blocked])
    if len(board) == 4:
        runouts = remaining[:, None]
    elif len(board) == 3:
        runouts = remaining[_FLOP_RUNOUTS]
    else:
        generator = np.random.default_rng(seed)
        rows = np.broadcast_to(remaining, (samples, len(remaining)))
        runouts = generator.permuted(rows, axis=1)[:, :5]
    boards = np.sort(np.column_stack((np.broadcast_to(board, (len(runouts), len(board))), runouts)), axis=1)
    indices = hand_eval._BINOMIAL[boards.astype(int), hand_eval._BOARD_POSITIONS].sum(axis=1)
    rows = hand_eval._BOARD_CLASS[indices, None]
    scores = []
    for hole in deal.hole_cards:
        pairs = hand_eval._PAIR_TYPE[tuple(np.asarray(hole)[hand_eval._HOLE_PAIRS])]
        scores.append(hand_eval._PAIR_SCORE[rows, pairs].max(axis=1))
    return (int(np.count_nonzero(scores[0] > scores[1])) - int(np.count_nonzero(scores[1] > scores[0]))) / len(runouts)


class InformationSet:
    def __init__(self):
        self.regrets = [0.0] * 4
        self.strategy_sum = [0.0] * 4
        self.visits = 0
        self.average_visits = 0

    def strategy(self, actions):
        positive = [max(0, self.regrets[a]) for a in actions]
        total = sum(positive)
        return [x / total for x in positive] if total else [1.0 / len(actions)] * len(actions)


class Trainer:
    def __init__(self, seed, burnin, samples, allin_samples, history_draw=False):
        self.seed = seed
        self.burnin = burnin
        self.samples = samples
        self.allin_samples = allin_samples
        self.history_draw = history_draw
        self.version = HISTORY_DRAW_FORMAT if history_draw else FORMAT
        self.steps = 0
        self.nodes = 0
        self.table = {}
        self.root = betting_graph(history_draw)

    def iteration(self):
        step = self.steps
        # Seat 0 is always button. Both roles receive uniform cards, and each
        # role becomes traverser on alternate independent deals.
        traverser = step % 2
        deal = make_deal("cfr/deals/" + self.seed, step)
        rng = random.Random(f"cfr/actions/{self.seed}/{step}")
        scores = [omaha_score(h, deal.board) for h in deal.hole_cards]
        winner = (scores[0] > scores[1]) - (scores[0] < scores[1])
        shares = {3: winner}
        bins = {}
        policies = {}

        def visit(node):
            if isinstance(node, Terminal):
                value = node.fold_net
                if node.showdown_stake:
                    if self.allin_samples and node.street not in shares:
                        # The payoff may use both hands after the betting is done.
                        # Average the remaining chance event to reduce regret noise.
                        shares[node.street] = allin_share(deal, node.street, self.allin_samples, step)
                    value = node.showdown_stake * shares.get(node.street, winner)
                return -value if traverser else value
            self.nodes += 1
            position = node.seat, node.street
            if position not in bins:
                hole = deal.hole_cards[node.seat]
                board = deal.board[:BOARD_CARDS_VISIBLE[node.street]]
                if board:
                    # Sample from the acting player's information independently
                    # of chance and action sampling. All branches share its bin.
                    rollout_rng = random.Random(f"cfr/equity/{self.seed}/{step}/{position}")
                    strength = agent.estimate_equity(hole, board, rollout_rng, self.samples)
                else:
                    strength = agent.preflop_equity(hole)
                bins[position] = private_key(0, node.street, strength)
                if self.history_draw:
                    bins[position] = 2 * bins[position] + draw_flag(hole, board)
            stride = 32 if self.history_draw else 16
            key = node.key * stride + bins[position]
            if key not in policies:
                if key not in self.table:
                    self.table[key] = InformationSet()
                info = self.table[key]
                # Freeze sigma throughout this traversal even when branches
                # revisit the same abstract information set.
                policies[key] = info, info.strategy(node.actions)
            info, strategy = policies[key]
            if node.seat == traverser:
                action_values = [visit(child) for child in node.children]
                value = sum(p * u for p, u in zip(strategy, action_values))
                for action, utility in zip(node.actions, action_values):
                    info.regrets[action] += utility - value
                info.visits += 1
                return value
            # Two-player external sampling: visits to the sampled player's
            # infosets weight its average strategy by its own reach.
            if step >= self.burnin:
                for action, prob in zip(node.actions, strategy):
                    info.strategy_sum[action] += prob
                info.average_visits += 1
            pick = rng.random()
            for child, prob in zip(node.children, strategy):
                pick -= prob
                if pick < 0:
                    return visit(child)
            return visit(node.children[-1])

        visit(self.root)
        self.steps += 1

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.table)
        rows = [self.table[k] for k in keys]
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as out:
            np.savez_compressed(out, format=self.version, seed=self.seed, burnin=self.burnin, samples=self.samples,
                                allin_samples=self.allin_samples,
                                steps=self.steps, nodes=self.nodes, keys=np.asarray(keys, dtype=np.uint32),
                                regrets=[r.regrets for r in rows], strategy_sum=[r.strategy_sum for r in rows],
                                visits=[r.visits for r in rows], average_visits=[r.average_visits for r in rows])
        temporary.replace(path)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as data:
            config = (int(data["format"]), str(data["seed"]), int(data["burnin"]), int(data["samples"]))
            if (config != (self.version, self.seed, self.burnin, self.samples)
                    or int(data.get("allin_samples", 0)) != self.allin_samples):
                raise ValueError("Checkpoint settings differ; use its original abstraction, seed, burnin and rollout counts")
            keys, regrets, sums, visits, averages = (data[name] for name in
                                                      ("keys", "regrets", "strategy_sum", "visits", "average_visits"))
            for key, r, s, v, a in zip(keys, regrets, sums, visits, averages):
                cell = InformationSet()
                cell.regrets, cell.strategy_sum = r.tolist(), s.tolist()
                cell.visits, cell.average_visits = int(v), int(a)
                self.table[int(key)] = cell
            self.steps, self.nodes = int(data["steps"]), int(data["nodes"])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iterations", type=int, default=10000, help="target total traversals, including resumed work")
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "cfr.npz")
    p.add_argument("--seed", default="equity-cfr-1")
    p.add_argument("--abstraction", choices=("equity", "history-draw"), default="equity",
                   help="history-draw retains the last prior-street aggressor and splits flop/turn draw potential")
    p.add_argument("--burnin", type=int, default=500, help="iterations before collecting average policy")
    p.add_argument("--samples", type=int, default=128, help="flop and turn equity rollouts per hand")
    p.add_argument("--allin-samples", type=int, default=64,
                   help="preflop all-in runouts; postflop all-ins are exact. Zero disables runout averaging")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if (min(args.iterations, args.samples, args.save_every) < 1 or args.allin_samples < 0
            or not 0 <= args.burnin < args.iterations):
        p.error("positive iterations, samples and save-every required; burnin must be below iterations")
    if args.output.exists() and not args.resume:
        p.error("output already exists; choose another output or pass --resume")
    warmup()
    trainer = Trainer(args.seed, args.burnin, args.samples, args.allin_samples, args.abstraction == "history-draw")
    if args.resume:
        trainer.restore(args.output)
    start = time.perf_counter()
    initial = trainer.steps
    while trainer.steps < args.iterations:
        trainer.iteration()
        if trainer.steps % args.save_every == 0 or trainer.steps == args.iterations:
            trainer.save(args.output)
            rate = (trainer.steps - initial) / (time.perf_counter() - start)
            supported = sum(min(x.visits, x.average_visits) >= 40 for x in trainer.table.values())
            print(f"{trainer.steps} traversals, {trainer.nodes} decision nodes, {len(trainer.table)} infosets, "
                  f"{supported} supported at 40 visits, {rate:.1f} traversals/s", flush=True)


if __name__ == "__main__":
    main()
