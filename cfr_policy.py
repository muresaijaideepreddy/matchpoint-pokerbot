"""CPU inference and public information abstraction for the MCCFR experiment.

Set CFR_MODEL to a checkpoint from tools/train_cfr.py to play it as a bot.
"""

import bisect
import os
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np

from bot_server import create_app
from game import Action


FORMAT = 1
HISTORY_DRAW_FORMAT = 2
EDGES = (
    (0.38, 0.43, 0.47, 0.50, 0.53, 0.56, 0.59, 0.63),
    (0.15, 0.29, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91),
    (0.13, 0.27, 0.40, 0.51, 0.62, 0.73, 0.84, 0.94),
    (0.10, 0.25, 0.40, 0.53, 0.66, 0.77, 0.88, 0.96),
)


def legal_choices(obs):
    if (Action.RAISE_HALF in obs.raise_to and Action.RAISE_POT in obs.raise_to
            and obs.raise_to[Action.RAISE_POT] == obs.raise_to[Action.RAISE_HALF]):
        return tuple(action for action in obs.legal_actions if action != Action.RAISE_HALF)
    return obs.legal_actions


def public_key(obs, actions, history_draw=False):
    """Keep public position, price, remaining depth, raise count, and legal menu."""
    price = obs.to_call / (obs.pot + obs.to_call)
    odds = 0 if price == 0 else (1 if price <= 0.26 else 2)
    depth = (obs.my_stack - obs.to_call) / (obs.pot + obs.to_call)
    spr = bisect.bisect_left((0.8, 2.0, 5.0), depth)
    key = int(obs.street) * 2 + int(obs.is_button)
    key = (key * 3 + odds) * 4 + spr
    key = key * 3 + min(2, obs.raises_this_street)
    key = key * 16 + sum(1 << int(action) for action in actions)
    if not history_draw:
        return key
    # The same pot size can arise after very different ranges took the lead.
    previous_street = int(obs.street) - 1
    last_aggressor = 0
    for event in reversed(obs.action_history):
        if event.street < previous_street:
            break
        if event.street == previous_street and event.action >= Action.RAISE_HALF:
            last_aggressor = 1 if event.is_me else 2
            break
    return key * 3 + last_aggressor


def private_key(public, street, equity):
    return public * 16 + bisect.bisect_right(EDGES[street], equity)


def draw_flag(hole, board):
    from hand_potential import future_equity_variation
    if len(board) in (3, 4):
        threshold = 0.285 if len(board) == 3 else 0.150
        return int(future_equity_variation(hole, board) >= threshold)
    return 0


class CFRPolicy:
    def __init__(self, entries, history_draw=False):
        self.entries = entries
        self.history_draw = history_draw

    @classmethod
    def load(cls, path, min_visits=40):
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format"])
            if version not in (FORMAT, HISTORY_DRAW_FORMAT):
                raise ValueError("Unknown MCCFR checkpoint format")
            keys, sums = data["keys"], data["strategy_sum"]
            support = np.minimum(data["visits"], data["average_visits"])
            if sums.shape != (len(keys), 4) or not np.isfinite(sums).all() or (sums < 0).any():
                raise ValueError("Invalid MCCFR action weights")
            entries = {}
            for key, row, n in zip(keys, sums, support):
                if n >= min_visits and row.sum() > 0:
                    entries[int(key)] = (row / row.sum()).tolist()
        return cls(entries, version == HISTORY_DRAW_FORMAT)

    def select(self, obs, memory, base, equity):
        actions = legal_choices(obs)
        public = public_key(obs, actions, self.history_draw)
        key = private_key(public, obs.street, equity)
        if self.history_draw:
            key *= 2
            if len(obs.board) in (3, 4) and self.entries.get(key) != self.entries.get(key + 1):
                # Computing the feature is only useful when switching the bit
                # can change the distribution or its incumbent fallback.
                visible = obs.hole_cards, obs.board
                if memory.get("cfr_draw_cards") != visible:
                    memory["cfr_draw_cards"] = visible
                    memory["cfr_draw_flag"] = draw_flag(*visible)
                key += memory["cfr_draw_flag"]
        weights = self.entries.get(key)
        if weights is None:
            return base
        if "cfr_rng" not in memory:
            memory["cfr_rng"] = Random(0xCF47)
        return memory["cfr_rng"].choices(actions, weights=[weights[a] for a in actions], k=1)[0]


@lru_cache(maxsize=1)
def _model(path, support):
    return CFRPolicy.load(path, support)


AGENT_NAME = "EquityCFR"
PARTICIPANT = "PokerBot"


def new_half(config, memory):
    import rivercount as agent
    agent.new_half(config, memory)
    path = os.environ.get("CFR_MODEL", str(Path(__file__).with_name("models") / "cfr_40k.npz"))
    memory["cfr"] = _model(path, int(os.environ.get("CFR_MIN_VISITS", "40")))


def on_hand_end(result, memory):
    import rivercount as agent
    agent.on_hand_end(result, memory)


def choose_action(obs, memory):
    import rivercount as agent
    base = agent.choose_action(obs, memory)
    action = memory["cfr"].select(obs, memory, base, memory["equity"])
    memory["last"] = obs, action
    return action


app = create_app(globals())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
