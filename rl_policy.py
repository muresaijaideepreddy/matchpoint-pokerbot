"""A frozen Monte Carlo policy-improvement table. Inference uses public information."""

import json
from pathlib import Path

from game import Action


def state_key(obs, memory, base, equity):
    """Coarse information set for generalizing paired action returns across deals."""
    opened = memory["counts"].get((None, False))
    faced = memory["counts"].get((None, True))
    if not opened or not faced or opened[1] + faced[1] < 18:
        return None
    aggression = (opened[2] + 2) / (opened[3] + 5)
    folds = (faced[0] + 2) / (faced[1] + 6)
    if aggression > 0.78:
        profile = "aggressive"
    elif folds > 0.48:
        profile = "folding"
    elif aggression < 0.10 and folds < 0.20:
        profile = "passive"
    else:
        profile = "mixed"
    odds = obs.to_call / max(1, obs.pot + obs.to_call)
    price = 0 if odds == 0 else (1 if odds <= 0.27 else 2)
    equity_bin = int(equity * (20 if obs.street == 0 else 10))
    short = int(obs.my_stack - obs.to_call <= obs.pot + obs.to_call)
    return "/".join(map(str, (profile, int(obs.street), int(obs.is_button), price, short,
                              min(2, obs.raises_this_street), equity_bin, int(base))))


class ResidualPolicy:
    def __init__(self, actions=None):
        self.actions = {} if actions is None else actions

    @classmethod
    def load(cls, path):
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        if checkpoint.get("format") != 1:
            raise ValueError("Unsupported residual policy checkpoint")
        return cls({key: Action(row["action"]) for key, row in checkpoint["states"].items()})

    def select(self, obs, memory, base, equity):
        key = state_key(obs, memory, base, equity)
        action = self.actions.get(key, base)
        return action if action in obs.legal_actions else base
