"""Independent call and raise adjustments around the deployed RiverCount policy."""

from dataclasses import dataclass

import rivercount as agent
from game import Action, Street


@dataclass(frozen=True)
class Adjustments:
    preflop_open: float = 0.0
    preflop_reraise: float = 0.0
    postflop_open: float = 0.0
    postflop_reraise: float = 0.0
    preflop_call: float = 0.0
    postflop_call: float = 0.0
    postflop_call_reraise: float = 0.0


def choose_adjusted(obs, memory, options):
    original = agent.choose_action(obs, memory)
    opened = memory["counts"].get((None, False))
    faced = memory["counts"].get((None, True))
    if opened and faced and opened[3] >= 12 and faced[1] >= 8:
        aggression = opened[2] / opened[3]
        if aggression > 0.86 or (aggression < 0.07 and faced[0] / faced[1] < 0.12):
            return original
    preflop = obs.street == Street.PREFLOP
    # An opening small blind owes a chip, although nobody has raised yet.
    facing_raise = obs.raises_this_street > 0 and obs.to_call > 0
    if preflop:
        raise_offset = options.preflop_reraise if facing_raise else options.preflop_open
        call_margin = options.preflop_call
    else:
        raise_offset = options.postflop_reraise if facing_raise else options.postflop_open
        call_margin = options.postflop_call
        if obs.raises_this_street >= 2:
            call_margin += options.postflop_call_reraise
    if not (raise_offset and obs.raise_to) and not (call_margin and obs.to_call):
        return original
    eq = memory["equity"]
    proposed = original
    if raise_offset and obs.raise_to:
        proposed = agent._pick_action(obs, memory, min(1.0, max(0.0, eq + raise_offset)))
    if proposed < Action.RAISE_HALF:
        # The raise offset only decides whether to raise. Compare calling with
        # folding using the unshifted hand strength and a separate price margin.
        if original < Action.RAISE_HALF and not call_margin:
            proposed = original
        elif not obs.to_call:
            proposed = Action.CHECK_CALL
        else:
            price = min(obs.to_call, obs.my_stack)
            pot = obs.pot + price
            behind = obs.my_stack - price
            _, aggression = agent._response_rates(memory, obs.street, None)
            future = 0 if obs.street == Street.RIVER else min(1, behind / pot)
            realization = (0.035 + 0.06 * aggression + 0.018 * (not obs.is_button)) * future
            share = agent._range_equity(obs, memory, eq)
            proposed = Action.CHECK_CALL if share >= price / pot + realization + call_margin else Action.FOLD
    memory["last"] = obs, proposed
    return proposed
