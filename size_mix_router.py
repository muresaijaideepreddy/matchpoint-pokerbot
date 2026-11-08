"""Use self-play against opponents who open and reraise with different bet sizes."""

import matchpoint
import rivercount
from experiments import balanced_router
from game import Action, Street


AGENT_NAME = 'SizeMixRouter'
PARTICIPANT = matchpoint.PARTICIPANT
new_half = matchpoint.new_half
on_hand_end = matchpoint.on_hand_end


def record_preflop_raise(obs, memory):
    if obs.street != Street.PREFLOP or not obs.action_history:
        return
    last = obs.action_history[-1]
    if last.is_me or last.action < Action.RAISE_HALF:
        return
    # A raise is first visible on the next decision. Ignore duplicate requests.
    event = obs.hand_index, len(obs.action_history)
    if memory.get('last_preflop_raise') == event:
        return
    memory['last_preflop_raise'] = event
    counts = memory.setdefault('preflop_raise_sizes', [[0, 0], [0, 0]])
    sizes = counts[int(obs.raises_this_street > 1)]
    sizes[0] += last.action == Action.RAISE_HALF
    sizes[1] += 1


def uses_distinct_sizes(memory):
    sizes = memory.get('preflop_raise_sizes')
    opened = memory['counts'].get((Street.PREFLOP, False))
    if not sizes or not opened or not opened[3]:
        return False
    (small_open, opens), (small_reraise, reraises) = sizes
    return (opens >= 25 and reraises >= 10
            and 0.52 <= opened[2] / opened[3] <= 0.85
            and (small_open + 3) / (opens + 6) >= 0.60
            and (small_reraise + 1) / (reraises + 4) <= 0.26)


def _choose(obs, memory, fallback):
    record_preflop_raise(obs, memory)
    if not uses_distinct_sizes(memory):
        return fallback(obs, memory)
    base = rivercount.choose_action(obs, memory)
    expert = matchpoint.select_expert(memory)
    if expert is matchpoint.VALUE:
        expert = matchpoint.SELF_PLAY
    action = expert.select(obs, memory, base, memory['equity'])
    memory['last'] = obs, action
    return action


def choose_action(obs, memory):
    return _choose(obs, memory, balanced_router.choose_action)


def choose_release_action(obs, memory):
    return _choose(obs, memory, matchpoint.choose_action)
