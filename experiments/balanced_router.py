"""Use the self-play policy against opponents who mix preflop and flop raises."""

import matchpoint
import rivercount
from game import Street


AGENT_NAME = "BalancedRouter"
PARTICIPANT = matchpoint.PARTICIPANT
new_half = matchpoint.new_half
on_hand_end = matchpoint.on_hand_end


def selects_self_play(memory):
    counts = memory["counts"]
    opening = counts.get((Street.PREFLOP, False))
    facing = counts.get((Street.PREFLOP, True))
    flop = counts.get((Street.FLOP, False))
    if not opening or not facing or not flop:
        return False
    if opening[3] < 35 or facing[3] < 25 or flop[3] < 24:
        return False

    # Bet frequencies count legal opportunities. Folds count all responses,
    # including calls or folds when a reraise was unavailable.
    return (0.30 <= opening[2] / opening[3] <= 0.60
            and 0.16 <= facing[2] / facing[3] <= 0.44
            and 0.09 <= facing[0] / facing[1] <= 0.37
            and 0.25 <= flop[2] / flop[3] <= 0.55)


def choose_action(obs, memory):
    base = rivercount.choose_action(obs, memory)
    expert = matchpoint.select_expert(memory)
    if expert is matchpoint.VALUE and selects_self_play(memory):
        expert = matchpoint.SELF_PLAY
    action = expert.select(obs, memory, base, memory["equity"])
    memory["last"] = obs, action
    return action
