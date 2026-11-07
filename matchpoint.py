"""MatchPoint: a self-play strategy with responses to common heads-up betting patterns."""

import os
from pathlib import Path

import uvicorn

import rivercount
from bot_server import create_app
from cfr_policy import CFRPolicy
from neural_policy import NeuralPolicy
from rl_policy import ResidualPolicy


AGENT_NAME = "MatchPoint"
PARTICIPANT = rivercount.PARTICIPANT
MODELS = Path(__file__).resolve().parent / "models"
SELF_PLAY = CFRPolicy.load(os.environ.get("MATCHPOINT_CFR", MODELS / "cfr_150k.npz"))
VALUE = NeuralPolicy.load(MODELS / "swirl_compact96.npz", response_rates=rivercount._response_rates)
AGGRESSION = ResidualPolicy.load(MODELS / "laptop_mc.json")
new_half = rivercount.new_half
on_hand_end = rivercount.on_hand_end


def select_expert(memory):
    opened = memory["counts"].get((None, False))
    faced = memory["counts"].get((None, True))
    if opened is None or faced is None or opened[3] < 24 or faced[1] < 20:
        return SELF_PLAY
    # Shrink early frequencies toward a moderate opponent. The denominators
    # for raises count opportunities: a player can't reraise an all-in.
    open_raise = (opened[2] + 2) / (opened[3] + 5)
    reraise = (faced[2] + 1) / (faced[3] + 5)
    if open_raise > 0.83 and reraise > 0.72:
        return AGGRESSION
    pre_open = memory["counts"].get((rivercount.Street.PREFLOP, False))
    pre_faced = memory["counts"].get((rivercount.Street.PREFLOP, True))
    if pre_open and pre_faced and pre_open[3] >= 18 and pre_faced[3] >= 12:
        p_open = (pre_open[2] + 2) / (pre_open[3] + 5)
        p_reraise = (pre_faced[2] + 1) / (pre_faced[3] + 5)
        p_fold = (pre_faced[0] + 2) / (pre_faced[1] + 7)
        if p_open > 0.61 and p_reraise > 0.22 and p_fold < 0.21:
            return SELF_PLAY
    return VALUE


def choose_action(obs, memory):
    base = rivercount.choose_action(obs, memory)
    expert = select_expert(memory)
    action = expert.select(obs, memory, base, memory["equity"])
    memory["last"] = obs, action
    return action


app = create_app(globals())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
