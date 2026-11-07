"""Frozen MatchPoint strategy with the boosted value expert."""

import os
from pathlib import Path

import matchpoint
from boosted_policy import BoostedPolicy
from bot_server import create_app

AGENT_NAME = "MatchPoint Boosted"
PARTICIPANT = matchpoint.PARTICIPANT
new_half = matchpoint.new_half
on_hand_end = matchpoint.on_hand_end
_DEFAULT = Path(__file__).resolve().parent / "models" / "swirl_boosted.npz"
VALUE = BoostedPolicy.load(os.environ.get("BOOSTED_MODEL", _DEFAULT), matchpoint.rivercount._response_rates)


def choose_action(obs, memory):
    base = matchpoint.rivercount.choose_action(obs, memory)
    expert = matchpoint.select_expert(memory)
    if expert is matchpoint.VALUE:
        expert = VALUE
    action = expert.select(obs, memory, base, memory["equity"])
    memory["last"] = obs, action
    return action


app = create_app(globals())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
