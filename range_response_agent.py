"""RangeResponse: a public-action preflop planner with a RiverCount fallback."""

import json
import os
from pathlib import Path

import numpy as np
import uvicorn

import rivercount as incumbent
import preflop_solver
from bot_server import create_app


PARTICIPANT = incumbent.PARTICIPANT
AGENT_NAME = "RangeResponse"
new_half = preflop_solver.new_half
on_hand_end = preflop_solver.on_hand_end

model_directory = Path(__file__).resolve().with_name("models")
with np.load(model_directory / "preflop_matchups.npz", allow_pickle=False) as model:
    _MATCHUPS = dict(model)
_SETTINGS = preflop_solver.Settings(**json.loads((model_directory / "preflop_response.json").read_text()))
preflop_solver.betting_tree()


def choose_action(obs, memory):
    return preflop_solver.choose_action(obs, memory, _MATCHUPS, _SETTINGS)


app = create_app(globals())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
