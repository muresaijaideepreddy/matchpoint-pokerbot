"""SizeReader: calibrate the incumbent's betting policy to observed bet sizes."""

import os

import uvicorn

import rivercount as incumbent
from bot_server import create_app
from game import Action, Street


PARTICIPANT = incumbent.PARTICIPANT
AGENT_NAME = "SizeReader"
new_half = incumbent.new_half
on_hand_end = incumbent.on_hand_end


def choose_action(obs, memory):
    key = (obs.hand_index, obs.hole_cards, obs.board)
    if memory["equity_key"] != key:
        equity = (incumbent.preflop_equity(obs.hole_cards) if obs.street == Street.PREFLOP
                  else incumbent.estimate_equity(obs.hole_cards, obs.board, memory["rng"]))
        memory["equity_key"], memory["equity"] = key, equity
    equity = memory["equity"]
    checked_to = obs.street != Street.PREFLOP and obs.is_button and not obs.to_call
    if obs.street == Street.PREFLOP:
        # Put more money in against first bets and tighten the four-bet range.
        offset = 0.012 if obs.raises_this_street < 2 else -0.035
    else:
        size = next((event.action for event in reversed(obs.action_history)
                     if event.street == obs.street and not event.is_me
                     and event.action >= Action.RAISE_HALF), None)
        offset = {Action.RAISE_HALF: 0.11, Action.RAISE_POT: -0.08}.get(size, 0)
        if checked_to:
            offset += 0.06
    action = incumbent._pick_action(obs, memory, min(0.999, max(0.001, equity + offset)))
    if checked_to and equity < 0.35 and Action.RAISE_POT in obs.legal_actions:
        action = Action.RAISE_POT
    if obs.street == Street.PREFLOP:
        if obs.raises_this_street >= 1 and action >= Action.RAISE_HALF and equity < 0.54:
            action = Action.CHECK_CALL
        if obs.raises_this_street <= 1 and action == Action.RAISE_HALF and equity > 0.50:
            action = Action.RAISE_POT
    memory["last"] = obs, action
    return action


app = create_app(globals())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
