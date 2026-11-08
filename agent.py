"""matchpointv5: Weights 3 play with a per-half stop-loss.

Weights 3 is strong against most of the field but loses to opponents that read its
ranges or adapt to it. The bot tracks its own results within each half:

* Checkpoints at hands 150, 300 and 450 compare the average so far with a floor,
  and after hand 450 a rolling check looks at the most recent 300 hands.
* Normally a trip switches to balanced CFR self-play adjusted by a GPU-trained
  counter to range-reading opponents. If the counter is also behind at its own
  checkpoints, the rest of the half uses the balanced Blueprint strategy
  (experiments/blueprint_brain, see its README for the source).
* While the Blueprint response mode is latched, looser floors apply and a trip goes
  straight to the Blueprint strategy.
* The 450,000-traversal CFR policy answers if the Blueprint strategy ever fails.
"""
import os, traceback, uvicorn
import matchpoint, range_counter, rivercount
from bot_server import create_app
from cfr_policy import CFRPolicy
from game import Action
from matchpoint import PARTICIPANT
from size_mix_router import choose_release_action, record_preflop_raise
from experiments.blue_gpu import gate
from experiments.field_response import runtime as field
from experiments.blueprint_brain import brain as blueprint

CFR450 = CFRPolicy.load(matchpoint.MODELS / "cfr_450k.npz")
blueprint.load()
blueprint.bind_actions(Action)

# (hands completed, switch if big blinds won per hand so far is below this)
STOP_LOSS = ((150, -1.2), (300, -0.1), (450, -0.3))
LATCHED_STOP_LOSS = ((150, -2.0), (300, -1.5), (450, -1.2))
# After hand 450, every LATE_EVERY hands: switch if the last LATE_WINDOW hands average below LATE_FLOOR.
LATE_WINDOW, LATE_FLOOR, LATE_EVERY = 300, -2.5, 50
# (hands since the switch, leave the counter if big blinds won per hand since the switch is below this)
COUNTER_STOP = ((100, 0.0), (400, 0.5))
WEIGHTS3, COUNTER, CFR = 0, 1, 2
AGENT_NAME = "matchpointv5"


def new_half(config, memory):
    gate.new_half(config, memory)
    memory["sl_net"] = 0
    memory["sl_hands"] = 0
    memory["sl_nets"] = []
    memory["sl_mode"] = WEIGHTS3
    memory["sl_switch_net"] = 0
    memory["sl_switch_hands"] = 0
    memory["sl_blueprint"] = None


def to_fallback(memory):
    memory["sl_mode"] = CFR
    try:
        memory["sl_blueprint"] = blueprint.Brain()
    except Exception:
        traceback.print_exc()
        memory["sl_blueprint"] = None


def behind(memory, n):
    latched = bool(memory.get("blue_gpu_latched"))
    for hands, floor in LATCHED_STOP_LOSS if latched else STOP_LOSS:
        if n == hands and memory["sl_net"] / 2 / n < floor:
            return True
    if n > 450 and n % LATE_EVERY == 0 and n >= LATE_WINDOW:
        return sum(memory["sl_nets"][-LATE_WINDOW:]) / 2 / LATE_WINDOW < LATE_FLOOR
    return False


def on_hand_end(result, memory):
    matchpoint.on_hand_end(result, memory)
    if memory["sl_blueprint"] is not None:
        try:
            memory["sl_blueprint"].hand_over(result)
        except Exception:
            traceback.print_exc()
            memory["sl_blueprint"] = None
    memory["sl_net"] += result.my_net
    memory["sl_hands"] += 1
    memory["sl_nets"].append(result.my_net)
    n = memory["sl_hands"]
    if memory["sl_mode"] == WEIGHTS3:
        if behind(memory, n):
            memory["sl_switch_net"] = memory["sl_net"]
            memory["sl_switch_hands"] = n
            if memory.get("blue_gpu_latched"):
                to_fallback(memory)
            else:
                memory["sl_mode"] = COUNTER
    elif memory["sl_mode"] == COUNTER:
        k = n - memory["sl_switch_hands"]
        for hands, floor in COUNTER_STOP:
            if k == hands and (memory["sl_net"] - memory["sl_switch_net"]) / 2 / k < floor:
                to_fallback(memory)
                break


def safe_action(obs, memory):
    base = rivercount.choose_action(obs, memory)
    if memory["sl_mode"] == COUNTER:
        action = matchpoint.SELF_PLAY.select(obs, memory, base, memory["equity"])
        action = range_counter.select(obs, memory, action)
    else:
        action = None
        if memory["sl_blueprint"] is not None:
            try:
                choice = Action(memory["sl_blueprint"].choose(obs))
                if choice in obs.legal_actions:
                    action = choice
            except Exception:
                traceback.print_exc()
                memory["sl_blueprint"] = None
        if action is None:
            action = CFR450.select(obs, memory, base, memory["equity"])
    memory["last"] = obs, action
    return action


def weights3_action(obs, memory):
    record_preflop_raise(obs, memory)
    if obs.hand_index >= 55:
        if memory.get("fr_hand") != obs.hand_index:
            field.classify(obs, memory)
        if memory["fr_index"] >= 0:
            return field.select(obs, memory, choose_release_action(obs, memory))
    if not memory["blue_gpu_latched"] and gate.detect(memory):
        memory["blue_gpu_latched"] = True
    if memory["blue_gpu_latched"]:
        return gate.policy.select(obs, memory)
    return choose_release_action(obs, memory)


def choose_action(obs, memory):
    if memory["sl_mode"] != WEIGHTS3:
        return safe_action(obs, memory)
    return weights3_action(obs, memory)


app = create_app(globals())
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5008)), access_log=False)
