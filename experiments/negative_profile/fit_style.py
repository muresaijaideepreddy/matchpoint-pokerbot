"""Append one disabled profile learned from separate public development traces."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/app")
from game import Action, Game, AgentConfig, make_deal
import rivercount
from size_mix_router import record_preflop_raise
from experiments.field_response.runtime import style


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = Path("/app/experiments/field_response/field_response.npz")
paths = sorted(Path("/training").glob("frozen-vs-*/clock-linux/gpu-compact96-linux-public.json"))
assert len(paths) == 2
plan = json.loads(Path("/training/plan.json").read_bytes())
seen_seeds = set()
styles = []
sources = []
for path in paths:
    report = json.loads(path.read_bytes())
    assert report["seed"] in plan["seeds"] and report["seed"] not in seen_seeds
    seen_seeds.add(report["seed"])
    assert report["staging"]["entrant"]["image_id"] == plan["parent_image"]
    assert report["staging"]["opponent"]["image_id"] == plan["opponent"]["image_id"]
    assert report["both_completed_cleanly"]
    assert report["match"]["num_deals"] == 1000
    assert {half["half"] for half in report["halves"]} == {1, 2}
    sources.append(dict(path=path.relative_to("/training").as_posix(), sha256=sha(path),
                        seed=report["seed"], staging=report["staging"]))
    for half in report["halves"]:
        assert len(half["logs"]) == 1000
        assert {hand["deal_index"] for hand in half["logs"]} == set(range(1000))
        seat = half["half"] - 1
        memory = {}
        rivercount.new_half(AgentConfig(), memory)
        for hand in half["logs"]:
            index = hand["deal_index"]
            game = Game(make_deal(report["seed"], index), hand["button_seat"], hand_index=index)
            for event in hand["actions"]:
                who = game.to_act
                observation = game.observation(who)
                action = Action[event["action"]]
                assert event["seat"] == who
                if who == seat:
                    record_preflop_raise(observation, memory)
                    memory["last"] = (observation, action)
                game.apply(action)
            rivercount.on_hand_end(game.hand_result(seat), memory)
        x, n = style(memory)
        styles.append(x)

assert seen_seeds == set(plan["seeds"])

with np.load(source, allow_pickle=False) as archive:
    data = {key: archive[key].copy() for key in archive.files}
old = {key: value.copy() for key, value in data.items()}
name = "negative-dimi-development"
assert name not in data["names"]
data["names"] = np.array([*data["names"], name])
data["styles"] = np.concatenate([data["styles"], np.mean(styles, axis=0)[None]])
data["enabled"] = np.append(data["enabled"], False)
data["margins"] = np.append(data["margins"], np.array([1], dtype=data["margins"].dtype))
for key in ("visits", "q"):
    data[key] = np.concatenate([data[key], np.zeros_like(data[key][:1])])
for key, value in old.items():
    assert np.array_equal(value, data[key] if key == "keys" else data[key][:-1])
out = Path("/output/field_response.npz")
np.savez_compressed(out, **data)
record = dict(original_sha256=sha(source), candidate_sha256=sha(out), added_profile=name,
              training_matches=2, training_played_hands=4000, training_sources=sources,
              half_profiles=[x.tolist() for x in styles], mean_profile=np.mean(styles, axis=0).tolist(),
              rule="Append disabled nearest-neighbor style; all existing rows and runtime unchanged.")
Path("/output/training.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
