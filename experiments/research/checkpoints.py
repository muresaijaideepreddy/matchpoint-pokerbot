"""Verify the optional research checkpoints or score a public observation on CPU."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from cfr_policy import CFRPolicy
from experiments.fold_anchored import AnchoredPolicy, encode as anchored_features
from experiments.suit_experts import SuitExpertPolicy
from game import Action, Game, Observation, make_deal
from neural_policy import NeuralPolicy, features
from residual_value import ResidualValue, encode as residual_features
from tabensemble_policy import TabEnsemble, public_features


def demo():
    game = Game(make_deal("research-model-example", 17), 1)
    obs = game.observation(game.to_act)
    return dict(features=obs.to_dict(), base=int(Action.CHECK_CALL), equity=.53,
                aggression=.4, fold_rate=.3, key="mixed")


def score_one(directory, spec, row):
    path = directory / spec["file"]
    if sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    kind = spec["family"]
    if kind == "cfr":
        model = CFRPolicy.load(path)
        if not model.entries:
            raise ValueError(f"{path.name} has no supported information sets")
        return dict(supported_information_sets=len(model.entries))
    obs = Observation.from_dict(row["features"])
    base = Action(row["base"])
    args = (obs, base, row["equity"], row["aggression"], row["fold_rate"])
    if kind == "residual":
        model, x = ResidualValue.load(path), residual_features(*args)
    elif kind == "mlp":
        model = NeuralPolicy.load(path)
        x = features(*args, detailed=model.weights[0].shape[0] != 28)
    elif kind == "tabensemble":
        model, x = TabEnsemble.load(path), public_features(*args, row["key"])
    elif kind == "anchored":
        model, x = AnchoredPolicy(path), anchored_features(*args, row["key"])
    elif kind == "suits":
        model, x = SuitExpertPolicy(path), features(*args, detailed=True)
    else:
        raise ValueError(f"Unknown checkpoint family {kind}")
    values = model.predict(x[None, :])
    if values.shape != (1, 4) or not np.isfinite(values).all():
        raise ValueError(f"Invalid scores from {path.name}")
    action = Action(int(model.actions(x[None, :], values)[0]))
    if base not in obs.legal_actions or action not in obs.legal_actions:
        raise ValueError("Illegal base or selected action")
    return dict(base=base.name, selected=action.name, relative_chips={
        a.name: round(float(values[0, a]-values[0, base]), 5) for a in obs.legal_actions})


def main():
    manifest = json.loads((HERE / "checkpoints.json").read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", type=Path, default=HERE / "weights")
    p.add_argument("--model", choices=tuple(manifest))
    p.add_argument("--row", type=Path, help="Read the first decision in a train_rl JSONL shard")
    args = p.parse_args()
    row = demo()
    if args.row:
        with args.row.open(encoding="utf-8") as stream:
            row = json.loads(next(stream))
    names = [args.model] if args.model else manifest
    for name in names:
        result = score_one(args.weights_dir, manifest[name], row)
        print(json.dumps(dict(checkpoint=name, **result), allow_nan=False))


if __name__ == "__main__":
    main()
