"""GPU-trained counter to range-reading opponents, applied on top of balanced CFR.

range_response.npz holds, for each public betting state and own-equity bucket, the
counterfactual value of each action against a behavior model fitted from match logs.
An action replaces CFR's only where it was visited often enough in training and its
value exceeds CFR's choice by MARGIN chips.
"""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import numpy as np
from cfr_policy import legal_choices, public_key
from game import Action
from experiments.field_response.runtime import EDGES

MARGIN = 1.0
MIN_VISITS = 120.0


@lru_cache(None)
def data():
    with np.load(Path(__file__).with_name("range_response.npz"), allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def select(obs, memory, base):
    d = data()
    legal = legal_choices(obs)
    key = public_key(obs, legal, history_draw=True)
    pos = int(np.searchsorted(d["keys"], key))
    if pos == len(d["keys"]) or d["keys"][pos] != key:
        return base
    bucket = bisect_right(EDGES[obs.street], memory["equity"])
    if d["visits"][0, pos, bucket] < MIN_VISITS:
        return base
    q = d["q"][0, pos, bucket]
    best = max(legal, key=lambda a: q[a])
    was = base if base in legal else Action.RAISE_POT
    return best if q[best] > q[was] + MARGIN else base
