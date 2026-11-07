"""Four compact action-value heads, with per-head rank-one factors folded into shared MLP weights."""
from pathlib import Path
import numpy as np
from game import Action
from neural_policy import NeuralPolicy, features
from rl_policy import state_key

STYLES = ("aggressive","folding","passive","mixed")


def public_features(obs, base, equity, aggression, fold_rate, key):
    style = key.split("/", 1)[0]
    if style not in STYLES:
        raise ValueError("Unknown observed style")
    return np.concatenate((features(obs,base,equity,aggression,fold_rate),
                           np.asarray([name == style for name in STYLES],dtype=np.float32)))


def encode(x, index, lower, inverse_width):
    ramps = np.clip((x[:, index] - lower) * inverse_width, 0., 1.)
    return np.concatenate((x, ramps), axis=1)


class TabEnsemble:
    actions = NeuralPolicy.actions

    @classmethod
    def load(cls, path, response_rates=None):
        self = cls()
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["tabensemble_format"]) != 1:
                raise ValueError("Unsupported TabEnsemble format")
            for name in ("index", "lower", "inverse_width", "w0", "b0", "w1", "b1", "w2", "b2", "supported"):
                setattr(self, name, data[name])
            self.margin = float(data["margin"])
        if self.w0.shape != (4, 32 + len(self.index), 64) or self.w1.shape != (4,64,64) or self.w2.shape != (4,64,4):
            raise ValueError("Invalid TabEnsemble dimensions")
        if self.supported.shape != (4,4,4) or self.supported.dtype != np.bool_ or self.margin <= 0:
            raise ValueError("Invalid support or margin")
        if self.index.dtype.kind not in "iu" or not ((self.index >= 0) & (self.index < 28)).all():
            raise ValueError("Invalid numeric feature index")
        if self.lower.shape != self.index.shape or self.inverse_width.shape != self.index.shape:
            raise ValueError("Invalid fitted numeric knots")
        if not all(np.isfinite(a).all() for a in (self.lower,self.inverse_width,self.w0,self.b0,self.w1,self.b1,self.w2,self.b2)):
            raise ValueError("Nonfinite network parameter")
        self.response_rates = response_rates
        return self

    def predict(self, x):
        embedded = encode(x,self.index,self.lower,self.inverse_width)
        # Keep the member axis first so NumPy dispatches four ordinary matrix products.
        hidden = np.maximum(embedded @ self.w0 + self.b0[:,None,:], 0.)
        hidden = np.maximum(hidden @ self.w1 + self.b1[:,None,:], 0.)
        return (hidden @ self.w2 + self.b2[:,None,:]).mean(axis=0)

    def select(self, obs, memory, base, equity):
        key = state_key(obs,memory,base,equity)
        if key is None or not self.supported[obs.street,base].any():
            return base
        rates = self.response_rates
        if rates is None:
            from rivercount import _response_rates as rates
        _, aggression = rates(memory, obs.street, None)
        fold, _ = rates(memory, obs.street, Action.RAISE_POT)
        batch = public_features(obs,base,equity,aggression,fold,key)[None,:]
        return Action(int(self.actions(batch)[0]))
