"""NumPy inference for compact boosted action-value trees."""

import numpy as np

from game import Action
from neural_policy import DETAILED_FEATURES, N_FEATURES, features
from rl_policy import state_key


class BoostedPolicy:
    @classmethod
    def load(cls, path, response_rates=None):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            if int(data["boosted_format"]) != 1:
                raise ValueError("Unsupported boosted checkpoint version")
            model.inputs = int(data["inputs"])
            model.depth = int(data["depth"])
            model.trees = int(data["trees"])
            if (model.inputs not in (N_FEATURES, DETAILED_FEATURES) or not 1 <= model.depth <= 8
                    or not 1 <= model.trees <= 512):
                raise ValueError("Invalid boosted checkpoint dimensions")
            for name in ("feature", "threshold", "left", "right", "value", "bias", "supported"):
                setattr(model, name, data[name])
            model.starts = data["starts"]
            model.margin = float(data["margin"])
        size = len(model.feature)
        if (model.starts.shape != (len(Action)*model.trees,) or model.bias.shape != (len(Action),)
                or model.supported.shape != (4, 4, 4) or model.supported.dtype != np.bool_
                or any(a.shape != (size,) for a in (model.feature, model.threshold, model.left, model.right, model.value))
                or any(a.dtype.kind not in "iu" for a in (model.starts, model.feature, model.left, model.right))
                or any(np.any(a < 0) or np.any(a >= size) for a in (model.left, model.right, model.starts))
                or np.any(model.feature < 0) or np.any(model.feature >= model.inputs)
                or not all(np.isfinite(a).all() for a in (model.threshold, model.value, model.bias))
                or not np.isfinite(model.margin) or model.margin <= 0):
            raise ValueError("Invalid boosted checkpoint arrays")
        model.response_rates = response_rates
        return model

    def predict_one(self, x, base):
        # Each feature test advances all four ensembles in one array operation.
        # Leaves point to themselves so shallow branches need no special case.
        pos = self.starts
        for _ in range(self.depth):
            pos = np.where(x[self.feature[pos]] <= self.threshold[pos], self.left[pos], self.right[pos])
        scores = self.value[pos].reshape(4, self.trees).sum(axis=1) + self.bias
        scores[base] = 0  # targets are chip advantages relative to this action
        return scores

    def predict(self, x):
        result = np.empty((len(x), 4))
        for start in range(0, len(x), 128):
            xb = x[start:start+128]
            rows = np.arange(len(xb))[:, None]
            pos = np.broadcast_to(self.starts, (len(xb), len(self.starts)))
            for _ in range(self.depth):
                pos = np.where(xb[rows, self.feature[pos]] <= self.threshold[pos], self.left[pos], self.right[pos])
            scores = self.value[pos].reshape(len(xb), 4, self.trees).sum(axis=2) + self.bias
            scores[np.arange(len(xb)), xb[:, 4:8].argmax(axis=1)] = 0
            result[start:start+len(xb)] = scores
        return result

    def actions(self, x, values=None):
        if values is None:
            values = self.predict(x)
        base = x[:, 4:8].argmax(axis=1)
        allowed = self.supported[x[:, :4].argmax(axis=1), base] & (x[:, 8:12] > 0)
        best = np.where(allowed, values, -np.inf).argmax(axis=1)
        row = np.arange(len(x))
        return np.where(allowed[row, best] & (values[row, best] >= self.margin), best, base)

    def select(self, obs, memory, base, equity):
        supported = self.supported[obs.street, base]
        if not supported.any() or state_key(obs, memory, base, equity) is None:
            return base
        rates = self.response_rates
        if rates is None:
            from rivercount import _response_rates as rates
        _, aggression = rates(memory, obs.street, None)
        fold, _ = rates(memory, obs.street, Action.RAISE_POT)
        x = features(obs, base, equity, aggression, fold, detailed=self.inputs == DETAILED_FEATURES)
        values = self.predict_one(x, base)
        best, gain = base, 0.0
        for action in obs.legal_actions:
            if supported[action] and values[action] > gain:
                best, gain = action, values[action]
        return best if gain >= self.margin else base
