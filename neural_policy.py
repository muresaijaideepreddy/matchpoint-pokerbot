"""NumPy inference for the optional action-value model trained by tools/fit_neural.py."""

import numpy as np

from game import Action, Street
from rl_policy import state_key


# The first three groups are the street, incumbent action, and legal action mask.
N_FEATURES = 28
DETAILED_FEATURES = N_FEATURES + 4 * 18 + 4 * 6
HIDDEN = 64
FORMAT = 1


def features(obs, base, equity, aggression, fold_rate, detailed=False):
    total = max(1, obs.pot + obs.my_stack + obs.opp_stack)
    price = min(obs.to_call, obs.my_stack)
    after_call = max(1, obs.pot + price)
    board_ranks = [card // 4 for card in obs.board]
    board_suits = [card % 4 for card in obs.board]
    half_cost = obs.raise_to.get(Action.RAISE_HALF, obs.my_bet) - obs.my_bet
    pot_cost = obs.raise_to.get(Action.RAISE_POT, obs.my_bet) - obs.my_bet
    opp_raises = sum(not a.is_me and a.action >= Action.RAISE_HALF for a in obs.action_history)
    values = [*(obs.street == street for street in Street),
              *(base == action for action in Action),
              *(action in obs.legal_actions for action in Action),
              equity, aggression, fold_rate, obs.is_button, obs.pot / total,
              price / after_call, price / total, obs.my_stack / total, obs.opp_stack / total,
              half_cost / total, pot_cost / total, obs.raises_this_street / 4,
              (len(board_ranks) - len(set(board_ranks))) / 3,
              max(board_suits.count(suit) for suit in range(4)) / 5,
              min(opp_raises, 5) / 5, min(1, max(0, obs.my_stack - price) / after_call)]
    if detailed:
        # Suit names are interchangeable. Canonicalize using the public board
        # first, then hole cards. This preserves blockers and suited draws.
        board_masks, hand_masks = [0] * 4, [0] * 4
        for card in obs.board:
            board_masks[card % 4] |= 1 << (card // 4)
        for card in obs.hole_cards:
            hand_masks[card % 4] |= 1 << (card // 4)
        for public, private in sorted(zip(board_masks, hand_masks), reverse=True):
            for mask in (public, private):
                values.extend((mask >> rank) & 1 for rank in range(9))
        # For each street distinguish a check, call, half-pot raise and pot
        # raise by the opponent. Preserve the sizes of our own raises as well.
        line = [0.] * 24
        for event in obs.action_history:
            if event.action >= Action.RAISE_HALF:
                slot = int(event.action) + (2 if event.is_me else 0)
            elif event.is_me:
                continue
            else:
                slot = int(event.amount > 0)
            line[6 * int(event.street) + slot] += 0.25
        values.extend(line)
    return np.asarray(values, dtype=np.float32)


class NeuralPolicy:
    def __init__(self, weights, supported, margin, response_rates):
        self.weights = weights
        self.supported = supported
        self.margin = margin
        self.response_rates = response_rates

    @classmethod
    def load(cls, path, response_rates=None):
        """Supply agent._response_rates when loading directly in the server."""
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format"].item())
            if version not in (FORMAT, 2):
                raise ValueError("Unsupported neural policy format")
            width = HIDDEN if version == FORMAT else int(data["width"].item())
            inputs = N_FEATURES if version == FORMAT else int(data["inputs"].item())
            if not 1 <= width <= 1024 or inputs not in (N_FEATURES, DETAILED_FEATURES):
                raise ValueError("Unsupported model dimensions")
            shapes = ((inputs, width), (width,), (width, width),
                      (width,), (width, len(Action)), (len(Action),))
            weights = []
            for index, shape in enumerate(shapes):
                array = data[f"w{index}"]
                if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid neural policy weight w{index}")
                weights.append(array)
            supported = data["supported"]
            margin = float(data["margin"].item())
            if supported.shape != (4, 4, 4) or supported.dtype != np.bool_:
                raise ValueError("Invalid action support mask")
            if not np.isfinite(margin) or margin <= 0:
                raise ValueError("Invalid override margin")
        return cls(weights, supported, margin, response_rates)

    def predict(self, x):
        """Return four action scores in chips, comparable within each decision."""
        w0, b0, w1, b1, w2, b2 = self.weights
        h = np.maximum(0, x @ w0 + b0)
        return np.maximum(0, h @ w1 + b1) @ w2 + b2

    def actions(self, x, values=None):
        """Choose from rows of encoded features using the same guards as live play."""
        if values is None:
            values = self.predict(x)
        base = x[:, 4:8].argmax(axis=1)
        street = x[:, :4].argmax(axis=1)
        allowed = self.supported[street, base] & (x[:, 8:12] > 0)
        best = np.where(allowed, values, -np.inf).argmax(axis=1)
        rows = np.arange(len(x))
        gain = values[rows, best] - values[rows, base]
        return np.where(allowed[rows, best] & (gain >= self.margin), best, base)

    def select(self, obs, memory, base, equity):
        # Replay collection omits decisions until enough opponent history exists.
        if state_key(obs, memory, base, equity) is None or not self.supported[obs.street, base].any():
            return base
        rates = self.response_rates
        if rates is None:
            from rivercount import _response_rates as rates
        _, aggression = rates(memory, obs.street, None)
        fold_rate, _ = rates(memory, obs.street, Action.RAISE_POT)
        x = features(obs, base, equity, aggression, fold_rate,
                     detailed=self.weights[0].shape[0] == DETAILED_FEATURES)
        return Action(int(self.actions(x[None, :])[0]))
