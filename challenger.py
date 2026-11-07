"""RangeScout: estimate betting values against a range of feasible holdings."""

import os
from pathlib import Path

import numpy as np
import uvicorn

import rivercount as incumbent
import equity as evaluator
from bot_server import create_app
from game import Action, Street
from rl_policy import ResidualPolicy


PARTICIPANT = incumbent.PARTICIPANT
AGENT_NAME = "RangeScout"
new_half = incumbent.new_half
_SPECIALIST = ResidualPolicy.load(Path(__file__).with_name("models") / "laptop_mc.json")


def on_hand_end(result, memory):
    last = memory["last"]
    if last is not None and last[0].hand_index == result.hand_index:
        sizes = memory.setdefault("sizes", {})
        for event in last[0].action_history:
            if not event.is_me and event.action >= Action.RAISE_HALF:
                counts = sizes.setdefault(event.street, [0, 0])
                counts[int(event.action == Action.RAISE_POT)] += 1
    incumbent.on_hand_end(result, memory)


def _specialist_action(obs, memory):
    key = obs.hand_index, obs.hole_cards, obs.board
    if memory["equity_key"] != key:
        equity = (incumbent.preflop_equity(obs.hole_cards) if not obs.board
                  else evaluator.estimate_equity(obs.hole_cards, obs.board, memory["rng"]))
        memory["equity_key"], memory["equity"] = key, equity
    equity = memory["equity"]
    base = incumbent._pick_action(obs, memory, equity)
    action = _SPECIALIST.select(obs, memory, base, equity)
    memory["last"] = obs, action
    return action


def _preflop_action(obs, memory):
    equity = incumbent.preflop_equity(obs.hole_cards)
    _, aggression = incumbent._response_rates(memory, Street.PREFLOP, None)
    # Opening ranges can be wide, but reraising commits a large fraction of
    # a 40-blind stack. Reserve those raises for hands that withstand another.
    raises = obs.raises_this_street
    threshold = (0.548, 0.565, 0.579, 0.575, 1.0)[min(raises, 4)]
    threshold += 0.024 * (0.45 - aggression) - 0.008 * obs.is_button
    if Action.RAISE_POT in obs.legal_actions and equity >= threshold:
        return Action.RAISE_POT
    price = obs.to_call / (obs.pot + obs.to_call)
    range_cost = min(raises, 3) * (0.041 - 0.037 * aggression)
    if price and equity < price + range_cost + 0.025:
        return Action.FOLD
    return Action.CHECK_CALL


def _new_particles(obs, memory):
    """Hold the same random opponent hands across streets and remove blockers."""
    board = obs.board[:3]
    remaining = sorted(set(range(36)).difference(obs.hole_cards, board))
    memory["rng"].shuffle(remaining)
    n = 160
    templates = evaluator._DRAW_TEMPLATES[3]
    start = memory["rng"].randrange(len(templates) - n + 1)
    hands = np.sort(np.asarray(remaining)[templates[start:start + n, :5]], axis=1)
    preflop_index = evaluator._BINOMIAL[hands, evaluator._BOARD_POSITIONS].sum(axis=1)
    return {
        "key": (obs.hand_index, tuple(obs.hole_cards), tuple(board)),
        "mask": (np.int64(1) << hands).sum(axis=1),
        "pairs": evaluator._PAIR_TYPE[tuple(hands[:, i] for i in evaluator._HOLE_PAIRS)],
        "hero_pairs": evaluator._PAIR_TYPE[tuple(np.asarray(obs.hole_cards)[evaluator._HOLE_PAIRS])],
        "rank": {Street.PREFLOP: _percentiles(incumbent._PREFLOP[preflop_index], np.ones(n))},
        "likelihood": np.ones(n),
        "seen": 0,
        "boards": {},
    }


def _street_values(particles, board, hole, rng):
    """Estimate each holding's equity from the portion of board visible then."""
    remaining = sorted(set(range(36)).difference(hole, board))
    if len(board) == 3:
        # Each possible turn appears once, and the paired rivers are distinct.
        rng.shuffle(remaining)
        shift = rng.randint(1, len(remaining) - 1)
        runouts = np.column_stack((remaining, np.roll(remaining, shift)))[::2]
    elif len(board) == 4:
        rng.shuffle(remaining)
        runouts = np.asarray(remaining)[::2, None]
    else:
        runouts = np.empty((1, 0), dtype=np.intp)
    public = np.broadcast_to(board, (len(runouts), len(board)))
    completed = np.sort(np.column_stack((public, runouts)), axis=1)
    indices = evaluator._BINOMIAL[completed, evaluator._BOARD_POSITIONS].sum(axis=1)
    rows = evaluator._BOARD_CLASS[indices]
    scores = evaluator._PAIR_SCORE[rows[:, None, None], particles["pairs"]].max(axis=2)
    hero = evaluator._PAIR_SCORE[rows[:, None], particles["hero_pairs"]].max(axis=1)
    blocked = (np.int64(1) << completed).sum(axis=1)
    valid = (blocked[:, None] & particles["mask"]) == 0

    # The score percentile on a completed board approximates showdown equity.
    # Equal scores get half the ties. Invalid holdings cannot enter that CDF.
    # Sort each runout once, then scatter tie-aware percentiles back to the
    # candidate holding's original column. Invalid hands sort to the end.
    sortable = np.where(valid, scores, 2**30)
    order = sortable.argsort(axis=1)
    ranked = np.take_along_axis(sortable, order, axis=1)
    starts = np.column_stack((np.ones(len(ranked), dtype=bool), ranked[:, 1:] != ranked[:, :-1]))
    ends = np.column_stack((starts[:, 1:], np.ones(len(ranked), dtype=bool)))
    positions = np.arange(ranked.shape[1])
    lo = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
    hi = np.minimum.accumulate(np.where(ends, positions + 1, len(positions))[:, ::-1], axis=1)[:, ::-1]
    fractions = np.empty_like(ranked, dtype=float)
    np.put_along_axis(fractions, order, (lo + hi) / (2 * np.maximum(1, valid.sum(axis=1)[:, None])), axis=1)
    strength = (fractions * valid).sum(axis=0)
    count = np.maximum(valid.sum(axis=0), 1)
    outcome = ((hero[:, None] > scores) + 0.5 * (hero[:, None] == scores))
    return strength / count, (outcome * valid).sum(axis=0) / count


def _percentiles(strength, weights):
    order = np.argsort(strength)
    # Weight tied ranks together, which matters on paired river boards.
    values = strength[order]
    mass = np.r_[0.0, np.cumsum(weights[order])]
    lo = np.searchsorted(values, strength)
    hi = np.searchsorted(values, strength, side="right")
    return (mass[lo] + mass[hi]) / (2 * mass[-1])


def _tails(percentiles, fold, raise_rate, spread):
    """Mix a strength-ordered response with frequency-matched random play."""
    if fold + raise_rate > 1:
        raise_rate = 1 - fold
    low = np.clip(0.5 + (fold - percentiles) / spread, 0, 1)
    high = np.clip(0.5 + (percentiles - (1 - raise_rate)) / spread, 0, 1)
    return (0.88 * low + 0.12 * fold) * bool(fold), (0.88 * high + 0.12 * raise_rate) * bool(raise_rate)


def _posterior(obs, memory):
    key = (obs.hand_index, tuple(obs.hole_cards), tuple(obs.board[:3]))
    particles = memory.get("particles")
    if particles is None or particles["key"] != key:
        particles = memory["particles"] = _new_particles(obs, memory)
    for n, street in ((3, Street.FLOP), (4, Street.TURN), (5, Street.RIVER)):
        if n > len(obs.board):
            break
        board = tuple(obs.board[:n])
        if board not in particles["boards"]:
            strength, outcome = _street_values(particles, board, obs.hole_cards, memory["rng"])
            blocked = sum(1 << c for c in board)
            live = (particles["mask"] & blocked) == 0
            particles["rank"][street] = _percentiles(strength, live.astype(float))
            particles["boards"][board] = outcome
    public_mask = sum(1 << c for c in obs.board)
    for i in range(particles["seen"], len(obs.action_history)):
        event = obs.action_history[i]
        if not event.is_me:
            previous = (obs.action_history[i - 1].action if i and obs.action_history[i - 1].street == event.street
                        else None)
            facing = previous in (Action.RAISE_HALF, Action.RAISE_POT)
            fold, aggression = incumbent._response_rates(memory, event.street, previous)
            ranks = particles["rank"][event.street]
            folds, raises = _tails(ranks, fold if facing else 0, aggression, 0.36)
            likelihood = raises if event.action >= Action.RAISE_HALF else 1 - folds - raises
            if event.action >= Action.RAISE_HALF:
                half, pot = memory.get("sizes", {}).get(event.street, (0, 0))
                pot_share = (pot + 2) / (half + pot + 4)
                # Mixed sizes often split a betting range. Temper this weak
                # tell, especially where one size is used almost exclusively.
                pot_cut = 1 - aggression * pot_share
                p_pot = np.clip(0.5 + (ranks - pot_cut) / 0.36, 0, 1)
                mixing = min(0.65, 2 * min(pot_share, 1 - pot_share))
                p_pot = mixing * p_pot + (1 - mixing) * pot_share
                likelihood = likelihood * (p_pot if event.action == Action.RAISE_POT else 1 - p_pot)
            # Opponents vary in how consistently they play marginal hands.
            particles["likelihood"] *= np.maximum(likelihood, 0.035)
    particles["seen"] = len(obs.action_history)
    live = (particles["mask"] & public_mask) == 0
    weights = particles["likelihood"] * live
    weights /= weights.sum()
    # A very concentrated posterior can depend on just a few sampled holdings.
    # Shrink those estimates toward the legal prior until more holdings support them.
    effective = 1 / (weights @ weights)
    if effective < 14:
        blend = (14 - effective) / 28
        weights = (1 - blend) * weights + blend * live / live.sum()
    rank = particles["rank"][obs.street]
    return weights, particles["boards"][tuple(obs.board)], rank


def _postflop_action(obs, memory):
    weights, wins, rank = _posterior(obs, memory)
    pot, price = obs.pot, obs.to_call
    after_call = pot + price
    behind = obs.my_stack - price
    equity = weights @ wins
    # An early marginal call pays again on future streets more often out of position.
    future = (0 if obs.street == Street.RIVER else min(1, behind / after_call))
    best_value = equity * after_call - price - future * after_call * (0.045 if obs.is_button else 0.075)
    best = Action.CHECK_CALL
    if price and best_value < 0:
        best, best_value = Action.FOLD, 0

    aggression = incumbent._response_rates(memory, obs.street, None)[1]
    if not price and not obs.is_button:
        # If we check, the opponent bets a stronger range, which we can fold to.
        _, bet = _tails(rank, 0, aggression, 0.4)
        target = min(obs.my_stack, obs.opp_stack, pot)
        checkback = weights @ ((1 - bet) * wins)
        facing_bet = weights @ (bet * (wins * (pot + 2 * target) - target))
        best_value = max(best_value, pot * checkback + max(0, facing_bet))

    for action, target in obs.raise_to.items():
        cost = target - obs.my_bet
        extra = cost - price
        fold, reraise = incumbent._response_rates(memory, obs.street, action)
        if price:
            fold *= 0.66
        if extra >= behind or obs.raises_this_street + 1 >= memory["config"].raise_cap:
            reraise = 0
        folds, raises = _tails(rank, fold, reraise, 0.35)
        if not reraise:
            raises = 0.0
        called_pot = after_call + 2 * extra
        call_return = wins * called_pot - cost
        future_raise = min(behind - extra, called_pot)
        facing_raise = weights @ (raises * (wins * (called_pot + 2 * future_raise) - cost - future_raise))
        fold_to_raise = -cost * (weights @ np.broadcast_to(raises, weights.shape))
        value = weights @ (folds * pot + (1 - folds - raises) * call_return)
        value += max(facing_raise, fold_to_raise)
        value -= extra * future * (0.035 if obs.street == Street.FLOP else 0.018)
        if value > best_value:
            best, best_value = action, value
    return best


def choose_action(obs, memory):
    opened = memory["counts"].get((None, False))
    facing = memory["counts"].get((None, True))
    if opened and facing and opened[3] >= 10 and facing[1] >= 5:
        aggression = opened[2] / opened[3]
        # Preserve the specialist response to consistent indiscriminate betting or calling.
        if aggression > 0.86 or (aggression < 0.07 and facing[0] / facing[1] < 0.12):
            return _specialist_action(obs, memory)
    action = _preflop_action(obs, memory) if obs.street == Street.PREFLOP else _postflop_action(obs, memory)
    memory["last"] = (obs, action)
    return action


app = create_app(globals())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5008")), access_log=False)
