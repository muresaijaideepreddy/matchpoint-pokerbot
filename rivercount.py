"""RiverCount, a heads-up short-deck PLO5 bot. Start with python agent.py."""

import math
import os
import random
import zlib
from pathlib import Path

import numpy as np
import uvicorn

from bot_server import create_app
from equity import estimate_equity
from game import Action, AgentConfig, HandResult, Observation, Street
from rl_policy import ResidualPolicy

PARTICIPANT = "PokerBot"
AGENT_NAME = "RiverCount"

# ===== YOUR CODE HERE =====

# The combinatorial index gives each sorted five-card hand one table entry.
# Load the read-only table once per server, outside the match time bank.
_PREFLOP = np.frombuffer(
    zlib.decompress(Path(__file__).with_name("preflop.bin.z").read_bytes()), dtype="<u2"
)


def preflop_equity(hole):
    index = sum(math.comb(card, position) for position, card in enumerate(sorted(hole), 1))
    return int(_PREFLOP[index]) / 65535


def new_half(config: AgentConfig, memory: dict) -> None:
    memory.update(config=config, rng=random.Random(0xA6A9), counts={},
                  equity_key=None, equity=0.5, last=None)


def _record_response(memory, street, faced, action, could_raise):
    """Share evidence across streets, while retaining street- and size-specific rates."""
    facing = faced in (Action.RAISE_HALF, Action.RAISE_POT)
    keys = [(None, facing), (street, facing)]
    if facing:
        keys.append((street, faced))
    for key in keys:
        counts = memory["counts"].setdefault(key, [0, 0, 0, 0])
        counts[0] += action == Action.FOLD
        counts[1] += 1
        counts[2] += action >= Action.RAISE_HALF
        counts[3] += could_raise


def _response_rates(memory, street, faced):
    facing = faced in (Action.RAISE_HALF, Action.RAISE_POT)
    fold = 0.29 if facing else 0.08
    raise_rate = 0.15 if facing else 0.40
    counts = memory["counts"]
    levels = [(None, facing, 12), (street, facing, 15)]
    if facing:
        levels.append((street, faced, 20))
    for group, situation, prior in levels:
        sample = counts.get((group, situation))
        if sample:
            folds, replies, raises, opportunities = sample
            fold = (fold * prior + folds) / (prior + replies)
            raise_rate = (raise_rate * prior + raises) / (prior + opportunities)
    return fold, raise_rate


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Replay the public actions, including the last response omitted by /act."""
    last = memory["last"]
    if last is None or last[0].hand_index != result.hand_index:
        # The small blind can surrender before the big blind gets a decision.
        if not result.went_to_showdown and result.winner == "me":
            _record_response(memory, Street.PREFLOP, None, Action.FOLD, True)
        return
    obs, my_action = last
    config = memory["config"]
    blinds = [config.big_blind, config.small_blind] if obs.is_button else [config.small_blind, config.big_blind]
    stacks = [config.starting_stack - blind for blind in blinds]  # opponent, then us
    previous, street, raises = None, Street.PREFLOP, 0
    bets = blinds.copy()
    for event in obs.action_history:
        if street != event.street:
            previous, street, raises = None, event.street, 0
            bets = [0, 0]
        seat = int(event.is_me)
        if not event.is_me:
            can_raise = stacks[1] > 0 and stacks[0] > max(0, bets[1] - bets[0]) and raises < config.raise_cap
            _record_response(memory, street, previous, event.action, can_raise)
        raises += event.action >= Action.RAISE_HALF
        stacks[seat] -= event.amount
        bets[seat] = event.street_total
        previous = event.action

    if my_action >= Action.RAISE_HALF:
        # A surviving hand would have generated another observation for us.
        reply = Action.CHECK_CALL if result.went_to_showdown else Action.FOLD
        target = obs.raise_to[my_action]
        can_raise = (obs.my_stack > target - obs.my_bet and obs.opp_stack > target - obs.opp_bet
                     and obs.raises_this_street + 1 < config.raise_cap)
        _record_response(memory, obs.street, my_action, reply, can_raise)
    elif my_action == Action.CHECK_CALL and obs.street == Street.RIVER and not obs.is_button and obs.to_call == 0:
        _record_response(memory, obs.street, None, Action.CHECK_CALL, True)
    memory["last"] = None


def _range_equity(obs, memory, equity):
    """Discount equity when the opponent has taken actions they choose infrequently."""
    exponent = 1.0
    previous, street = None, Street.PREFLOP
    for event in obs.action_history:
        if street != event.street:
            previous, street = None, event.street
        if not event.is_me:
            fold, aggression = _response_rates(memory, street, previous)
            age = (0.50 ** (obs.street - street))
            if street == Street.PREFLOP:
                # Preflop equities cluster near 50% and change sharply on the flop.
                age *= 0.26 if obs.street == Street.PREFLOP else 0.15
            strength = age * (0.48, 0.63, 0.85, 1.12)[street]
            if event.action >= Action.RAISE_HALF:
                exponent -= strength * math.log(max(0.035, aggression))
            elif previous in (Action.RAISE_HALF, Action.RAISE_POT):
                exponent -= 0.65 * strength * math.log(max(0.06, 1.0 - fold))
            elif event.amount == 0 and street != Street.PREFLOP:
                # A check from a frequent bettor leaves a weaker checking range.
                exponent -= 0.25 * strength * min(0.65, aggression)
        previous = event.action
    return equity ** max(0.8, min(5, exponent))


def _pick_action(obs, memory, equity):
    """Compare folding, continuing, and both legal raise sizes in chips."""
    street = obs.street
    pot = obs.pot
    price = min(obs.to_call, obs.my_stack)
    after_call = pot + price
    behind = max(0, obs.my_stack - price)
    equity = _range_equity(obs, memory, equity)
    _, aggression = _response_rates(memory, street, None)

    # A player who bets and reraises almost every chance supplies the bets on
    # later streets too. Buy the next card cheaply, then value-raise the river.
    fold_to_bet, reraise = _response_rates(memory, street, Action.RAISE_POT)
    if aggression > 0.86 and reraise > 0.80 and fold_to_bet < 0.09:
        if street != Street.RIVER and (price or street == Street.PREFLOP or not obs.is_button):
            implied_chips = 0.35 * min(behind, 2 * after_call)
            if price > equity * (after_call + implied_chips) and Action.FOLD in obs.legal_actions:
                return Action.FOLD
            return Action.CHECK_CALL
        if street == Street.RIVER and not price and not obs.is_button:
            return Action.CHECK_CALL

    # Future betting makes a marginal call harder to realize out of position.
    future = min(1.0, behind / max(1, after_call)) if street != Street.RIVER else 0.0
    realization_cost = (0.035 + 0.06 * aggression + 0.018 * (not obs.is_button)) * future
    call_value = (equity - realization_cost) * after_call - price
    best = Action.CHECK_CALL
    best_value = call_value
    if Action.FOLD in obs.legal_actions and call_value < 0:
        best, best_value = Action.FOLD, 0.0

    if not obs.raise_to:
        return best

    # Checking in front of an aggressive player often induces a bet we can raise.
    if not obs.is_button and not price and street != Street.PREFLOP:
        best_value += max(0, 2 * equity - 1) * aggression ** 2 * min(pot, behind)

    for action, target in obs.raise_to.items():
        cost = target - obs.my_bet
        extra = cost - price
        if extra <= 0:
            continue
        fold, reraise = _response_rates(memory, street, action)
        if obs.raises_this_street > 0:
            # A player who has already bet typically defends more than one facing an opening bet.
            fold *= 0.72
        # Removing the weakest folds improves the range that calls the raise.
        selection = (0.19, 0.58, 0.78, 0.94)[street]
        called_equity = max(0.0, equity - selection * fold * (1.0 - equity) / max(0.01, 1.0 - fold))
        value = fold * pot + (1.0 - fold) * (called_equity * (after_call + 2 * extra) - cost)

        # Preserve the option to commit chips after seeing the next board card.
        # Against a passive caller there is still value in building a pot early.
        patience = (0.065, 0.040, 0.020, 0.0)[street]
        if street == Street.PREFLOP and aggression > 0.7:
            patience += 0.15 * aggression
        can_reraise = extra < behind and obs.raises_this_street + 1 < memory["config"].raise_cap
        if can_reraise:
            patience += 0.16 * reraise * (1.0 - equity)
        value -= extra * patience
        if value > best_value:
            best, best_value = action, value
    return best


def _load_policy():
    model = os.environ.get("RIVERCOUNT_MODEL", "rl_policy.json")
    if model == "base":
        return ResidualPolicy()
    path = Path(__file__).resolve().parent / model
    if path.suffix == ".npz":
        from neural_policy import NeuralPolicy
        return NeuralPolicy.load(path, response_rates=_response_rates)
    return ResidualPolicy.load(path)


_POLICY = _load_policy()


def choose_action(obs: Observation, memory: dict) -> Action:
    key = (obs.hand_index, obs.hole_cards, obs.board)
    if memory["equity_key"] != key:
        if obs.street == Street.PREFLOP:
            equity = preflop_equity(obs.hole_cards)
        else:
            equity = estimate_equity(obs.hole_cards, obs.board, memory["rng"], samples=128)
        memory["equity"], memory["equity_key"] = equity, key
    base_action = _pick_action(obs, memory, memory["equity"])
    action = _POLICY.select(obs, memory, base_action, memory["equity"])
    memory["last"] = (obs, action)
    return action

# ===== END OF YOUR CODE =====


app = create_app(globals())


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
