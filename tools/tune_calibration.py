"""Reproduce SizeReader's laptop policy search against RiverCount.

Run from the project root with python tools/tune_calibration.py. The deal seeds
in this file belong to development. Use separate seeds to evaluate a finalist.
During selection, competing policies see common Monte Carlo draws for each
seat and street. The final diagnostic also measures the server's normal
per-half random stream on exactly the same cards.
"""

import random
import sys
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rivercount as agent
from game import Action, AgentConfig, Game, Street, make_deal
from tools.compare_agents import scores as live_scores


class Calibration(NamedTuple):
    half_bet: float
    pot_bet: float
    after_check: float
    preflop_open: float
    preflop_reraise: float
    oop_check: float
    bluff_below: float
    bluff_half: bool
    min_reraise_equity: float
    upgrade_open: bool


SETTINGS = [Calibration(*row) for row in [
    (0, 0, 0, 0, 0, 0, 0, False, 0, False),
    (.09, -.07, .08, 0, 0, -.025, 0, False, 0, False),
    (.055, -.07, .12, 0, 0, -.04, .30, True, 0, False),
    (.08, -.055, .06, -.02, -.04, -.015, .20, False, .545, True),
    (0, 0, 0, .012, -.035, 0, 0, False, .54, True),
    (0, 0, -.01, 0, 0, 0, .40, True, 0, False),
    (0, 0, .15, 0, 0, -.09, 0, False, 0, False),
    (0, 0, 0, -.055, -.065, 0, 0, False, .56, False),
    (-.03, .06, 0, 0, 0, .04, 0, False, 0, False),
    (.06, -.04, .09, .04, .02, .04, .35, True, 0, True),
    (.11, -.08, .06, 0, 0, 0, .35, False, 0, False),
    (0, 0, 0, .05, -.04, 0, 0, False, 0, True),
    (0, 0, 0, -.02, .08, 0, 0, False, 0, False),
]]


def calibration_action(obs, memory, equity, settings, *, combined=False):
    preflop = obs.street == Street.PREFLOP
    checked_to = not preflop and obs.is_button and not obs.to_call
    last_size = next((a.action for a in reversed(obs.action_history) if a.street == obs.street
                      and not a.is_me and a.action >= Action.RAISE_HALF), None)
    offset = {Action.RAISE_HALF: settings.half_bet, Action.RAISE_POT: settings.pot_bet}.get(last_size, 0)
    if preflop:
        offset *= 0 if combined else 0.45
        offset += settings.preflop_open if obs.raises_this_street < 2 else settings.preflop_reraise
    elif checked_to:
        offset += settings.after_check
    elif not obs.to_call and not obs.is_button:
        offset += settings.oop_check
    action = agent._pick_action(obs, memory, min(.999, max(.001, equity + offset)))
    if checked_to and equity < settings.bluff_below and Action.RAISE_HALF in obs.legal_actions:
        action = Action.RAISE_HALF if settings.bluff_half else Action.RAISE_POT
    if preflop:
        if obs.raises_this_street and action >= Action.RAISE_HALF and equity < settings.min_reraise_equity:
            action = Action.CHECK_CALL
        if settings.upgrade_open and obs.raises_this_street <= 1 and action == Action.RAISE_HALF and equity > .5:
            action = Action.RAISE_POT
    return action


@lru_cache(None)
def equity_at(seed, index, seat, street):
    cards = make_deal(seed, index)
    if street == Street.PREFLOP:
        return agent.preflop_equity(cards.hole_cards[seat])
    board = cards.board[:int(street) + 2]
    rng = random.Random(f"{seed}:{index}:{seat}:{int(street)}")
    return agent.estimate_equity(cards.hole_cards[seat], board, rng)


def trial(settings, seed, deals, *, combined=False):
    paired_chips = [0] * deals
    for hero in (0, 1):
        memories = [{}, {}]
        for memory in memories:
            agent.new_half(AgentConfig(), memory)
        for hand in range(deals):
            game = Game(make_deal(seed, hand), hand % 2, hand_index=hand)
            while not game.is_terminal:
                seat = game.to_act
                obs = game.observation(seat)
                equity = equity_at(seed, hand, seat, obs.street)
                if seat == hero:
                    action = calibration_action(obs, memories[seat], equity, settings, combined=combined)
                else:
                    base = agent._pick_action(obs, memories[seat], equity)
                    action = agent._POLICY.select(obs, memories[seat], base, equity)
                memories[seat]["last"] = obs, action
                game.apply(action)
            paired_chips[hand] += game.hand_result(hero).my_net
            for seat in (0, 1):
                agent.on_hand_end(game.hand_result(seat), memories[seat])
    per_hand = [chips / (2 * AgentConfig().big_blind) for chips in paired_chips]
    return mean(per_hand), pstdev(per_hand) / deals ** .5


def main():
    survivors = list(range(1, len(SETTINGS)))
    stages = [("rc-shift-A", 360, 5), ("rc-shift-B", 720, 3), ("rc-shift-C", 900, 1)]
    for seed, deals, keep in stages:
        control, _ = trial(SETTINGS[0], seed, deals)
        scores = [(trial(SETTINGS[i], seed, deals)[0] - control, i) for i in survivors]
        scores.sort(reverse=True)
        print(seed, "BB/hand change by candidate:", [(i, round(v, 4)) for v, i in scores], flush=True)
        survivors = [i for _, i in scores[:keep]]
        equity_at.cache_clear()
    # Candidate 4 changes the opening and reraising ranges; 10 reads postflop bet sizes.
    combo = SETTINGS[10]._replace(preflop_open=.012, preflop_reraise=-.035,
                                  min_reraise_equity=.54, upgrade_open=True)
    for seed in ("rc-combined-I", "rc-combined-II"):
        value, error = trial(combo, seed, 1000, combined=True)
        challenger = Path(agent.__file__).with_name("calibration_agent.py")
        actual, _ = live_scores(str(challenger), agent.__file__, seed, 1000)
        print(f"{seed}: common draws {value:+.4f} BB/hand (SE {error:.4f}), "
              f"live callbacks {mean(actual):+.4f} BB/hand", flush=True)


if __name__ == "__main__":
    main()
