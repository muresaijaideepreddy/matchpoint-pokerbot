from dataclasses import replace
from itertools import permutations

import numpy as np

from cfr_policy import legal_choices
from game import Action, Observation, Street
from matchpoint import AGGRESSION, SELF_PLAY, VALUE, select_expert
from neural_policy import DETAILED_FEATURES, features
from tools.smoke_test import SAMPLE_OBSERVATIONS


def test_detailed_model_is_invariant_to_suit_names():
    obs = Observation.from_dict(SAMPLE_OBSERVATIONS[2][1])
    args = (Action.CHECK_CALL, 0.54, 0.32, 0.23)
    expected = features(obs, *args, detailed=True)
    assert expected.shape == (DETAILED_FEATURES,)
    for suit in permutations(range(4)):
        def relabel(cards):
            return tuple(4 * (c // 4) + suit[c % 4] for c in cards)
        other = replace(obs, board=relabel(obs.board), hole_cards=relabel(obs.hole_cards))
        np.testing.assert_array_equal(expected, features(other, *args, detailed=True))


def test_cfr_counts_equal_all_in_raise_sizes_once():
    obs = Observation.from_dict(SAMPLE_OBSERVATIONS[2][1])
    target = obs.my_bet + obs.my_stack
    same_size = {Action.RAISE_HALF: target, Action.RAISE_POT: target}
    actions = legal_choices(replace(obs, raise_to=same_size))
    assert Action.RAISE_HALF not in actions and Action.RAISE_POT in actions
    assert Action.CHECK_CALL in actions


def test_expert_selection_uses_observed_opportunities():
    assert select_expert({"counts": {}}) is SELF_PLAY
    caller = {(None, False): [0, 60, 0, 60], (None, True): [0, 60, 0, 25]}
    assert select_expert({"counts": caller}) is VALUE
    raiser = {(None, False): [0, 60, 60, 60], (None, True): [0, 60, 25, 25]}
    assert select_expert({"counts": raiser}) is AGGRESSION
    active = {(None, False): [0, 100, 60, 100], (None, True): [10, 100, 25, 75]}
    active[(Street.PREFLOP, False)] = [0, 45, 35, 45]
    active[(Street.PREFLOP, True)] = [2, 40, 18, 32]
    assert select_expert({"counts": active}) is SELF_PLAY
