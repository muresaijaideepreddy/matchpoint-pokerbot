"""Verify the trainer's shortcuts against the tournament engine."""

from itertools import combinations
from random import Random

import pytest

from cfr_policy import legal_choices, public_key
from evaluator import omaha_score
from game import BOARD_CARDS_VISIBLE, Game, Street, make_deal
from tools.train_cfr import Terminal, allin_share, betting_graph


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN])
def test_postflop_all_in_payoff_averages_every_legal_runout(street):
    deal = make_deal("cfr-runout-test", 12)
    board = deal.board[:BOARD_CARDS_VISIBLE[street]]
    unseen = sorted(set(range(36)).difference(deal.hole_cards[0], deal.hole_cards[1], board))
    payoffs = []
    for future in combinations(unseen, 5 - len(board)):
        a, b = [omaha_score(hand, board + future) for hand in deal.hole_cards]
        payoffs.append((a > b) - (a < b))
    assert allin_share(deal, street, samples=1, seed=41) == sum(payoffs) / len(payoffs)


def test_enriched_graph_matches_referee_information_and_terminal_payoffs():
    # Revisiting an equivalent betting state must retain the prior aggressor.
    root = betting_graph(history_draw=True)
    rng = Random(119)
    for hand in range(120):
        deal = make_deal("cfr-public-graph-test", hand)
        game = Game(deal, button_seat=0)
        node = root
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            assert node.seat == game.to_act and node.street == game.street
            assert node.actions == legal_choices(obs)
            assert node.key == public_key(obs, node.actions, history_draw=True)
            i = rng.randrange(len(node.actions))
            game.apply(node.actions[i])
            node = node.children[i]
        assert isinstance(node, Terminal)
        a, b = [omaha_score(hole, deal.board) for hole in deal.hole_cards]
        assert node.fold_net + node.showdown_stake * ((a > b) - (a < b)) == game.result().nets[0]
