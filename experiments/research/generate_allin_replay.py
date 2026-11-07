"""Offline-only train_rl entry replacing terminal *training returns* with conditional payouts."""
from itertools import combinations
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from evaluator import omaha_score
from game import Action, Game, Street, make_deal
from tools import train_rl
from tools.train_cfr import allin_share

ORIGINAL_PROFILE = train_rl.train_profile
COUNTS = Counter()


def terminal_label(game, seat, seed, cache):
    result = game.result()
    street = result.history[-1].street
    if not result.went_to_showdown or street == Street.RIVER:
        return result.nets[seat]
    if 0 not in game.stacks or game.contributed[0] != game.contributed[1]:
        raise ValueError("Early showdown without a matched all-in")
    if street not in cache:
        signed_share = allin_share(game.deal, street, 128, seed + 131 * int(street))
        cache[street] = (1 + signed_share) / 2
    seat0 = cache[street]
    return (seat0 if seat == 0 else 1-seat0) * result.pot - game.contributed[seat]


class AveragedEpisode(train_rl.Episode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allin_cache = {}

    def play(self, game, **kwargs):
        # The original simulator still produces the entire hand and subsequent
        # on_hand_end callbacks receive Game.hand_result(), including its actual
        # board and payout. Only the scalar recorded as a training label changes.
        original = super().play(game, **kwargs)
        result = game.result()
        street = result.history[-1].street
        value = terminal_label(game, self.hero, self.seed, self.allin_cache)
        if result.went_to_showdown and street != Street.RIVER:
            COUNTS["actual_path_label" if kwargs["real"] else "counterfactual_branch_label"] += 1
            COUNTS[street.name] += 1
            COUNTS["differed_from_raw"] += abs(value-original) > 1e-8
        return value


def scaled_train_profile(job):
    COUNTS.clear()
    result=ORIGINAL_PROFILE(job)
    profile,_,_,_,_,directory,shard,_=job
    stats=Path(directory).parent/"shard-stats"/Path(directory).name
    stats.mkdir(parents=True,exist_ok=True)
    (stats/f"{profile}-{shard}.json").write_text(json.dumps(dict(COUNTS)))
    return result


def runout(game, street):
    while not game.is_terminal:
        obs = game.observation(game.to_act)
        action = Action.RAISE_POT if obs.street >= street and Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
        game.apply(action)
    return game


def verify():
    tied = 0
    for i in range(100):
        deal = make_deal("scaled-label-verify", i)
        game = runout(Game(deal, i%2),Street.RIVER)
        assert game.result().history[-1].street == Street.RIVER
        for seat in (0,1):
            assert game.hand_result(seat).my_net == terminal_label(game,seat,42,{})
        tied += game.result().winner_seat is None
    assert tied > 0
    deal = make_deal("scaled-label-verify", 3)
    for street,count in ((Street.PREFLOP,128),(Street.FLOP,253),(Street.TURN,22)):
        game = runout(Game(deal,0),street)
        assert game.result().history[-1].street == street
        cache = {}
        pay = [terminal_label(game,s,42,cache) for s in (0,1)]
        assert len(cache) == 1 and abs(sum(pay)) < 1e-10 and pay[0] == terminal_label(game,0,42,{})
        if street != Street.PREFLOP:
            board = deal.board[:3 if street==Street.FLOP else 4]
            live = sorted(set(range(36)).difference(*deal.hole_cards,board))
            wins = []
            for tail in combinations(live,5-len(board)):
                # Independent reference evaluator obeys Omaha's two-hole/three-board rule.
                a,b = [omaha_score(h,board+tail) for h in deal.hole_cards]
                wins.append((a>b)+.5*(a==b))
            assert len(wins)==count
            assert abs(pay[0]-(sum(wins)/count*game.result().pot-game.contributed[0])) < 1e-9
    game=Game(deal,0)
    game.apply(Action.FOLD)
    assert all(terminal_label(game,s,42,{}) == game.hand_result(s).my_net for s in (0,1))
    print(f"Verified fold and full-board payouts, {tied} ties across 100 river hands, zero-sum seat symmetry, exact flop and turn against Omaha reference.",flush=True)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        if "--eval-deals" not in sys.argv or sys.argv[sys.argv.index("--eval-deals")+1] != "0":
            raise ValueError("This averaging wrapper only generates offline training data.")
        train_rl.Episode = AveragedEpisode
        train_rl.train_profile = scaled_train_profile
        train_rl.main()
elif __name__ == "__mp_main__":
    train_rl.Episode = AveragedEpisode
    train_rl.train_profile = scaled_train_profile
