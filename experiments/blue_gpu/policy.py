"""Best response trained against a frozen betting policy. Inference uses our own equity."""
from bisect import bisect_right
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from game import Action
import rivercount

PREFLOP_EDGES=(.32,.36,.39,.41,.43,.45,.47,.49,.51,.53,.55,.57,.59,.62,.65,.70,.76)
POSTFLOP_EDGES=(.08,.16,.24,.32,.40,.46,.52,.58,.64,.70,.76,.82,.87,.91,.95,.98,.995)
BUCKETS=18

def public_key(obs):
    """Price, effective stacks, aggressor and most recent opposing preflop size."""
    street=int(obs.street)
    price=obs.to_call/max(1,obs.pot+obs.to_call)
    depth=(obs.my_stack-obs.to_call)/max(1,obs.pot+obs.to_call)
    odds=0 if price==0 else (1 if price<.27 else 2)
    spr=bisect_right((.5,1.5,4),depth)
    raises=min(3,obs.raises_this_street)
    previous=0
    opening=0
    for event in obs.action_history:
        if event.action>=Action.RAISE_HALF:
            if int(event.street)==0 and not event.is_me and opening==0:
                opening=1+(event.action==Action.RAISE_POT)
            if int(event.street)==street-1:
                previous=1+int(not event.is_me)
    legal=sum(1<<int(a) for a in obs.legal_actions)
    distinct=bool(Action.RAISE_HALF in obs.raise_to and obs.raise_to[Action.RAISE_HALF]!=obs.raise_to.get(Action.RAISE_POT))
    key=street*2+int(obs.is_button)
    for base,value in ((3,odds),(4,spr),(4,raises),(3,previous),(3,opening),(16,legal),(2,int(distinct))):
        key=key*base+value
    return key

def detailed_key(obs):
    """Distinguish complete preflop action paths and pressure on the preceding street."""
    pf=0
    prior_count=0
    prior_size=0
    for ev in obs.action_history:
        if int(ev.street)==0:
            pf=4*pf+int(ev.action)
        if int(ev.street)==int(obs.street)-1 and ev.action>=2:
            prior_count+=1
            prior_size=int(ev.action)
    # Preflop can have at most a limp, four raises, and a closing call.
    return (((public_key(obs)*16384+pf)*3+min(2,prior_count))*4+prior_size)

@lru_cache(maxsize=1)
def weights():
    with np.load(Path(__file__).with_name("response.npz"),allow_pickle=False) as archive:
        return {int(k):i for i,k in enumerate(archive["public_keys"])},archive["action"],int(archive["version"]) if "version" in archive else 1

def shape_class(obs):
    """Estimate future draw instability, board texture and preflop hand diversity."""
    if not obs.board:
        suits=[sum(c%4==s for c in obs.hole_cards) for s in range(4)]
        return int(len({c//4 for c in obs.hole_cards})>=4)+2*int(sum(x>=2 for x in suits)>=2)
    suits=[sum(c%4==s for c in obs.board) for s in range(4)]
    paired=len({c//4 for c in obs.board})<len(obs.board)
    if len(obs.board)==5:return int(paired)+2*int(max(suits)>=3)
    from hand_potential import future_equity_variation
    threshold=.285 if len(obs.board)==3 else .150
    return int(future_equity_variation(obs.hole_cards,obs.board)>=threshold)+2*int(paired or max(suits)>=3)

@lru_cache(maxsize=6000)
def current_strength(hole, board):
    """Current made-hand percentile against sampled legal private hands."""
    from equity import _PAIR_TYPE, _TRIPLE_TYPE, _FIVE_SCORE, _HOLE_PAIRS
    from zlib import crc32
    known=set(hole+board)
    rem=np.array([c for c in range(36) if c not in known])
    hist=np.array(list(combinations(board,3)))
    triples=_TRIPLE_TYPE[tuple(hist.T)]
    ranks=_FIVE_SCORE[:,triples].max(axis=1)
    hero=ranks[_PAIR_TYPE[tuple(np.array(hole)[_HOLE_PAIRS])]].max()
    rng=np.random.default_rng(crc32(bytes(hole+board)))
    hands=rng.permuted(np.broadcast_to(rem,(128,len(rem))),axis=1)[:,:5]
    rr=ranks[_PAIR_TYPE[hands[:,_HOLE_PAIRS[0]],hands[:,_HOLE_PAIRS[1]]]].max(1)
    return float((np.count_nonzero(hero>rr)+.5*np.count_nonzero(hero==rr))/len(rr))

def additional_class(obs,equity):
    if len(obs.board) in (3,4):
        shift=current_strength(tuple(obs.hole_cards),tuple(obs.board))-equity
        return int(shift>=-.14)+int(shift>=.14)
    if not obs.board:
        cnt={}
        for c in obs.hole_cards:cnt[c//4]=cnt.get(c//4,0)+1
        return min(2,sum(v>=2 for v in cnt.values())+int(any(r>=6 and v>=2 for r,v in cnt.items())))
    return min(2,sum(c//4>=7 for c in obs.hole_cards))

def select(obs,memory):
    # rivercount computes legal fallback actions and caches equity across raises.
    base=rivercount.choose_action(obs,memory)
    mapping,table,version=weights()
    state=mapping.get(detailed_key(obs) if version==3 else public_key(obs))
    bucket=bisect_right(PREFLOP_EDGES if int(obs.street)==0 else POSTFLOP_EDGES,memory["equity"])
    if version>=2:
        visible=(obs.hole_cards,obs.board)
        if memory.get("blue_shape_cards")!=visible:
            memory["blue_shape_cards"]=visible
            memory["blue_shape"]=shape_class(obs)
        bucket=4*bucket+memory["blue_shape"]
        if version==3:
            bucket=3*bucket+additional_class(obs,memory["equity"])
    raw=int(table[state,bucket]) if state is not None else -1
    action=Action(raw) if raw>=0 and Action(raw) in obs.legal_actions else base
    memory["last"]=(obs,action)
    return action
