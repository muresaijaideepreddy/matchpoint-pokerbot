"""CPU inference for experimental action values with fixed pot-odds priors."""
from pathlib import Path
import numpy as np

from game import Action
from neural_policy import NeuralPolicy, features
from rl_policy import state_key


CHIPS = 40.
WINDOWS = [set(range(i,i+5)) for i in range(5)] + [{8,0,1,2,3}]


def encode(obs, base, equity, aggression, fold_rate):
    first = features(obs, base, equity, aggression, fold_rate)
    line = [0.] * 24
    for event in obs.action_history:
        if event.action >= Action.RAISE_HALF:
            slot = int(event.action) + (2 if event.is_me else 0)
        elif event.is_me:
            continue
        else:
            slot = int(event.amount > 0)
        line[6*int(event.street)+slot] += .25
    pot = obs.pot
    cost = min(obs.to_call,obs.my_stack)
    call = equity*(pot+cost)-cost
    priors = [0.,call]
    for action in (Action.RAISE_HALF,Action.RAISE_POT):
        target = obs.raise_to.get(action,obs.my_bet+cost)
        bet = max(0,target-obs.my_bet)
        extra = min(obs.opp_stack,max(0,target-obs.opp_bet))
        called_value = equity*(pot+bet+extra)-bet
        priors.append(fold_rate*pot+(1-fold_rate)*called_value)
    board,hand = obs.board,obs.hole_cards
    board_ranks = {c//4 for c in board}
    hand_ranks = {c//4 for c in hand}
    board_suits = [sum(c%4==s for c in board) for s in range(4)]
    hand_suits = [sum(c%4==s for c in hand) for s in range(4)]
    # A straight requires two distinct hole ranks and three distinct board
    # ranks inside one five-rank window. Overlapping ranks cannot be used twice.
    completions = [min(2,len(w & hand_ranks))+min(3,len(w & board_ranks)) for w in WINDOWS]
    best_straight = max(min(n,len(w & (board_ranks | hand_ranks))) for n,w in zip(completions,WINDOWS))
    live_suits = [board_suits[s] for s in range(4) if hand_suits[s]>=2]
    ace_boards = [board_suits[c%4] for c in hand if c//4==8]
    texture = [(5-len(hand_ranks))/4, max(hand_suits)/5, len(hand_ranks & board_ranks)/5,
               max(board_ranks,default=0)/8, max(live_suits,default=0)/5,
               max(ace_boards,default=0)/5, best_straight/5,
               sum(n>=4 and len(w & (board_ranks|hand_ranks))>=4 for n,w in zip(completions,WINDOWS))/6]
    price = cost/max(1,pot+cost)
    interaction = [equity-price, equity*(1-equity)*4, (equity-.5)*aggression,
                   (equity-.5)*fold_rate, (equity-.5)*first[16], fold_rate*first[16]]
    return np.asarray([*first,*line,*texture,*interaction,*(p/CHIPS for p in priors)],dtype=np.float32)


INPUTS = 28+24+8+6+4


class ResidualValue(NeuralPolicy):
    @classmethod
    def load(cls,path,response_rates=None):
        with np.load(Path(path),allow_pickle=False) as d:
            if int(d["residual_format"])!=1 or int(d["inputs"])!=INPUTS:
                raise ValueError("Unsupported residual network")
            width=int(d["width"])
            weights=[d[f"w{i}"] for i in range(6)]
            shapes=[(INPUTS,width),(width,),(width,width),(width,),(width,4),(4,)]
            if any(w.shape!=s or w.dtype!=np.float32 or not np.isfinite(w).all() for w,s in zip(weights,shapes)):
                raise ValueError("Invalid residual weight")
            return cls(weights,d["supported"],float(d["margin"]),response_rates)

    def predict(self,x):
        return super().predict(x)+CHIPS*x[..., -4:]

    def select(self,obs,memory,base,equity):
        if state_key(obs,memory,base,equity) is None or not self.supported[obs.street,base].any():
            return base
        _, aggression = self.response_rates(memory,obs.street,None)
        fold_rate, _ = self.response_rates(memory,obs.street,Action.RAISE_POT)
        x=encode(obs,base,equity,aggression,fold_rate)[None,:]
        return Action(int(self.actions(x)[0]))
