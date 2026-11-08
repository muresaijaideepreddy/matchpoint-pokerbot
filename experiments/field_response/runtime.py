"""Use a response trained on simulated action histories when a public betting pattern matches."""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from cfr_policy import legal_choices, public_key
from game import Action, Street
from size_mix_router import choose_release_action

EDGES = ((.345,.383,.412,.436,.458,.477,.495,.512,.529,.548,.568,.592,.620,.658,.713),
         (.12,.22,.30,.37,.43,.48,.53,.58,.63,.68,.73,.79,.85,.91,.965),
         (.10,.19,.27,.35,.42,.48,.54,.60,.66,.72,.78,.84,.89,.94,.98),
         (.08,.16,.24,.32,.40,.48,.56,.64,.71,.78,.84,.89,.94,.97,.992))

def context(obs):
    price=obs.to_call
    facing=0 if not price else (3 if obs.raises_this_street>=2 else (1 if price*5 <= 3*(obs.pot-price) else 2))
    return (((int(obs.street)*2+int(obs.is_button))*4+facing)*2+int(bool(obs.raise_to)))

def style(memory):
    x=[]; n=[]
    for street in range(4):
        for facing in (False,True):
            fold,acts,raises,avail=memory["counts"].get((Street(street),facing),(0,0,0,0))
            x.append((raises+1.5)/(avail+3));n.append(avail)
            if facing:
                x.append((fold+1.5)/(acts+3));n.append(acts)
    for half,count in memory.get("preflop_raise_sizes",((0,0),(0,0))):
        x.append((half+1.5)/(count+3));n.append(count)
    return np.array(x),np.array(n)

@lru_cache(None)
def data():
    with np.load(Path(__file__).with_name("field_response.npz"),allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def classify(obs,memory):
    d=data()
    if memory.get("fr_hand")!=obs.hand_index:
        memory["fr_hand"]=obs.hand_index
        x,n=style(memory); w=np.minimum(n/60,1.)
        diff=((d["styles"]-x)**2*w/(.075+d["styles"]*(1-d["styles"]))).sum(axis=1)/max(1,w.sum())
        nearest=int(diff.argmin())
        memory["fr_index"]=nearest if diff[nearest]<.070 and d["enabled"][nearest] else -1
    return memory["fr_index"]
def select(obs,memory,base):
    """Called after Weights 2 observed this decision; update last if we change its action."""
    d=data()
    if obs.hand_index<55:return base
    classify(obs,memory)
    i=memory["fr_index"]
    if i<0:return base
    legal=legal_choices(obs);key=public_key(obs,legal,history_draw=True)
    pos=np.searchsorted(d["keys"],key)
    if pos==len(d["keys"]) or d["keys"][pos]!=key:return base
    bucket=bisect_right(EDGES[obs.street],memory["equity"])
    if d["visits"][i,pos,bucket]<120:return base
    q=d["q"][i,pos,bucket];best=max(legal,key=lambda a:q[a]);was=base if base in legal else Action.RAISE_POT
    a=best if q[best]>q[was]+d["margins"][i] else base
    memory["last"]=obs,a
    return a

def choose_action(obs,memory):
    return select(obs,memory,choose_release_action(obs,memory))
