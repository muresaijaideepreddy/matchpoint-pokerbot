"""MatchPoint switches to a trained response when preflop sizing identifies the style."""
import math
import os
import uvicorn
import matchpoint
import size_mix_router
from bot_server import create_app
from experiments.blue_gpu import policy
from game import Street

AGENT_NAME="MatchPoint"
PARTICIPANT=matchpoint.PARTICIPANT
on_hand_end=matchpoint.on_hand_end

def new_half(config,memory):
    matchpoint.new_half(config,memory)
    memory["blue_gpu_latched"]=False

def detect(memory):
    """Compare two empirically observed raise-size rates to the best single rate."""
    sizes=memory.get("preflop_raise_sizes")
    opens=memory["counts"].get((Street.PREFLOP,False))
    if sizes is None or opens is None:return False
    (small_open,n_open),(small_raise,n_raise)=sizes
    if n_open<7 or n_raise<3 or opens[3]<7:return False
    if not .43<=(opens[2]+2)/(opens[3]+4)<=.88:return False
    if (small_open+2)/(n_open+4)<.59 or (small_raise+1)/(n_raise+4)>.31:return False
    half=small_open+small_raise;total=n_open+n_raise
    if not 0<half<total:return False
    size_rate=half/total
    common=half*math.log(size_rate)+(total-half)*math.log1p(-size_rate)
    split=(small_open*math.log(.76)+(n_open-small_open)*math.log(.24)+
           small_raise*math.log(.12)+(n_raise-small_raise)*math.log(.88))
    return split-common>=2.5

def choose_action(obs,memory):
    if not memory["blue_gpu_latched"]:
        size_mix_router.record_preflop_raise(obs,memory)
        if detect(memory):
            memory["blue_gpu_latched"]=True
        else:
            return size_mix_router.choose_release_action(obs,memory)
    return policy.select(obs,memory)

app=create_app(globals())
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT","5008")),access_log=False)
