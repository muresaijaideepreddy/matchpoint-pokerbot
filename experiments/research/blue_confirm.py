"""Precommit a 40-seed Blueprint test; stage arms separately and verify raw returns."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import secrets
from statistics import mean, stdev
import sys

N = 40
BLUE = "public-dushyant-betala"
BLUE_ID = "sha256:5caaadc7de305967f2e2c46a52a1ce6dbf3c0202b80efef306ae3f6b88436498"
W2 = "sha256:b8144b50b34c55a09ed6238f291120d511c139fc0c49539eec9653ebcdbacdfe"
def rd(p): return json.loads(p.read_bytes())
def wr(p, d):
    t = p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(d, indent=2)+"\n", encoding="utf8")
    t.replace(p)
def freeze(folder, peername):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"plan.json").open("x") as f:
        nonce = secrets.token_hex(16)
        opp=("rematch-rangerunner-f452-native","sha256:78639d9ee4c3e92fa873b05140c774dedd8ae833c618bdb1e124fee3327565ee") if peername=="rangerunner" else (BLUE,BLUE_ID)
        json.dump(dict(opponent=opp[0],opponent_image=opp[1], baseline_image=W2,
                  rule="Candidate frozen before its first confirmation match; no training on these seeds. Target: fewer than 20% official losses, more wins than losses, positive mean BB/duplicate with 95% seed-block interval above zero and no entrant faults. Test the public field separately.",
                  seeds=[f"bc40-{nonce}-{i:02d}" for i in range(N)]), f, indent=2)
def score(rows):
    vs=[r["bb_per_duplicate"] for r in rows]
    ps=[r["points"] for r in rows]
    mu=mean(vs); h=2.02269092*stdev(vs)/math.sqrt(N)
    # Exact, one-sided binomial 95% upper confidence limit for observed proportion lost.
    k=ps.count(0); low=0.; high=1.
    for _ in range(60):
        p=(high+low)/2
        if sum(math.comb(N,i)*p**i*(1-p)**(N-i) for i in range(k+1)) > .05: low=p
        else: high=p
    return dict(wins=ps.count(1),draws=ps.count(.5),losses=k,points=sum(ps),
                observed_loss_rate=k/N,loss_rate_upper_one_sided95=high,
                bb_per_duplicate=mu,ci95=[mu-h,mu+h],faults=sum(r["entrant_faults"] for r in rows),
                opponent_faults=sum(r["opponent_faults"] for r in rows),
                max_half_seconds=max(r["max_half_seconds"] for r in rows))
def run(folder, arm, manifest, workers):
    parent=folder.parent
    sys.path.insert(0,str(parent))
    import recheck_round2 as referee
    referee.HERE=folder
    plan=rd(folder/"plan.json"); assert len(set(plan["seeds"]))==N
    name=plan.get("opponent",BLUE)
    peer=rd(parent/(name+"-image.json")); assert peer["image_id"]==plan["opponent_image"]
    m=rd(manifest)
    if arm=="weights2": assert m["image_id"]==W2
    for key, data in ((arm,m),(name,peer)):
        p=folder/(key+"-image.json")
        if p.exists(): assert rd(p)==data
        else: wr(p,data)
    def play(seed):
        referee.match((arm,name,seed))
        path=folder/(arm+"-vs-"+name+"--"+seed)/"clock-linux/gpu-compact96-linux-public.json"
        report=rd(path); match=report["match"]; vec=match["paired_bb"]
        assert report["staging"]==dict(entrant=m,opponent=peer) and report["seed"]==match["match_seed"]==seed and len(vec)==1000
        stats={}
        for role in ("entrant","opponent"):
            ss=[p for p in report["processes"] if p["role"]==role]; assert len(ss)==2 and all(p["startup_pass"] for p in ss)
            stats[role+"_faults"]=sum(p["agent_stats"][key] for p in ss for key in ("errors","timed_out","skipped_calls","forfeited"))
            if role=="entrant":stats["max_half_seconds"]=max(p["agent_stats"]["time_seconds"] for p in ss)
        avg=mean(vec); margin=1.96*stdev(vec)/math.sqrt(len(vec))
        pts=1. if avg-margin>0 else 0. if avg+margin<0 else .5
        assert pts==match["points"]["agent1"] and abs(avg-match["mean_bb_per_deal"])<1e-8
        return dict(seed=seed,points=pts,bb_per_duplicate=avg,**stats,paired_bb=vec)
    rows=[]
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(play,s) for s in plan["seeds"]]):
            rows.append(fut.result())
    rows.sort(key=lambda r:r["seed"])
    with gzip.open(folder/(arm+"-returns.jsonl.gz"),"wt") as out:
        for r in rows: out.write(json.dumps([r["seed"],r.pop("paired_bb")],separators=(",",":"))+"\n")
    wr(folder/(arm+"-scores.json"),dict(arm=m,opponent=peer,plan_sha256=hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest(),summary=score(rows),rows=rows))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("folder",type=Path); p.add_argument("action",choices=("freeze","run"))
    p.add_argument("--arm",choices=("weights2","challenger"));p.add_argument("--manifest",type=Path);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.action=="freeze":freeze(a.folder.resolve(),"rangerunner" if "range" in a.folder.name else "blueprint")
    else:run(a.folder.resolve(),a.arm,a.manifest,a.workers)
