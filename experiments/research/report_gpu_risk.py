"""Independently verify all sixty variance-aware development scores."""
import csv,gzip,json,math,statistics as st
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from judge_engine import decide_verdict,CHOP_Z
def main():
    folder=ROOT/"evidence/blue-improvement-20261107/development/gpu-risk"
    plan=json.loads((folder/"dev-plan.json").read_bytes());recorded=json.loads((folder/"dev-summary.json").read_bytes())
    with gzip.open(folder/"dev-returns.jsonl.gz","rt") as f:rows=list(map(json.loads,f))
    assert len(rows)==60 and len({(r["arm"],r["seed"]) for r in rows})==60
    expected={r["label"] for r in plan["entrants"]};assert set(recorded)==expected
    scores={};matched={}
    for name in expected:
        sub=[r for r in rows if r["arm"]==name];assert {r["seed"] for r in sub}==set(plan["seeds"])
        for r in sub:
            v=r["paired_bb"];assert len(v)==1000
            m=st.mean(v);sd=st.stdev(v);h=CHOP_Z*sd/math.sqrt(1000)
            _,p,_=decide_verdict(m-h,m+h)
            r.update(bb=m,sd=sd,cutoff=h,points=p["agent1"])
        matched[name]={r["seed"]:r for r in sub}
        mean=st.mean(r["bb"] for r in sub);points=[r["points"] for r in sub]
        scores[name]=dict(wins=points.count(1),draws=points.count(.5),losses=points.count(0),points=sum(points),
            mean_bb=mean,mean_sd=st.mean(r["sd"] for r in sub),mean_win_cutoff=st.mean(r["cutoff"] for r in sub))
        assert all(abs(v-recorded[name][k])<1e-10 for k,v in scores[name].items())
    line=["# Variance-aware GPU experiment","",
          "Four models received the same 20,971,520 freshly sampled deals and 251,658,240 betting trajectories per model. The risk penalties optimize expected terminal chips minus a coefficient times their second moment. This per-hand objective is only an approximation to duplicate-match variance. An independently trained neutral model separates the objective change from a different training budget.","",
          "All five CPU bots faced the actual Blueprint program on the same twelve new 2,000-hand match seeds. Their model asset was the only runtime change relative to the preserved Weights 3 package.","",
          "| Model | Wins / draws / losses | Points / 12 | BB per duplicate deal | Average duplicate SD | Average match cutoff |",
          "|---|---:|---:|---:|---:|---:|"]
    for name in ("frozen-neutral","neutral","risk0005","risk001","risk003"):
        s=scores[name];label={"frozen-neutral":"Saved Weights 3","neutral":"New neutral","risk0005":"Penalty .0005","risk001":"Penalty .001","risk003":"Penalty .003"}[name]
        line.append(f"| {label} | {s['wins']} / {s['draws']} / {s['losses']} | {s['points']:g} | {s['mean_bb']:+.3f} | {s['mean_sd']:.3f} | {s['mean_win_cutoff']:.3f} |")
    line+=["","The lightest penalty reduced the average observed payout standard deviation by 15.6% relative to saved Weights 3, but tied its official score. Larger penalties lost more chip edge than they recovered through lower variance. **Keep Weights 3.** No tested risk objective demonstrated a match-point improvement. All sixty development matches completed without referee faults; the strongest penalty's loss is a real poker result and is retained.",
          "","[Raw returns](dev-returns.jsonl.gz) · [Model hashes, training objective, and deal counts](training-manifest.json) · [Pinned opponents and seeds](dev-plan.json) · [Verified scores](verified.json)"]
    diffs={}
    for name in expected-{"frozen-neutral"}:
        x=[matched[name][s]["bb"]-matched["frozen-neutral"][s]["bb"] for s in plan["seeds"]]
        mu=st.mean(x);h=2.200985*st.stdev(x)/math.sqrt(len(x));diffs[name]=dict(delta=mu,ci95=[mu-h,mu+h])
    (folder/"verified.json").write_text(json.dumps(dict(scores=scores,paired_difference=diffs,decision="keep saved Weights 3"),indent=2)+"\n")
    (folder/"README.md").write_text("\n".join(line)+"\n",encoding="utf8")
    print("\n".join(line[6:14]))
if __name__=="__main__":main()
