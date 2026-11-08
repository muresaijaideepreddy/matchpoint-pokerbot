"""Independently verify both fresh 40-match confirmations using saved per-deal returns."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, statistics as stats, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from judge_engine import CHOP_Z,decide_verdict
from blue_confirm import score
def interval(x):
    m=stats.mean(x);h=2.02269092*stats.stdev(x)/len(x)**.5;return [m,m-h,m+h]
def main(folder):
    lines=["# MatchPoint: independent Blueprint and RangeRunner confirmation","",
           "Each rival faced both the published Weights 2 and the same frozen challenger on 40 new match seeds. Each match used 1,000 deals played from both seats (2,000 hands), the official referee, and its CPU-only time limits. No result from this panel was used to select the challenger.","",
           "| Rival | Version | Wins | Draws | Losses | Points / 40 | BB per duplicate deal (95% interval) |",
           "|---|---|---:|---:|---:|---:|---:|"]
    records=[];results={};seeds=set();archives=set()
    for rival in ("blueprint","rangerunner"):
        root=folder/rival;plan=json.loads((root/"plan.json").read_bytes());rseeds=set(plan["seeds"])
        assert len(rseeds)==40 and not seeds&rseeds;seeds|=rseeds
        by={};summaries={};cutoffs={}
        for arm,name in (("weights2","Weights 2"),("challenger","Challenger")):
            d=json.loads((root/(arm+"-scores.json")).read_bytes())
            assert d["plan_sha256"]==hashlib.sha256((root/"plan.json").read_bytes()).hexdigest()
            assert d["opponent"]["image_id"]==plan["opponent_image"]
            if arm=="challenger":archives.add(d["arm"]["image_id"])
            else: assert d["arm"]["image_id"]==plan["baseline_image"]
            with gzip.open(root/(arm+"-returns.jsonl.gz"),"rt") as f:ret=dict(map(json.loads,f))
            by[arm]={r["seed"]:r for r in d["rows"]};assert len(d["rows"])==40 and set(ret)==set(by[arm])==rseeds
            cutoffs[arm]=dict(average_duplicate_return_sd=stats.mean(stats.stdev(v) for v in ret.values()),
                             average_match_half_interval=stats.mean(CHOP_Z*stats.stdev(v)/math.sqrt(len(v)) for v in ret.values()))
            for seed,vals in ret.items():
                assert len(vals)==1000
                m=stats.mean(vals); h=CHOP_Z*stats.stdev(vals)/math.sqrt(len(vals)); _,p,_=decide_verdict(m-h,m+h)
                row=by[arm][seed];assert abs(m-row["bb_per_duplicate"])<1e-10 and p["agent1"]==row["points"]
                records.append(dict(rival=rival,arm=arm,**row))
            s=score(d["rows"]);assert s==d["summary"];summaries[arm]=s
            lo,hi=s["ci95"]
            lines.append(f"| {rival.title()} | {name} | {s['wins']} | {s['draws']} | {s['losses']} | {s['points']:g} | {s['bb_per_duplicate']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        dx=[by["challenger"][k]["bb_per_duplicate"]-by["weights2"][k]["bb_per_duplicate"] for k in sorted(rseeds)]
        results[rival]=dict(arms=summaries,paired_chip_change=interval(dx),variance=cutoffs)
    assert len(archives)==1
    lines+=["","## Differences on the same deals",""]
    for rival,r in results.items():
        c=r["arms"]["challenger"];w=r["arms"]["weights2"];mean,lo,hi=r["paired_chip_change"]
        lines.append(f"- **{rival.title()}:** {c['points']-w['points']:+g} match points and {mean:+.3f} BB per duplicate deal (95% seed-level interval {lo:+.3f} to {hi:+.3f}). Challenger errors, skips, timeouts or forfeits: {c['faults']}; opponent: {c['opponent_faults']}. Maximum charged challenger time in one half: {c['max_half_seconds']:.3f} seconds. Observed loss rate: {c['observed_loss_rate']:.1%}; one-sided 95% binomial upper bound: {c['loss_rate_upper_one_sided95']:.1%}.")
    lines+=["","Confidence intervals use 40 independent match seeds as blocks. The per-match win threshold depends on that match’s variance; there is no fixed +0.7 BB/deal cutoff. A low or zero observed loss rate does not guarantee future results.","","[All 160 match scores](confirmation.csv) · [Blueprint raw returns](blueprint/challenger-returns.jsonl.gz) · [RangeRunner raw returns](rangerunner/challenger-returns.jsonl.gz).",
            "","The reported official leader scored 21.25/25. Beating it by three means 24.25/25. This head-to-head confirmation does not measure the official 25-opponent score."]
    v=results["blueprint"]["variance"]
    lines+=["","## Why so many Blueprint draws","",
            f"The mean of the 40 match-level 95% half-widths increased from {v['weights2']['average_match_half_interval']:.3f} BB/deal for Weights 2 to {v['challenger']['average_match_half_interval']:.3f} for the challenger. Average within-match duplicate-return standard deviation increased from {v['weights2']['average_duplicate_return_sd']:.2f} to {v['challenger']['average_duplicate_return_sd']:.2f} BB. Raising chip expectation alone may not improve the chance of an outright match win enough; the new chip edge still falls inside many individual-match intervals."]
    with (folder/"confirmation.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (folder/"confirmation.json").write_text(json.dumps(dict(candidate_image=next(iter(archives)),results=results),indent=2)+"\n")
    (folder/"confirmation.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines[:17]))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);a=p.parse_args();main(a.folder)
