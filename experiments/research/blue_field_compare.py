"""Recompute both public-field standings and pair them by identical deals and opponent images."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import mean, stdev

from blue_field import read, write


def load_arm(root, arm, panel):
    summary=read(root/arm/"summary.json")
    if summary["errors"] or summary["completed"]!=len(panel["opponents"]):
        raise ValueError(arm+" is incomplete")
    scores={row["slug"]:row for row in summary["rows"]}
    with gzip.open(root/arm/"returns.jsonl.gz","rt") as stream:
        returns={row["slug"]:row for row in map(json.loads,stream)}
    for fixture in panel["opponents"]:
        slug=fixture["slug"]; raw=returns[slug]; row=scores[slug]
        if (raw["seed"],row["seed"],row["opponent_image"])!=(fixture["seed"],fixture["seed"],fixture["image_id"]):
            raise ValueError("Mismatched fixture "+slug)
        values=raw["paired_bb"]; n=len(values)
        assert n==1000
        m=mean(values); radius=1.96*stdev(values)/math.sqrt(n)
        points=1 if m-radius>0 else 0 if m+radius<0 else .5
        if points!=row["points"] or abs(row["bb_per_deal"]-m)>1e-10:
            raise ValueError("Mismatch against retained returns for "+slug)
    return summary, scores


def compare(root, new, old):
    panel=read(root/"panel.json")
    summaries={}; scores={}
    for arm in (new,old):
        summaries[arm],scores[arm]=load_arm(root,arm,panel)
    table=[]
    for f in panel["opponents"]:
        slug=f["slug"]; a=scores[new][slug]; b=scores[old][slug]
        table.append(dict(repo=f["repo"],commit=f["commit"],seed=f["seed"],candidate_points=a["points"],
                          reference_points=b["points"],candidate_bb_per_deal=a["bb_per_deal"],
                          reference_bb_per_deal=b["bb_per_deal"],delta_bb_per_deal=a["bb_per_deal"]-b["bb_per_deal"],
                          candidate_clean=a["clean"],reference_clean=b["clean"]))
    def rollup(rows):
        gains=[r["delta_bb_per_deal"] for r in rows]; m=mean(gains)
        interval=None
        if len(gains)>1:
            radius=2.02*stdev(gains)/math.sqrt(len(gains))
            interval=[m-radius,m+radius]
        return dict(fixtures=len(rows),paired_bb_per_deal=m,approx_ci95_by_fixture=interval,
                    candidate_points=sum(r["candidate_points"] for r in rows),
                    reference_points=sum(r["reference_points"] for r in rows))
    team_rows=[r for r in table if r["repo"].lower()!="tamu-datathon-org/heads-up-bundle"
               and not r["repo"].lower().startswith("rw-77/heads-up-dummy-")]
    report=dict(candidate=new,reference=old,all_fixtures=rollup(table),teams_only=rollup(team_rows) if team_rows else None,
        arms={a:{k:summaries[a][k] for k in ("wins","draws","losses","points","clean","mean_bb_per_deal","entrant")} for a in summaries})
    for arm in summaries:
        report["arms"][arm]["faults"]={role:{k:sum(r["stats"][role][k] for r in scores[arm].values())
            for k in ("errors","skipped_calls","forfeited","timed_out")} for role in ("entrant","opponent")}
    write(root/(new+"-vs-"+old+".json"),report)
    with (root/(new+"-vs-"+old+".csv")).open("w",newline="",encoding="utf8") as out:
        writer=csv.DictWriter(out,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path);parser.add_argument("candidate");parser.add_argument("reference",default="weights2",nargs="?")
    arg=parser.parse_args();compare(arg.root,arg.candidate,arg.reference)
