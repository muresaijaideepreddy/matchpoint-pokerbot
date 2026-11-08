"""Recompute every match point from the downloaded duplicate-deal returns."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from judge_engine import CHOP_Z, decide_verdict

def verify(root):
    panel=json.loads((root/"panel.json").read_bytes())
    decision=json.loads((root/"decision.json").read_bytes())
    assert hashlib.sha256((root/"panel.json").read_bytes()).hexdigest()==decision["panel_sha256"]
    with gzip.open(root/"returns.jsonl.gz","rt",encoding="utf-8") as stream:
        vectors=[json.loads(line) for line in stream]
    assert len(panel["fixtures"])==len(decision["fixtures"])==len(vectors)==50
    totals={a:[] for a in ("released","sizemix")}
    table=[]
    for fixture,report,v in zip(panel["fixtures"],decision["fixtures"],vectors,strict=True):
        assert all(report[k]==val for k,val in fixture.items()) and v["fixture"]==fixture["fixture"]
        row=dict(fixture=fixture["fixture"],opponent=fixture["repo"],seed=fixture["seed"],commit=fixture["commit"])
        for arm in totals:
            raw=v[arm];assert len(raw)==1000
            m=mean(raw);r=CHOP_Z*stdev(raw)/math.sqrt(len(raw))
            verdict,p,_=decide_verdict(m-r,m+r)
            assert p["agent1"]==report[arm]["points"] and verdict==report[arm]["verdict"] and abs(report[arm]["bb_per_deal"]-m)<1e-10
            totals[arm].append(p["agent1"])
            row[arm+"_points"]=p["agent1"];row[arm+"_bb_per_deal"]=m
        table.append(row)
    for a,pts in totals.items():
        d=decision["results"][a];assert [sum(pts),pts.count(1),pts.count(.5),pts.count(0)]==[d[k] for k in ("points","wins","chops","losses")]
    promote=sum(totals["sizemix"])>sum(totals["released"]) and decision["results"]["sizemix"]["entrant_faults"]["forfeited"]==0
    assert decision["decision"]==("promote" if promote else "keep")
    with (root/"fixtures.csv").open("w",newline="",encoding="utf8") as f:
        wr=csv.DictWriter(f,fieldnames=list(table[0]));wr.writeheader();wr.writerows(table)
    lines=["# MatchPoint: 50-fixture promotion test","",
           "**Decision: "+("promote SizeMix." if promote else "keep the published version.")+"** The rule frozen before play was to promote only on strictly more official tournament points and no SizeMix forfeits. A tie or loss keeps the previous version.","",
           "Each version played 50 full matches under the official referee in isolated, CPU-only Linux containers. All 37 distinct public opponent snapshots appeared once; 13 of them were randomly selected for a second match without using previous performance. Each fixture had a fresh seed; both versions faced the same opponent image and the same 1,000 deals in both seats. This is 100 matches and 200,000 played hands.","",
           "| Version | Wins | Draws | Losses | Points / 50 | Mean BB per duplicate deal |",
           "|---|---:|---:|---:|---:|---:|"]
    for a,label in [("released","Published v1"),("sizemix","SizeMix")]:
        d=decision["results"][a];lines.append(f"| {label} | {d['wins']} | {d['chops']} | {d['losses']} | {d['points']:g} | {d['bb_per_deal']:+.3f} |")
    ci=decision["paired_fixture_bb_delta_ci95"]
    lines+=["",f"SizeMix earned more points than v1 in {decision['fixtures_ahead']} fixtures, fewer in {decision['fixtures_behind']}, and the same in {decision['fixtures_equal']}. "
            f"Its paired chip margin changed by {decision['bb_per_deal_delta']:+.3f} BB per duplicate deal (approximate 95% interval across the 50 fixture differences: {ci[0]:+.3f} to {ci[1]:+.3f}). The release decision uses points.",
            "","[All 50 scores](fixtures.csv) · [Frozen opponent/seed/image manifest](panel.json) · [Referee outcomes and fault counts](decision.json) · [Per-deal returns](returns.jsonl.gz)","",
            "Both versions contain identical CFR150k and Compact96 trained weights. SizeMix changes only how observed preflop bet sizes select a policy. This benchmark does not establish a tournament ranking."]
    (root/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:11]))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);verify(p.parse_args().root)
