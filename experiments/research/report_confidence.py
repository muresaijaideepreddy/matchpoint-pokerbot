"""Recompute the strict classifier experiment's forty matched outcomes."""
import gzip,json,math,statistics as st
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from judge_engine import decide_verdict,CHOP_Z
def main():
    d=ROOT/"evidence/blue-improvement-20261107/development/confidence"
    overview=json.loads((d/"summary.json").read_bytes());scores={};faults=0;rows={}
    for arm in ("frozen","confidence020"):
        pts=[];vals=[];by={}
        for block in ("first","second"):
            summary=json.loads((d/block/arm/"summary.json").read_bytes())
            assert summary["completed"]==summary["total"]==10 and not summary["errors"]
            with gzip.open(d/block/arm/"returns.jsonl.gz","rt") as f:raw=list(map(json.loads,f))
            original={r["seed"]:r for r in summary["rows"]};assert len(raw)==10
            for r in raw:
                v=r["paired_bb"];assert len(v)==1000
                m=st.mean(v);h=CHOP_Z*st.stdev(v)/math.sqrt(len(v));ver,p,_=decide_verdict(m-h,m+h)
                s=original[r["seed"]];assert ver==s["verdict"] and p["agent1"]==s["points"] and abs(m-s["bb_per_deal"])<1e-10
                faults+=sum(n for role in s["stats"].values() for n in role.values())
                pts.append(p["agent1"]);vals.append(m);by[r["seed"]]=s
        assert len(by)==20
        rows[arm]=by;scores[arm]=dict(wins=pts.count(1),draws=pts.count(.5),losses=pts.count(0),points=sum(pts),bb_per_deal=st.mean(vals))
    assert rows["frozen"].keys()==rows["confidence020"].keys() and faults==0
    lines=["# Stricter opponent classifier","",
           "Only the allowed distance between an observed betting profile and a learned opponent profile changed: 0.070 to 0.020. Both versions faced ten pinned public team bots on two freshly generated match seeds each. Every outcome below was recomputed from all 1,000 duplicate returns in its official referee match. All forty matches were fault-free.","",
           "| Model | Wins / draws / losses | Points / 20 | BB per duplicate deal |","|---|---:|---:|---:|"]
    for name,s in scores.items():lines.append(f"| {name} | {s['wins']} / {s['draws']} / {s['losses']} | {s['points']:g} | {s['bb_per_deal']:+.3f} |")
    lines+=["","**Keep saved Weights 3.** Both policies tied on points. The stricter classifier recovered two wins against dimi1729 and an extra half point against MiniBurger, but gave up 1.5 points against aura-farmer, including a loss. It also reduced measured chip returns against several other familiar opponent styles. The initial two-opponent diagnosis was insufficient to justify a global threshold change.","",
        "| Public opponent | Weights 3 points / 2 | Strict points / 2 | Strict minus original BB/deal |","|---|---:|---:|---:|"]
    for r in overview["aggregate"]:
        lines.append(f"| {r['owner']} | {r['frozen_points']:g} | {r['narrow_points']:g} | {r['chip_change_bb_per_deal']:+.3f} |")
    lines+=["","[All per-match results and exact model hashes](summary.json). The unchanged verified archive remains in weights/weights-3.zip."]
    (d/"verified.json").write_text(json.dumps(dict(scores=scores,faults=faults,decision="keep saved Weights 3"),indent=2)+"\n")
    (d/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8");print(scores)
if __name__=="__main__":main()
