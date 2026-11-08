"""Verify the released local bundle against every retained referee report."""
from pathlib import Path
import gzip, hashlib, json, math, statistics as st, sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from judge_engine import decide_verdict,CHOP_Z
from zipfile import ZipFile
def main():
    folder=ROOT/"evidence/blue-improvement-20261107"
    meta=json.loads((folder/"candidate-image.json").read_bytes());archive=ROOT/"weights/weights-3.zip"
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();assert sha==meta["archive"]
    with ZipFile(archive) as z:
        assert not z.testzip();p=json.loads(z.read("BuildProvenance.json"))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in p["files_sha256"].items())
    sources={};records=[]
    def verdict(vec):
        assert len(vec)==1000
        m=st.mean(vec);hw=CHOP_Z*st.stdev(vec)/math.sqrt(len(vec));v,points,_=decide_verdict(m-hw,m+hw)
        return v,points["agent1"]
    for panel in sorted((folder/"field").glob("*/challenger/referee")):
        for source in panel.glob("*.json"):
            d=json.loads(source.read_bytes());m=d["match"]
            assert d["staging"]["entrant"]["image_id"]==meta["image_id"]
            v,p=verdict(m["paired_bb"]);assert v==m["verdict"] and p==m["points"]["agent1"]
            procs=[row for row in d["processes"] if row["role"]=="entrant"]
            faults=sum(row["agent_stats"][k] for row in procs for k in ("errors","skipped_calls","forfeited","timed_out"))
            assert len(procs)==2 and all(p["startup_pass"] for p in procs) and faults==0
            records.append(dict(panel=panel.parents[1].name,seed=d["seed"],points=p))
            sources[str(source.relative_to(folder)).replace("\\","/")]=hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(records)==102
    for name in ("blueprint","rangerunner"):
        d=json.loads((folder/name/"challenger-scores.json").read_bytes())
        assert d["arm"]["image_id"]==meta["image_id"]
        rows={r["seed"]:r for r in d["rows"]}
        with gzip.open(folder/name/"challenger-returns.jsonl.gz","rt") as f:
            for seed,values in map(json.loads,f):
                v,p=verdict(values);assert p==rows[seed]["points"] and rows[seed]["entrant_faults"]==0
                records.append(dict(panel=name,seed=seed,points=p))
    assert len(records)==len({(r["panel"],r["seed"]) for r in records})==182
    assert not any(r["points"]==0 for r in records)
    smoke=json.loads((folder/"windows-smoke.json").read_bytes())
    assert smoke["sha256"]==sha and smoke["http_smoke_passed"]
    out=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha,bytes=archive.stat().st_size,image_id=meta["image_id"],
             verified_matches=len(records),played_hands=2000*len(records),wins=sum(r["points"]==1 for r in records),
             draws=sum(r["points"]==.5 for r in records),losses=0,entrant_faults=0,windows_http_smoke=True,
             note="40 Blueprint and 40 RangeRunner confirmations, 52 broad public/control/archive fixtures, and 50 fresh public proxy fixtures. Public snapshots are not the official leaderboard.",
             field_referee_sha256=sources)
    (folder/"verified-release.json").write_text(json.dumps(out,indent=2)+"\n")
    print({k:v for k,v in out.items() if k!="field_referee_sha256"})
if __name__=="__main__":main()
