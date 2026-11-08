"""Play a pinned entrant against every frozen public opponent on independent duplicate deal blocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import secrets
import statistics
import subprocess
import sys


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    temporary.replace(path)


def freeze(root, catalog):
    panel=root/"panel.json"
    if panel.exists():
        raise ValueError("Panel already frozen")
    seen=set()
    opponents=[]
    salt=secrets.token_hex(12)
    for item in sorted(read(catalog),key=lambda r:r["repo"].lower()):
        manifest=read(root.parent/(item["slug"]+"-image.json"))
        if manifest["commit"]!=item["commit"] or manifest["repo"].lower()!=item["repo"].lower():
            raise ValueError("Opponent manifest mismatch: "+item["repo"])
        image=manifest["image_id"]
        if image in seen:
            raise ValueError("Duplicate image in roster")
        seen.add(image)
        seed="blue-field-"+salt+"-"+str(len(seen))
        opponents.append(dict(repo=item["repo"],slug=item["slug"],commit=item["commit"],
                              image_id=image,manifest=manifest,seed=seed))
    write(panel,dict(salt=salt,played_hands_per_match=2000,opponents=opponents))
    print("Frozen public panel:",len(opponents))


def verify_image(image):
    actual=subprocess.check_output(["docker","image","inspect","--format","{{.Id}}",image],text=True).strip()
    if image!=actual:raise ValueError("Unexpected Docker image "+image)


def play(root, arm, image_path, workers):
    panel=read(root/"panel.json")
    root_manifest=read(image_path)
    verify_image(root_manifest["image_id"])
    out=root/arm
    out.mkdir(exist_ok=True)
    arm_path=out/"entrant.json"
    if arm_path.exists() and read(arm_path)!=root_manifest:raise ValueError("Entrant changed while resuming")
    write(arm_path,root_manifest)
    runbase=root.parent.parent
    judge=read(runbase/"clock-linux/frozen-image.json")
    clock=runbase/"clock-linux-code/linux_clock_match.py"
    error={}
    results={}

    def game(f):
        peer=f["manifest"]
        verify_image(f["image_id"])
        work=out/f["slug"]
        work.mkdir(exist_ok=True)
        prior=sorted(work.glob("attempt-*"))
        for previous in prior:
            previous_file=previous/"clock-linux/gpu-compact96-linux-public.json"
            if previous_file.exists():
                data=read(previous_file)
                if data.get("match"):break
        else:
            dest=work/("attempt-"+str(len(prior)+1).zfill(2))
            stage=dest/"clock-linux";stage.mkdir(parents=True)
            for alias, m in (("compact96",root_manifest),("quadj",peer),("frozen",judge)):
                write(stage/(alias+"-image.json"),m)
            with (dest/"referee.log").open("w") as log:
                subprocess.run([sys.executable,"-u",str(clock),"compact96","--opponent","quadj","--seed",f["seed"]],
                               cwd=dest,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
            data=read(stage/"gpu-compact96-linux-public.json")
        match=data["match"]
        if data["staging"]!={"entrant":root_manifest,"opponent":peer} or data["seed"]!=f["seed"] or match["match_seed"]!=f["seed"]:
            raise ValueError("Wrong matchup on disk")
        samples=match["paired_bb"]
        if len(samples)!=1000 or match["num_deals"]!=1000:
            raise ValueError("Incomplete matchup")
        stats={}
        for role in ("entrant","opponent"):
            procs=[p for p in data["processes"] if p["role"]==role]
            assert len(procs)==2 and all(p["startup_pass"] for p in procs)
            stats[role]={k:sum(p["agent_stats"][k] for p in procs) for k in ("errors","skipped_calls","forfeited","timed_out")}
        return dict(repo=f["repo"],slug=f["slug"],commit=f["commit"],seed=f["seed"],opponent_image=f["image_id"],
                    points=match["points"]["agent1"],verdict=match["verdict"],bb_per_deal=statistics.mean(samples),
                    clean=data["both_completed_cleanly"],stats=stats,paired_bb=samples)

    def save():
        rows=[{k:v for k,v in r.items() if k!="paired_bb"} for _,r in sorted(results.items())]
        points=[r["points"] for r in rows]
        record=dict(arm=arm,entrant=root_manifest,total=len(panel["opponents"]),completed=len(rows),
                    wins=points.count(1),draws=points.count(.5),losses=points.count(0),points=sum(points),
                    clean=sum(r["clean"] for r in rows),errors=error,rows=rows)
        if rows:record["mean_bb_per_deal"]=statistics.mean(r["bb_per_deal"] for r in rows)
        write(out/"summary.json",record)
    save()
    with ThreadPoolExecutor(workers) as pool:
        jobs={pool.submit(game,f):f for f in panel["opponents"]}
        for job in as_completed(jobs):
            f=jobs[job]
            try:results[f["slug"]]=job.result()
            except Exception as exc:error[f["slug"]]=str(exc)
            save()
            print(f"{arm}: {len(results)}/{len(jobs)} complete; errors {len(error)}",flush=True)
    if error:raise RuntimeError(error)
    dest=out/"returns.jsonl.gz"
    tmp=dest.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf8") as stream:
        for slug,r in sorted(results.items()):
            stream.write(json.dumps(dict(slug=slug,seed=r["seed"],paired_bb=r["paired_bb"]),separators=(",",":"))+"\n")
    tmp.replace(dest)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("freeze");a.add_argument("root",type=Path);a.add_argument("catalog",type=Path)
    a=sub.add_parser("play");a.add_argument("root",type=Path);a.add_argument("arm");a.add_argument("manifest",type=Path);a.add_argument("--workers",type=int,default=4)
    args=p.parse_args();root=args.root.resolve()
    if args.action=="freeze":freeze(root,args.catalog)
    else:play(root,args.arm,args.manifest,args.workers)
