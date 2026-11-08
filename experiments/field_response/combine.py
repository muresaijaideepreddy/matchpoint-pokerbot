"""Package the selected field and Blueprint specialists on top of Weights 2."""
import argparse, hashlib, io, json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from challenger_archive import write_archive

SOURCE='''"""MatchPoint: choose a trained policy using only observed betting behavior."""
import os, uvicorn
from bot_server import create_app
from matchpoint import AGENT_NAME, PARTICIPANT, on_hand_end
from size_mix_router import choose_release_action, record_preflop_raise
from experiments.blue_gpu import gate
from experiments.field_response import runtime as field
{rich_import}
new_half=gate.new_half

def choose_action(obs,memory):
    record_preflop_raise(obs,memory)
    if obs.hand_index>=55:
        if memory.get("fr_hand")!=obs.hand_index:
            field.classify(obs,memory)
        if memory["fr_index"]>=0:
            return field.select(obs,memory,choose_release_action(obs,memory))
    if not memory["blue_gpu_latched"] and gate.detect(memory):
        memory["blue_gpu_latched"]=True
    if memory["blue_gpu_latched"]:
        {blue}
    return choose_release_action(obs,memory)

app=create_app(globals())
if __name__=="__main__":uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT",5008)),access_log=False)
'''
def digest(b):return hashlib.sha256(b).hexdigest()
def make(args):
    here=Path(__file__).resolve().parent
    with ZipFile(ROOT/"weights/weights-2.zip") as z:files={n:z.read(n) for n in z.namelist()}
    with ZipFile(args.gpu) as z:
        gate=z.read("agent.py")
        for name in ("experiments/blue_gpu/policy.py","experiments/blue_gpu/response.npz"):
            files[name]=z.read(name)
    files["experiments/blue_gpu/gate.py"]=gate
    rich_import=""; blue="return gate.policy.select(obs,memory)"
    if args.rich_mask:
        rich_import="from experiments.blue_counter import expert"
        # This branch is already guarded by real observations. Retain the table's
        # own confidence test, allowing early activation once the outer gate latches.
        text=(args.rich/"expert.py").read_text()
        text=text.replace("not active(memory,mode)","not (memory.get('blue_gpu_latched') or active(memory,mode))")
        files["experiments/blue_counter/expert.py"]=text.encode()
        files["experiments/blue_counter/response.npz"]=(args.rich/"response.npz").read_bytes()
        if args.qmargin is not None:
            text=text.replace("base = choose_release_action(obs, memory)","from experiments.blue_gpu import policy\n    base = policy.select(obs, memory)")
            text=text.replace("+ margin else base","+ "+str(args.qmargin)+" else base")
            files["experiments/blue_counter/expert.py"]=text.encode()
        for name in ("features.py","features.npz","centers.npz","hand_features.npz"):
            if (args.rich/name).exists():files["experiments/blue_counter/"+name]=(args.rich/name).read_bytes()
        blue=f"if {args.rich_mask} & (1 << int(obs.street)):\n            action=expert.choose_blue(obs,memory)\n            if action is not None:return action\n        "+blue
    files["agent.py"]=SOURCE.format(blue=blue,rich_import=rich_import).encode()
    roster=json.loads((ROOT/"evidence/blue-improvement-20261107/field/blue-field-v1/panel.json").read_bytes())["opponents"]
    slug={r["repo"].split("/")[0].lower():r["slug"] for r in roster}
    rows=json.loads((here/"selection-development.json").read_bytes())["rows"]
    # Guard against fitting to a single lucky or clock-impaired game.
    use={slug[r["owner"].lower()] for r in rows if not r["control"] and r["delta_positive_seeds"]==3 and
           r["clean_pairs"]>=1 and r["delta_points"]>=0 and r["delta_bb_per_deal"]>.35}
    with np.load(here/"field_response.npz",allow_pickle=False) as f:p={k:f[k] for k in f.files}
    p["enabled"]=np.array([v in use for v in p["names"]])
    bio=io.BytesIO();np.savez_compressed(bio,**p)
    files["experiments/field_response/runtime.py"]=(here/"runtime.py").read_bytes()
    files["experiments/field_response/field_response.npz"]=bio.getvalue()
    files["README.md"]=b"# MatchPoint tournament challenger\n\nRun with Python 3.12: install requirements.txt, then run python agent.py. No GPU or internet is needed to play. Development candidates are not tournament submissions. See BuildProvenance.json for exact runtime hashes.\n"
    files["BuildProvenance.json"]=(json.dumps(dict(base_sha256=digest((ROOT/"weights/weights-2.zip").read_bytes()),
        gpu_bundle_sha256=digest(args.gpu.read_bytes()),rich_street_mask=args.rich_mask,minimum_advantage_chips=args.qmargin,field_policies_enabled=sorted(use),
        files_sha256={k:digest(v) for k,v in sorted(files.items()) if k!="BuildProvenance.json"}),indent=2)+"\n").encode()
    write_archive(files,args.out)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=Path,default=ROOT/"experiments/blue_gpu/MatchPoint-BlueGPU-Latched.zip")
    p.add_argument("--rich",type=Path,default=ROOT/"experiments/blue_counter/selected_full")
    p.add_argument("--rich-mask",type=int,default=0);p.add_argument("--qmargin",type=float);p.add_argument("--out",required=True,type=Path);make(p.parse_args())
