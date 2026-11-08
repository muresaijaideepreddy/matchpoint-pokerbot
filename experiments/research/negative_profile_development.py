"""Test a missing negative style prototype without changing the saved W3 release."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import statistics
import subprocess
import sys

EXPECTED = "sha256:0b5c3793397cc94e05f1a2dc6fc900a18f8d9021e1859320e21fd4d9591b9a99"
OWNERS = ("650joshlee", "junehwang0930-lang", "harishkumar123tamu", "sahithreddy-github",
          "aashrithtutuku02-stack", "dimi1729", "zuhairrr07", "burnerdrew2-prog",
          "utsavdotcom", "tylerpham89", "dushyant-betala", "shaswat-singh23")


def read(path): return json.loads(path.read_bytes())
def write(path, data): path.write_text(json.dumps(data, indent=2) + "\n")
def docker(*args): return subprocess.check_output(["docker", *args], text=True).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(parent):
    sys.path.insert(0, str(parent))
    from trace_public import harness
    folder = parent / "negative-profile-v1" / "training"
    plan = read(folder / "plan.json")
    harness.HERE = folder
    jobs = [("frozen", plan["opponent"]["slug"], seed) for seed in plan["seeds"]]
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(harness.match, jobs))


def main(parent, fit_script):
    sys.path.insert(0, str(parent / "blue-field-v1"))
    from blue_field import freeze, play
    from blue_field_compare import load_arm
    frozen = parent / "combined-dev-v1/largegpu-image.json"
    base = read(frozen)
    assert base["image_id"] == docker("image", "inspect", base["tag"], "--format", "{{.Id}}") == EXPECTED
    folder = parent / "negative-profile-v1"
    folder.mkdir(exist_ok=True)
    train = folder / "training"
    train.mkdir(exist_ok=True)
    all_peers = read(parent / "blue-field-v1/panel.json")["opponents"]
    peers = {r["repo"].split("/")[0].lower(): r for r in all_peers}
    opponent = peers["dimi1729"]
    if not (train / "plan.json").exists():
        salt = secrets.token_hex(12)
        write(train / "plan.json", dict(parent_image=EXPECTED, opponent=opponent,
              seeds=["negative-profile-training-" + salt + s for s in ("-a", "-b")]))
    shutil.copyfile(frozen, train / "frozen-image.json")
    write(train / (opponent["slug"] + "-image.json"), opponent["manifest"])
    subprocess.run([sys.executable, "-u", __file__, str(parent), "--collect"], check=True)
    model = folder / "field_response.npz"
    if not model.exists():
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
                        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp",
                        "-v", str(train) + ":/training:ro", "-v", str(folder) + ":/output",
                        "-v", str(fit_script) + ":/fit_style.py:ro", base["tag"], "python", "/fit_style.py"], check=True)
    manifest = folder / "negative-image.json"
    if not manifest.exists():
        (folder / "Dockerfile").write_text(f'FROM {base["tag"]}\nCOPY --chmod=0444 field_response.npz /app/experiments/field_response/field_response.npz\n')
        tag = "mp-negative-profile:dimi-v1"
        subprocess.run(["docker", "build", "--network", "none", "-q", "-t", tag, "."], cwd=folder, check=True)
        write(manifest, dict(label="negative", tag=tag, image_id=docker("image", "inspect", tag, "--format", "{{.Id}}"),
              frozen_parent=EXPECTED, changed_file="experiments/field_response/field_response.npz",
              model_sha256=sha(model), training=read(folder / "training.json")))
    candidate = read(manifest)
    assert candidate["frozen_parent"] == EXPECTED
    assert candidate["model_sha256"] == sha(model) == read(folder / "training.json")["candidate_sha256"]
    assert candidate["image_id"] == docker("image", "inspect", candidate["tag"], "--format", "{{.Id}}")
    assert candidate["training"] == read(folder / "training.json")
    catalog = folder / "roster.json"
    write(catalog, [peers[owner] for owner in OWNERS])
    folders = [parent / ("negative-profile-eval-" + suffix) for suffix in ("a", "b")]
    for panel in folders:
        panel.mkdir(exist_ok=True)
        if not (panel / "panel.json").exists(): freeze(panel, catalog)
        assert not set(read(train / "plan.json")["seeds"]) & {r["seed"] for r in read(panel / "panel.json")["opponents"]}
    acceptance = dict(stage="development only; never automatically replaces release",
          minimum_total_point_gain=.5, no_target_point_regression=["dimi1729", "utsavdotcom", "dushyant-betala", "aashrithtutuku02-stack"],
          all_fault_free=True, no_losses=True, mean_chip_change_nonnegative=True,
          independent_holdout_required_before_promotion=True)
    if (folder / "acceptance.json").exists(): assert read(folder / "acceptance.json") == acceptance
    else: write(folder / "acceptance.json", acceptance)
    def lane(arm, path, workers):
        for panel in folders: play(panel, arm, path, workers)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(lane, "frozen", frozen, 2), pool.submit(lane, "negative", manifest, 3)]
        for job in jobs: job.result()
    rows = []
    for panel in folders:
        fixtures = read(panel / "panel.json")
        _, old = load_arm(panel, "frozen", fixtures)
        _, new = load_arm(panel, "negative", fixtures)
        for item in fixtures["opponents"]:
            a, b = old[item["slug"]], new[item["slug"]]
            rows.append(dict(repo=item["repo"], seed=item["seed"], block=panel.name, opponent_image=item["image_id"],
                        frozen_points=a["points"], candidate_points=b["points"], frozen_bb=a["bb_per_deal"], candidate_bb=b["bb_per_deal"],
                        frozen_faults=a["stats"], candidate_faults=b["stats"], clean=a["clean"] and b["clean"]))
    aggregate = []
    for owner in OWNERS:
        sample = [r for r in rows if r["repo"].split("/")[0].lower() == owner]
        assert len(sample) == 2
        aggregate.append(dict(owner=owner, frozen_points=sum(r["frozen_points"] for r in sample),
                             candidate_points=sum(r["candidate_points"] for r in sample),
                             chip_change_bb_per_deal=statistics.mean(r["candidate_bb"] - r["frozen_bb"] for r in sample)))
    scores = {}
    for arm, key in (("frozen", "frozen"), ("negative", "candidate")):
        points = [r[key + "_points"] for r in rows]
        scores[arm] = dict(wins=points.count(1), draws=points.count(.5), losses=points.count(0), points=sum(points),
                           bb_per_deal=statistics.mean(r[key + "_bb"] for r in rows))
    protected = {"dimi1729", "utsavdotcom", "dushyant-betala", "aashrithtutuku02-stack"}
    success = (scores["negative"]["points"] >= scores["frozen"]["points"] + .5 and
               scores["negative"]["losses"] == 0 and scores["negative"]["bb_per_deal"] >= scores["frozen"]["bb_per_deal"] and
               all(r["clean"] and not any(n for role in r[k].values() for n in role.values())
                   for r in rows for k in ("frozen_faults", "candidate_faults")) and
               all(r["candidate_points"] >= r["frozen_points"] for r in aggregate if r["owner"] in protected))
    write(folder / "summary.json", dict(frozen=read(frozen), candidate=read(manifest), rows=rows,
          aggregate=aggregate, scores=scores, passes_development_gate=success,
          decision="Needs independent holdout before any promotion" if success else "Keep saved Weights 3; development gate failed"))
    print(json.dumps(dict(scores=scores, aggregate=aggregate, passes_development_gate=success), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--fit-script", type=Path)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect: collect(args.parent.resolve())
    else: main(args.parent.resolve(), args.fit_script.resolve())
