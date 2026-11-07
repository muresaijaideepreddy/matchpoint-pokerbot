"""Run the three preregistered GPU checkpoints against frozen MatchPoint on the training host."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from statistics import fmean
import json
import os
import subprocess
import sys
import traceback


TRIALS = ("compact96", "cards128", "cards256")
SEEDS = ("swirl-sweep-blind-01", "swirl-sweep-blind-02")
BASELINES = ("always_half", "always_pot", "calling_station", "random")
FINAL_SEEDS = ("swirl-final-f01", "swirl-final-f02")
FINAL_POLICIES = ("compact96", "fresh64", "matchpoint")
FINAL_OPPONENTS = ("value", "tight", "polarized")
WRAPPER = '''"""Private MatchPoint instance using the adjacent trial checkpoint."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

directory = Path(__file__).resolve().parent
project = directory.parent / "experiment"
sys.path.insert(0, str(project))
spec = spec_from_file_location("_trial_matchpoint", project / "matchpoint.py")
candidate = module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate.VALUE = candidate.NeuralPolicy.load(
    directory / "policy.npz", response_rates=candidate.rivercount._response_rates)
PARTICIPANT = candidate.PARTICIPANT
AGENT_NAME = "MatchPoint-" + directory.name
new_half = candidate.new_half
choose_action = candidate.choose_action
on_hand_end = candidate.on_hand_end
app = candidate.app
'''


def worker_env(root):
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if (root / "field-deps").is_dir():
        env["PYTHONPATH"] = str(root / "field-deps") + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("RIVERCOUNT_MODEL", "MATCHPOINT_CFR", "MATCHPOINT_CANDIDATE_WEIGHTS", "CFR_MODEL", "CFR_MIN_VISITS"):
        env.pop(name, None)
    return env


def run_trial(root, trial):
    directory = root / trial
    project = root / "experiment"
    output = directory / "field.json"
    exit_file = directory / "field.exit"
    if exit_file.exists() and exit_file.read_text().strip() == "0" and output.is_file():
        return 0
    if (directory / "exit.code").read_text().strip() != "0" or not (directory / "policy.npz").is_file():
        raise ValueError(f"Missing successful fit for {trial}")
    wrapper = directory / "field_agent.py"
    wrapper.write_text(WRAPPER, encoding="utf-8")
    opponents = [str(project / "baseline-bots" / f"baseline_{name}.py") for name in BASELINES]
    opponents.extend(("value", "tight", "polarized"))
    partial = directory / "field.partial.json"
    command = ["nice", "-n", "10", sys.executable, "-u", str(project / "tools" / "compare_agents.py"),
               "--candidate", str(wrapper), "--incumbent", str(project / "matchpoint.py"),
               "--seeds", *SEEDS, "--deals", "1000", "--opponents", *opponents, "--json", str(partial)]
    exit_file.unlink(missing_ok=True)
    code = 1
    with (directory / "field.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen(command, cwd=project, env=worker_env(root), stdout=log, stderr=subprocess.STDOUT)
            (directory / "field.pid").write_text(str(process.pid), encoding="ascii")
            code = process.wait()
            if code == 0:
                code = 1
                record = json.loads(partial.read_text(encoding="utf-8"))
                if (record["deals_per_seed"] != 1000 or tuple(row["seed"] for row in record["seeds"]) != SEEDS
                        or any(set(row["opponents"]) != set(opponents) for row in record["seeds"])):
                    raise ValueError("Incomplete field report")
                partial.replace(output)
                code = 0
        except Exception:
            traceback.print_exc(file=log)
        finally:
            tmp = directory / "field.exit.tmp"
            tmp.write_text(str(code), encoding="ascii")
            tmp.replace(exit_file)
    return code


def confirm_worker(root, name):
    project = root / "experiment"
    sys.path.insert(0, str(project))
    from tools.compare_agents import scores, warmup
    warmup()
    directory = root / "final-confirmation"
    paths = {"compact96": root / "compact96" / "field_agent.py", "fresh64": root / "fresh" / "field_agent.py",
             "matchpoint": project / "matchpoint.py"}
    direct = {"compact96": ("compact96", "fresh64"), "fresh64": ("fresh64", "matchpoint"),
              "matchpoint": ("compact96", "matchpoint")}[name]
    requested = [(name, opponent) for opponent in FINAL_OPPONENTS] + [direct]
    results = []
    for seed in FINAL_SEEDS:
        for first, second in requested:
            values, verdict = scores(str(paths[first]), str(paths.get(second, second)), seed, 1000)
            assert len(values) == 1000
            results.append(dict(first=first, second=second, seed=seed, pairs=values, verdict=verdict))
            print(f"{seed} {first} vs {second} {fmean(values):+.4f} BB/hand {verdict}", flush=True)
    output = directory / (name + ".json")
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
    tmp.replace(output)


def confirmation(root):
    project = root / "experiment"
    directory = root / "final-confirmation"
    directory.mkdir(exist_ok=True)
    output = directory / "summary.json"
    if output.exists():
        return 0
    hashes = {}
    for path, expected in (
        (root / "compact96/policy.npz", "7e028b62affa45a3a89b309d65ee019725e3e34ff683e6a0e1751bf0b6d08dc3"),
        (root / "fresh/gpu_compact_fresh.npz", "43aa9a77f464f8cd10a48da43a1560361c9ecb637793add0dd9540aa8aa5c05e"),
        (project / "matchpoint.py", "37274a281a842851bcc761aa4b7c0c2b36edb3c0ec60d6ff08ca6fcde1af202d"),
    ):
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Unrecognized frozen artifact {path.name}")
        hashes[str(path.relative_to(root))] = digest
    (root / "fresh/field_agent.py").write_text(
        WRAPPER.replace('directory / "policy.npz"', 'directory / "gpu_compact_fresh.npz"'), encoding="utf-8")
    if (root / "compact96/field_agent.py").read_text(encoding="utf-8") != WRAPPER:
        raise ValueError("compact96 wrapper differs from the field wrapper.")
    def worker(name):
        command = ["nice", "-n", "10", sys.executable, "-u", str(Path(__file__).resolve()), str(root),
                   "--confirm-worker", name]
        with (directory / (name + ".log")).open("w", encoding="utf-8") as log:
            return subprocess.run(command, cwd=project, env=worker_env(root), stdout=log, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=3) as pool:
        if any(list(pool.map(worker, FINAL_POLICIES))):
            raise RuntimeError("Confirmation worker failed (see per-policy logs).")
    sys.path.insert(0, str(project))
    from tools.compare_agents import summary
    matches = {}
    for name in FINAL_POLICIES:
        for r in json.loads((directory / (name + ".json")).read_text(encoding="utf-8")):
            matches[r["first"], r["second"], r["seed"]] = r
    points = {"AGENT1_WIN": 1, "CHOP": .5, "AGENT2_WIN": 0}
    field_points = {name: sum(points[matches[name, opp, seed]["verdict"]] for seed in FINAL_SEEDS
                              for opp in FINAL_OPPONENTS) for name in FINAL_POLICIES}
    rows = []
    for first, second in (("compact96", "matchpoint"), ("fresh64", "matchpoint"), ("compact96", "fresh64")):
        direct, paired = [], []
        verdicts = []
        for seed in FINAL_SEEDS:
            r = matches[first, second, seed]
            direct.extend(r["pairs"])
            verdicts.append(r["verdict"])
            per_opponent = []
            for opp in FINAL_OPPONENTS:
                a, b = (matches[k, opp, seed]["pairs"] for k in (first, second))
                per_opponent.append([x-y for x,y in zip(a, b, strict=True)])
            paired.extend(fmean(v) for v in zip(*per_opponent, strict=True))
        rows.append(dict(first=first, second=second, direct=summary(direct), direct_verdicts=verdicts,
                         field_difference=summary(paired)))
    result = dict(seeds=FINAL_SEEDS, opponents=FINAL_OPPONENTS, duplicate_deals_per_match=1000,
                  field_points=field_points, units="BB/hand", comparisons=rows, sha256=hashes)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--confirm-worker", choices=FINAL_POLICIES)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    os.umask(0o077)
    if args.confirm_worker:
        confirm_worker(root, args.confirm_worker)
        return 0
    manager = "confirmation-manager" if args.confirm else "field-manager"
    if args.detach:
        log = root / (manager + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), str(root)]
        if args.confirm:
            command.append("--confirm")
        with log.open("a") as stream:
            child = subprocess.Popen(command,
                                     cwd=root, start_new_session=True, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        print(f"Started {manager} PID {child.pid}")
        return 0
    import fcntl
    with (root / (manager + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        (root / (manager + ".pid")).write_text(str(os.getpid()), encoding="ascii")
        if args.confirm:
            return confirmation(root)
        with ThreadPoolExecutor(max_workers=3) as pool:
            return int(any(list(pool.map(lambda trial: run_trial(root, trial), TRIALS))))


if __name__ == "__main__":
    raise SystemExit(main())
