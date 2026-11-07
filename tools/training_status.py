"""Serve the live training page on this laptop and read metrics from a GPU over SSH."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time


TOOLS = Path(__file__).resolve().parent
RUNS = TOOLS.parent / "runs"
REFRESH = 5
sys.path.insert(0, str(TOOLS.parent))
from game import (ACTION_TIMEOUT_SECONDS, ERROR_LIMIT, NUM_DEALS, STARTUP_TIMEOUT_SECONDS,
                  TIME_BANK_SECONDS, Action)

REFEREE = dict(startup_seconds=STARTUP_TIMEOUT_SECONDS, request_seconds=ACTION_TIMEOUT_SECONDS,
               bank_seconds=TIME_BANK_SECONDS, hands_per_half=NUM_DEALS, error_limit=ERROR_LIMIT,
               actions=[action.name for action in Action])


def read_result(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def evaluations():
    rows = []
    reports = [
        ("initial", "gpu-compact-field.json", "swirl-5090-quad-01"),
        ("fresh", "gpu-fresh-field.json", "swirl-fresh-quad-01"),
    ]
    for name in ("compact96", "cards128", "cards256", "boosted"):
        reports.append((name, f"swirl_trials/{name}/field.json", f"swirl-{name}-quad-01"))
    for run_id, field_file, public_prefix in reports:
        field = read_result(RUNS / field_file)
        if field:
            seeds = field["seeds"]
            verdicts = [r["candidate_verdict"] for seed in seeds for r in seed["opponents"].values()]
            reference = [r["incumbent_verdict"] for seed in seeds for r in seed["opponents"].values()]

            def points(results):
                return sum({"AGENT1_WIN": 1, "CHOP": 0.5, "AGENT2_WIN": 0}[r] for r in results)

            diff = field["overall"]["field_difference"]
            rows.append(dict(run_id=run_id, label="Seven-opponent field", state="completed",
                             mean=diff["bb_per_hand"], ci95=diff["ci95"],
                             detail=f"Candidate {points(verdicts):g}/{len(verdicts)} points · "
                                    f"MatchPoint {points(reference):g}/{len(reference)}. "
                                    "Paired gain over MatchPoint."))
            head = field["overall"]["head_to_head"]
            rows.append(dict(run_id=run_id, label="Head-to-head vs MatchPoint", state="completed",
                             mean=head["bb_per_hand"], ci95=head["ci95"],
                             detail=f"{len(seeds)} matches of {2 * field['deals_per_seed']:,} hands."))
        else:
            log = (RUNS / field_file).with_suffix(".log")
            if log.exists():
                lines = log.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                done = sum(bool(re.match(r"^\s*vs .+: candidate [+-]", line)) for line in lines)
                failed = any("Traceback (most recent call last)" in line for line in lines)
                exit_file = log.with_suffix(".exit")
                if exit_file.is_file():
                    failed |= exit_file.read_text().strip() not in ("", "0")
                latest = next((s.strip() for s in reversed(lines) if s.strip()), "Starting")
                state = "failed" if failed else "running"
                if not failed and time.time() - log.stat().st_mtime > 600:
                    state = "stale"
                rows.append(dict(run_id=run_id, label="Seven-opponent field",
                                 state=state, mean=None, ci95=None,
                                 detail=f"{done}/14 opponent comparisons reported. {latest[:220]}"))

        left = read_result(RUNS / (public_prefix + "-compact.json"))
        right = read_result(RUNS / (public_prefix + "-matchpoint.json"))
        if not left or not right:
            continue
        if (left["seed"], left["peer_commit"]) != (right["seed"], right["peer_commit"]):
            raise ValueError("Quad-J reports have different seeds or opponent versions")
        pairs = [a - b for a, b in zip(left["paired_bb_per_hand"], right["paired_bb_per_hand"], strict=True)]
        gain = statistics.mean(pairs)
        radius = 1.96 * statistics.stdev(pairs) / math.sqrt(len(pairs))
        rows.append(dict(run_id=run_id, label="Quad-J", state="completed", mean=gain,
                         ci95=[gain-radius, gain+radius],
                         detail=f"Paired gain over MatchPoint on {len(pairs):,} duplicate deals."))
    return rows


def timed_rows():
    rows = []
    for run_id in ("fresh", "compact96", "cards128", "cards256", "boosted"):
        for suffix, label in (("timed", "Windows referee clock"), ("linux-timed", "Linux referee clock")):
            result = read_result(RUNS / f"gpu-{run_id}-{suffix}.json")
            if result:
                rows.append(timing_row(run_id, label, result))
        public = read_result(RUNS / f"gpu-{run_id}-linux-public.json")
        if public:
            row = timing_row(run_id, "Timed Linux match vs Quad-J", public)
            rival = [p["agent_stats"] for p in public.get("processes", [])
                     if p.get("role") == "opponent" and "agent_stats" in p]
            if rival:
                row["detail"] += f" Opponent skipped {sum(s['skipped_calls'] for s in rival)} requests."
            rows.append(row)
    return rows


def public_finalist_rows():
    results = [read_result(path) for path in RUNS.glob("swirl-compact96-public-*.json")]
    new = read_result(RUNS / "swirl-rangerunner-compact96.json")
    old = read_result(RUNS / "swirl-rangerunner-frozen.json")
    if new:
        results.append(new)
    unique = {r["peer_commit"]: r for r in results if r and r.get("verdict")}
    if not unique:
        return []
    outcomes = [r["verdict"] for r in unique.values()]
    wins, chops = outcomes.count("AGENT1_WIN"), outcomes.count("CHOP")
    count = len(outcomes)
    rows = [dict(run_id="compact96", label="Public opponents", state="completed" if count >= 12 else "running",
                 mean=None, ci95=None, detail=f"{count}/12 matches reported. {wins} wins, {chops} chops, "
                                                f"{count-wins-chops} losses. {wins+0.5*chops:g}/{count} points.")]
    if new and old and new["seed"] == old["seed"] and new["peer_commit"] == old["peer_commit"]:
        pairs = [n-o for n, o in zip(new["paired_bb_per_hand"], old["paired_bb_per_hand"], strict=True)]
        gain = statistics.mean(pairs)
        radius = 1.96*statistics.stdev(pairs)/math.sqrt(len(pairs))
        rows.append(dict(run_id="compact96", label="RangeRunner", state="completed", mean=gain,
                         ci95=[gain-radius, gain+radius],
                         detail=f"Paired gain over MatchPoint on {len(pairs):,} duplicate deals. Practice clock disabled."))
    return rows


def confirmation_rows():
    report = read_result(RUNS / "swirl_trials/final-confirmation/summary.json")
    if not report:
        return []
    ids = {"compact96": "compact96", "fresh64": "fresh"}
    return [dict(run_id=ids[row["first"]], label="Independent three-opponent field", state="completed",
                 mean=row["field_difference"]["bb_per_hand"], ci95=row["field_difference"]["ci95"],
                 detail="Paired gain over frozen MatchPoint on two fresh sets of 1,000 duplicate deals.")
            for row in report["comparisons"] if row["second"] == "matchpoint" and row["first"] in ids]


def prediction_comparison():
    trials = RUNS / "swirl_trials"
    rows = []
    for label, name in (("Original validation", "validation"), ("Independent replay", "independent")):
        result = read_result(trials / "rmse-scaled" / (name+".json"))
        if result:
            rows.append(dict(label=label, decisions=result["decision_rows"], reference=result["compact96_rmse"],
                             scaled=result["residual_rmse"], delta=result["difference_chips"], ci95=result["ci95"]))
    blind = read_result(trials / "lockbox/blind_score.json")
    if blind:
        result = blind["all_legal"]["overall"]
        rows.append(dict(label="Final blind test", decisions=result["decision_rows"], reference=result["compact96_rmse"],
                         scaled=result["nominee_rmse"], delta=result["difference_chips"], ci95=result["ci95"]))
    return rows


def final_match():
    report = read_result(RUNS / "final-live-match.json")
    if not report:
        return read_result(RUNS / "final-live-progress.json")
    match = report.get("match") or {}
    if not match.get("verdict"):
        return {"state": "failed", "phase": "The referee did not produce a complete result."}
    cumulative = [0.0]
    for paired in match["paired_bb"]:
        cumulative.append(cumulative[-1] + paired)
    halves = []
    for process in report.get("processes", []):
        stats = process.get("agent_stats", {})
        halves.append(dict(half=process["half"], role=process["role"], startup=process.get("startup_s"),
                           bank=stats.get("time_seconds"), errors=stats.get("errors"),
                           skips=stats.get("skipped_calls"), calls=stats.get("calls")))
    return dict(state="completed", phase=match["verdict"], seed=report["seed"],
                verdict=match["verdict"], hands=2*match["num_deals"], points=match["points"],
                bb_per_hand=match["mean_bb_per_deal"]/2, ci95=[v/2 for v in match["ci"]],
                cumulative_bb=cumulative, halves=halves)


def timing_row(run_id, label, result):
    processes = result.get("processes", [])
    entrant = [p for p in processes if p.get("role") in ("entrant", "candidate")] or processes[::2]
    stats = [p["agent_stats"] for p in entrant if "agent_stats" in p]
    starts = [p["startup_s"] for p in entrant if p.get("startup_s") is not None]
    match = result.get("match") or {}
    verified = (len(stats) == len(starts) == 2 and all(t <= STARTUP_TIMEOUT_SECONDS for t in starts)
                and all(not (s.get("errors") or s.get("skipped_calls") or s.get("timed_out") or s.get("forfeited"))
                        and s.get("time_seconds", TIME_BANK_SECONDS + 1) <= TIME_BANK_SECONDS for s in stats))
    bank = " / ".join(f"{s['time_seconds']:.2f}" for s in stats)
    startups = " / ".join(f"{t:.2f}" for t in starts)
    ci, mean = match.get("ci"), match.get("mean_bb_per_deal")
    return dict(run_id=run_id, label=label, state="completed" if verified else "failed",
                mean=mean/2 if mean is not None else None, ci95=[x/2 for x in ci] if ci else None,
                detail=f"Startup {startups or 'unavailable'} s. Charged {bank or 'unavailable'} s. "
                       f"Errors {sum(s['errors'] for s in stats)}, skipped {sum(s['skipped_calls'] for s in stats)}.")


class Monitor:
    def __init__(self, args):
        self.ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                    "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", f'UserKnownHostsFile="{args.known_hosts.resolve().as_posix()}"',
                    "-i", str(args.identity.expanduser()), args.target,
                    "python3 - " + shlex.quote(args.remote_dir)]
        self.label = args.label
        self.lock = threading.Lock()
        self.last_view = time.monotonic()
        self.last = {}
        self.error = None
        self.stop = threading.Event()

    def poll(self):
        while not self.stop.is_set():
            with self.lock:
                recently_viewed = time.monotonic() - self.last_view < 45
            if recently_viewed:
                try:
                    probe = (TOOLS / "training_probe.py").read_bytes()
                    result = subprocess.run(self.ssh, input=probe, capture_output=True, timeout=10,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode:
                        reason = result.stderr.decode(errors="replace").strip().splitlines()
                        raise RuntimeError(reason[-1][:250] if reason else "SSH connection failed")
                    observed = json.loads(result.stdout)
                    if not isinstance(observed.get("runs"), list) or not observed.get("observed_at"):
                        raise ValueError("Remote probe returned an incomplete observation")
                    with self.lock:
                        self.last, self.error = observed, None
                except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
                    with self.lock:
                        self.error = str(exc)
            self.stop.wait(REFRESH)

    def snapshot(self):
        with self.lock:
            self.last_view = time.monotonic()
            data, error = self.last, self.error
        device = data.get("device") or {}
        try:
            matches = evaluations() + timed_rows() + public_finalist_rows() + confirmation_rows()
        except (ValueError, OSError, KeyError) as exc:
            matches = [dict(run_id=k, label="Match results", state="failed", mean=None, ci95=None,
                            detail=f"Could not read match results: {exc}") for k in ("initial", "fresh")]
        merged = {(row["run_id"], row["label"]): row for row in data.get("evaluations", [])}
        for row in matches:
            key = row["run_id"], row["label"]
            if key not in merged or row["state"] == "completed":
                merged[key] = row
        return dict(updated_at=datetime.now(timezone.utc).isoformat(), refresh_seconds=REFRESH,
                    source=dict(host=self.label, connected=bool(data) and error is None,
                                observed_at=data.get("observed_at"), error=error, device=device.get("name"),
                                gpu_utilization=device.get("utilization_pct"), gpu_memory_mib=device.get("memory_used_mib"),
                                gpu_total_mib=device.get("memory_total_mib")),
                    referee=REFEREE, runs=data.get("runs", []), evaluations=list(merged.values()),
                    prediction_comparison=prediction_comparison(), final_match=final_match())


def serve(args):
    monitor = Monitor(args)
    html = TOOLS / "training_status.html"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.partition("?")[0]
            if path == "/api/status":
                payload = json.dumps(monitor.snapshot(), allow_nan=False).encode()
                content_type = "application/json"
            elif path == "/":
                payload = html.read_bytes()
                content_type = "text/html; charset=utf-8"
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as http:
        thread = threading.Thread(target=monitor.poll, daemon=True)
        thread.start()
        print(f"Poker training status at http://127.0.0.1:{http.server_port}/", flush=True)
        try:
            http.serve_forever()
        finally:
            monitor.stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="SSH user@host")
    parser.add_argument("--remote-dir", required=True, help="training run directory on that host")
    parser.add_argument("--identity", type=Path, default=Path("~/.ssh/id_ed25519"))
    saved_hosts = RUNS / "swirl_known_hosts"
    parser.add_argument("--known-hosts", type=Path, default=saved_hosts if saved_hosts.is_file() else Path.home() / ".ssh" / "known_hosts")
    parser.add_argument("--label", default="Training GPU")
    parser.add_argument("--port", type=int, default=8765)
    serve(parser.parse_args())
