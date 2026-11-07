"""Play a full duplicate match against a source-reviewed local competitor.

Usage: python tools/peer_match.py --candidate models/laptop_neural.npz --peer PATH --seed SEED
The peer runs in a fresh Python interpreter per half. This isolates imports
and half state. Only run code you have audited. The judge's time bank is
disabled here; the peer's own time-based decisions are preserved.
"""

import argparse
from collections import Counter
from contextlib import closing, redirect_stdout
import json
from pathlib import Path
import runpy
import subprocess
import sys


def worker(directory):
    sys.path.insert(0, str(directory))
    channel = sys.stdout
    with redirect_stdout(sys.stderr):
        from game import AgentConfig, HandResult, Observation
        peer = runpy.run_path(str(directory / "agent.py"))
        if "_strategy" in peer and peer["_strategy"] is None:
            raise RuntimeError("The peer failed to import its strategy")
        memory = {}
        counters = Counter()
        monitor = sys.monitoring
        # Count calls without altering the opponent's policy or error handling.
        fallbacks = [peer[k] for k in ("_fallback", "fallback") if k in peer]
        tool = monitor.PROFILER_ID
        monitor.use_tool_id(tool, "peer diagnostics")
        def count_fallback(code, offset):
            counters["fallback_calls"] += 1
        monitor.register_callback(tool, monitor.events.PY_START, count_fallback)
        for fn in fallbacks:
            monitor.set_local_events(tool, fn.__code__, monitor.events.PY_START)
        if peer["AGENT_NAME"] == "BrownBallers":
            wrappers = {peer[name].__code__ for name in ("choose_action", "on_hand_end", "new_half")}
            def count_caught(code, offset, error):
                if code in wrappers:
                    counters["caught_exceptions"] += 1
            monitor.register_callback(tool, monitor.events.EXCEPTION_HANDLED, count_caught)
            monitor.set_events(tool, monitor.events.EXCEPTION_HANDLED)
        def reply(data):
            print(json.dumps(data, separators=(",", ":")), file=channel, flush=True)
        reply(peer["AGENT_NAME"])
        for line in sys.stdin:
            data = json.loads(line)
            for raw in data.get("results", []):
                peer["on_hand_end"](HandResult.from_dict(raw), memory)
            if "config" in data:
                peer["new_half"](AgentConfig.from_dict(data["config"]), memory)
                reply(True)
            elif "observation" in data:
                obs = Observation.from_dict(data["observation"])
                action = peer["choose_action"](obs, memory)
                if action not in obs.legal_actions:
                    raise ValueError(f"Illegal peer action {action!r}")
                counters["decisions"] += 1
                reply(action.name)
            else:
                diag = dict(counters, wrapper_errors=memory.get("errors", counters["caught_exceptions"]))
                if "diagnostics" in peer:
                    diag.update(peer["diagnostics"](memory))
                if "charged" in memory:
                    diag["estimated_charged_seconds"] = memory["charged"]
                if peer["AGENT_NAME"] == "ShortDeckEquity":
                    diag.update(numpy_loaded=peer["np"] is not None, fast_tables=peer["_PAIR_IDS"] is not None,
                                compute_seconds=memory["compute_seconds"], hands_recorded=memory["hands_played"],
                                fallback_calls=counters["fallback_calls"])
                    if not diag["numpy_loaded"] or not diag["fast_tables"]:
                        raise RuntimeError("NumPy strategy or fast equity table unavailable")
                if peer["AGENT_NAME"] == "RangeRunner":
                    import equity_engine
                    import fast_engine
                    diag.update(backend=fast_engine.BACKEND, preflop_keys=len(equity_engine.PREFLOP_KEYS),
                                compute_seconds=memory["cpu"], hands_recorded=memory["hands_played"])
                    if diag["preflop_keys"] != 19998:
                        raise RuntimeError("Incomplete RangeRunner preflop table")
                if peer["AGENT_NAME"] == "BrownBallers":
                    from strategy import equity
                    diag["table_loaded"] = equity._EQUITY is not None
                    diag["hands_recorded"] = memory.get("hands_played")
                    if not diag["table_loaded"]:
                        raise RuntimeError("Peer did not load its preflop equity table")
                reply(diag)
                break


class PeerBot:
    def __init__(self, directory, config):
        from judge_engine import AgentStats
        self.stats = AgentStats()
        self.config = config
        self.results = []
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--worker", str(directory)],
            cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.name = self.receive()
        self.diagnostics = None

    def receive(self):
        raw = self.proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"Peer exited with status {self.proc.poll()}")
        return json.loads(raw)

    def request(self, data):
        self.proc.stdin.write(json.dumps(data, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        return self.receive()

    def start_half(self):
        self.request({"config": self.config.to_dict()})

    def describe(self):
        return self.name

    def act(self, obs):
        from game import Action
        action = Action[self.request({"observation": obs.to_dict(), "results": self.results})]
        self.results.clear()
        return action

    def hand_end(self, result):
        self.results.append(result.to_dict())

    def flush_hand_results(self):
        self.diagnostics = self.request({"results": self.results})
        self.results.clear()

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--peer", type=Path, required=True, help="trusted directory with agent.py")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--save-pairs", action="store_true",
                        help="include individual duplicate-deal bb/hand scores for paired model comparisons")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    from compare_agents import StrategyBot, summary
    from game import AgentConfig, NUM_DEALS
    from judge_engine import combine_halves, play_half
    if not 2 <= args.deals <= NUM_DEALS:
        parser.error(f"--deals must be 2..{NUM_DEALS}")
    config = AgentConfig(num_hands=args.deals)
    halves = []
    diagnostics = []
    for half in (1, 2):
        candidate = StrategyBot(args.candidate, config)
        with closing(PeerBot(args.peer.resolve(), config)) as other:
            halves.append(play_half(candidate, other, args.seed, half, args.deals))
            diagnostics.append(other.diagnostics)
            print(f"Half {half}: {other.name}, {other.diagnostics}", flush=True)
    match = combine_halves(*halves)
    metadata = args.peer / "source.json"
    commit = None
    if metadata.is_file():
        commit = json.loads(metadata.read_text(encoding="utf-8"))["commit"]
    elif (args.peer / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(args.peer), "rev-parse", "HEAD"],
                                         text=True).strip()
    result = dict(candidate=args.candidate, opponent=other.name, directory=str(args.peer),
                  peer_commit=commit, seed=args.seed, verdict=match.verdict, peer_diagnostics=diagnostics,
                  **summary([pair / 2 for pair in match.paired_bb]))
    print(json.dumps(result), flush=True)
    if args.save_pairs:
        result["paired_bb_per_hand"] = [pair / 2 for pair in match.paired_bb]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(Path(sys.argv[2]))
    else:
        main()
