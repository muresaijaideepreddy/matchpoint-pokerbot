"""Play duplicate matches in-process with the official referee and strategy callbacks.

    python tools/benchmark.py                         # agent vs all four baselines
    python tools/benchmark.py --round-robin           # every pair in the practice field
    python tools/benchmark.py --deals 100 --seed 7 agent.py baseline-bots/baseline_random.py

Install each bot's requirements first. Only load trusted local bots. Timing
excludes HTTP transport and web server overhead. Use judge_engine.py with
separate servers for network timing and hard deadlines on unresponsive bots.
"""

import argparse
import runpy
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import warmup
from game import NUM_DEALS, AgentConfig, HandResult, Observation
from judge_engine import HttpAgentHandle, combine_halves, format_report, play_half


class LocalBot(HttpAgentHandle):
    """Swap the HTTP transport for local callbacks; keep the judge's checks."""

    def __init__(self, path, config):
        super().__init__("http://local", config)
        self.module = runpy.run_path(str(path))
        self.memory = {}

    def describe(self):
        return self.module["AGENT_NAME"]

    def _request(self, method, path, body, timeout):
        start = time.perf_counter()
        errors = []
        status, reply = 200, {"hand_end_errors": errors}
        try:
            for raw in body.get("hand_results", body.get("results", [])):
                try:
                    self.module["on_hand_end"](HandResult.from_dict(raw), self.memory)
                except Exception as exc:
                    errors.append({"error": str(exc), "traceback": traceback.format_exc()})
            if path == "/new-half":
                self.memory.clear()
                self.module["new_half"](AgentConfig.from_dict(body["config"]), self.memory)
            elif path == "/act":
                obs = Observation.from_dict(body["observation"])
                reply["action"] = self.module["choose_action"](obs, self.memory)
            elif path != "/hand-end":
                raise ValueError(f"Unknown route {path!r}")
        except Exception as exc:
            status = 500
            reply.update(error=str(exc), traceback=traceback.format_exc())
        if time.perf_counter() - start > timeout:
            raise TimeoutError
        return status, reply


def match(first, second, seed, deals):
    config = AgentConfig(num_hands=deals)
    halves = []
    for half in (1, 2):
        # Reload each entry point so its global variables also start fresh.
        left = LocalBot(first, config)
        right = LocalBot(second, config)
        halves.append(play_half(left, right, seed, half, deals))
    return combine_halves(*halves)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bots", nargs="*", type=Path,
                        help="Python bot files (default: agent.py and bundled baselines)")
    parser.add_argument("--deals", type=int, default=NUM_DEALS, help="duplicate deals per match")
    parser.add_argument("--seed", default="1", help="seed shared by both halves")
    parser.add_argument("--round-robin", action="store_true", help="play every pair of bots")
    args = parser.parse_args()
    bots = args.bots or [ROOT / "agent.py", *sorted((ROOT / "baseline-bots").glob("baseline_*.py"))]
    if len(bots) < 2 or args.deals < 1:
        parser.error("provide at least two bots and a positive number of deals")
    if len({path.resolve() for path in bots}) != len(bots):
        parser.error("each bot must have a distinct path")

    warmup()
    print("Local strategy benchmark (timing excludes HTTP and web server overhead).", flush=True)
    points = dict.fromkeys(bots, 0.0)
    pairs = combinations(bots, 2) if args.round_robin else ((bots[0], bot) for bot in bots[1:])
    for first, second in pairs:
        result = match(first, second, args.seed, args.deals)
        print(format_report(result), flush=True)
        points[first] += result.points["agent1"]
        points[second] += result.points["agent2"]

    if args.round_robin:
        print("\nRound-robin standings")
        for bot in sorted(bots, key=points.get, reverse=True):
            print(f"{points[bot]:4g} points   {bot.name}")


if __name__ == "__main__":
    main()
