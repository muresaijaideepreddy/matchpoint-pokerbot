"""Score a local round robin in tournament points. Each pairing plays duplicate hands.

    python tools/league.py agent.py models/laptop_mc.json models/laptop_neural.npz polarized --seeds season1 season2
"""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

from compare_agents import NUM_DEALS, scores, summary, warmup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrants", nargs="+", help="bot files, policy checkpoints, or supported practice profiles")
    parser.add_argument("--seeds", nargs="+", required=True, help="independent round-robin repeats")
    parser.add_argument("--deals", type=int, default=NUM_DEALS, help="duplicate deals in each match")
    parser.add_argument("--json", type=Path, help="write individual results after every match")
    args = parser.parse_args()
    entrants = args.entrants
    if len(entrants) < 2 or len(set(entrants)) != len(entrants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least two distinct entrants and distinct seeds")
    if not 2 <= args.deals <= NUM_DEALS:
        parser.error(f"--deals must be between 2 and {NUM_DEALS}")
    warmup()
    points = {entrant: 0.0 for entrant in entrants}
    standings = {entrant: defaultdict(int) for entrant in entrants}
    report = dict(entrants=entrants, seeds=args.seeds, deals_per_match=args.deals, matches=[],
                  standings=standings, points=points)
    for seed in args.seeds:
        for index, (first, second) in enumerate(combinations(entrants, 2)):
            match_seed = f"{seed}/pair-{index}"
            values, verdict = scores(first, second, match_seed, args.deals)
            left, right = {"AGENT1_WIN": (1, 0), "AGENT2_WIN": (0, 1), "CHOP": (0.5, 0.5)}[verdict]
            for entrant, result in ((first, left), (second, right)):
                points[entrant] += result
                standings[entrant][{0: "losses", 0.5: "chops", 1: "wins"}[result]] += 1
            result = summary(values)
            report["matches"].append(dict(seed=match_seed, first=first, second=second, verdict=verdict, **result))
            print(f"{match_seed} {Path(first).stem} vs {Path(second).stem}: {verdict}, "
                  f"{result['bb_per_hand']:+.3f} bb/hand", flush=True)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                pending = args.json.with_suffix(args.json.suffix + ".tmp")
                pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                pending.replace(args.json)
    print("\nPoints   W / C / L   Entrant")
    for entrant in sorted(points, key=points.get, reverse=True):
        row = standings[entrant]
        print(f"{points[entrant]:5g}   {row['wins']} / {row['chops']} / {row['losses']}   {entrant}")


if __name__ == "__main__":
    main()
