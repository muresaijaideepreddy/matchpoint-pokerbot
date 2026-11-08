# Blueprint risk-heuristic comparison

Each version played twelve fresh matches of 2,000 hands against Blueprint, with the same deals and frozen opponent image for every version. The referee measured actual chip payoffs and the tournament’s confidence-band verdict. All 60 matches finished cleanly.

| Version | Wins / draws / losses | Points / 12 | BB per duplicate deal | Mean payout SD | Mean win cutoff |
| --- | --- | --- | --- | --- | --- |
| Weights 2 | 0 / 10 / 2 | 5 | −0.321 | 10.835 | 0.672 |
| Deep-history 3.15M | 2 / 10 / 0 | 7 | +0.489 | 13.917 | 0.863 |
| Risk penalty 0.025 | 2 / 10 / 0 | 7 | +0.454 | 13.714 | 0.850 |
| Risk penalty 0.05 | 1 / 11 / 0 | 6.5 | +0.495 | 13.253 | 0.821 |
| Risk penalty 0.10 | 2 / 10 / 0 | 7 | +0.524 | 12.666 | 0.785 |

The risk penalty subtracts a coefficient times newly invested chips times one minus estimated equity from an action’s predicted value. Training weights are fixed for these heuristic comparisons.

The 0.10 version lowered mean payout standard deviation by 9.0% and retained the unmodified expert’s seven points. Its observed chip gain was +0.035 BB per duplicate deal, with a paired 95% interval from −0.231 to +0.300. The 0.025 version also retained seven points, and the 0.05 version earned 6.5. The results leave the relative playing strength of the 0.10 variant uncertain.

The 0.10 variant is preserved in `experiments/blue_counter/selected_risk010/`. Per-game scores, model provenance, and the recomputed confidence cutoffs are recorded in [cfr-risk.json](cfr-risk.json).
