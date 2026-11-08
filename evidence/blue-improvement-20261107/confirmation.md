# MatchPoint: independent Blueprint and RangeRunner confirmation

Each rival faced both the published Weights 2 and the same frozen challenger on 40 new match seeds. Each match used 1,000 deals played from both seats (2,000 hands), the official referee, and its CPU-only time limits. No result from this panel was used to select the challenger.

| Rival | Version | Wins | Draws | Losses | Points / 40 | BB per duplicate deal (95% interval) |
|---|---|---:|---:|---:|---:|---:|
| Blueprint | Weights 2 | 0 | 28 | 12 | 14 | -0.436 [-0.543, -0.329] |
| Blueprint | Challenger | 5 | 35 | 0 | 22.5 | +0.509 [+0.355, +0.664] |
| Rangerunner | Weights 2 | 37 | 3 | 0 | 38.5 | +1.678 [+1.539, +1.817] |
| Rangerunner | Challenger | 38 | 2 | 0 | 39 | +3.259 [+2.953, +3.565] |

## Differences on the same deals

- **Blueprint:** +8.5 match points and +0.945 BB per duplicate deal (95% seed-level interval +0.778 to +1.113). Challenger errors, skips, timeouts or forfeits: 0; opponent: 0. Maximum charged challenger time in one half: 0.533 seconds. Observed loss rate: 0.0%; one-sided 95% binomial upper bound: 7.2%.
- **Rangerunner:** +0.5 match points and +1.581 BB per duplicate deal (95% seed-level interval +1.268 to +1.895). Challenger errors, skips, timeouts or forfeits: 0; opponent: 0. Maximum charged challenger time in one half: 0.380 seconds. Observed loss rate: 0.0%; one-sided 95% binomial upper bound: 7.2%.

Confidence intervals use 40 independent match seeds as blocks. The per-match win threshold depends on that match’s variance; there is no fixed +0.7 BB/deal cutoff. A low or zero observed loss rate does not guarantee future results.

[All 160 match scores](confirmation.csv) · [Blueprint raw returns](blueprint/challenger-returns.jsonl.gz) · [RangeRunner raw returns](rangerunner/challenger-returns.jsonl.gz).

The reported official leader scored 21.25/25. Beating it by three means 24.25/25. This head-to-head confirmation does not measure the official 25-opponent score.

## Why so many Blueprint draws

The mean of the 40 match-level 95% half-widths increased from 0.663 BB/deal for Weights 2 to 1.025 for the challenger. Average within-match duplicate-return standard deviation increased from 10.70 to 16.53 BB. Raising chip expectation alone may not improve the chance of an outright match win enough; the new chip edge still falls inside many individual-match intervals.
