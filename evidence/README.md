# MatchPoint evaluation evidence

MatchPoint won this 2,000-hand match against Quad-J by **1,748 chips**, or **0.437 big blinds per played hand** (95% interval **0.100 to 0.774**). Each bot played both seats of 1,000 deals under the organizer’s HTTP referee. All four bot processes finished with zero errors, skipped calls or forfeits. The result covers this opponent and seed.

| File | What it records |
| --- | --- |
| [Final match](final-live-match.json) | Seed, opponent commit, referee source hashes, both players’ actions and timing by half, per-deal chip returns, and cumulative score. |
| [Release smoke test](final-live-smoke.json) | Archive SHA-256, loaded model hashes, all five legal-action probes, and added image size. |
| [Blind action-value evaluation](blind-action-value-evaluation.json) | Scaled residual96 versus Compact96 on 24,000 fresh simulated hands. The prespecified nominee reduced legal-action RMSE from 27.1452 to 27.0569 chips. This comparison measures prediction error. |
| [Future-board variance](board-variance.json) | Variation in terminal action gains when only future board cards change. Across 2,550 decisions this accounted for 76.4% of Compact96’s squared error on the diagnostic population (95% interval 73.8% to 79.0%). |

The match JSON reports chips from MatchPoint’s perspective. Entry `i` of `paired_net_chips` combines both seats of deal `i`. Summing the entries through a given deal reproduces the cumulative score. Two chips equal one big blind. The referee charged the amount above its one-millisecond grace period on each request. Smoke-test durations are rounded to whole milliseconds.

The match and smoke test identify the exact evaluated archive by its SHA-256. The match JSON records the organizer’s repository name, referee source hashes, and the separately evaluated Quad-J commit.
