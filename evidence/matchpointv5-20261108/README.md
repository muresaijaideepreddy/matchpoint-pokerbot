# matchpointv5: tournament-format tests (2026-11-08)

`matchpointv5` is the bot at the repository root and in [weights/matchpointv5.zip](../../weights/matchpointv5.zip). It plays exactly like Weights 3 until its own results within a half fall below a floor. It then switches to safer play:

1. **Stop-loss checks.** Hands 150, 300 and 450 compare the average result so far with a floor (−1.2, −0.1 and −0.3 BB/hand). After hand 450, a rolling check looks at the most recent 300 hands.
2. **Range-reader counter.** After a trip, the bot plays balanced CFR self-play adjusted by a GPU-trained counter to range-reading opponents.
3. **Blueprint strategy.** If the counter is also behind at its own checkpoints (hands 100 and 400 after the switch), the rest of the half uses the balanced [Blueprint strategy](../../experiments/blueprint_brain/README.md) by Dushyant-Betala. The 450,000-traversal CFR policy answers if that strategy ever fails.
4. **Blueprint-response mode.** While this mode is latched, looser floors apply and a trip goes straight to the Blueprint strategy.

## Test format

Every match uses the official judge with 1,000 duplicate deals per half and the seats swapped. Points are 1 for a win (95% interval of the paired BB/deal above zero), 0.5 for a draw and 0 for a loss.

Each bot ran in its own container with a dedicated CPU core, 1 CPU, 4 GB, a read-only filesystem and no Internet. Containers were fresh for every half. All 917 matches here finished with zero errors, timeouts or forfeits.

In the raw files, `mp-g5:n1` is matchpointv5 before its name was set; play is byte-identical apart from the name. `mp-g4:n1` is Guard v4, `mp-w4-official:1acc1015` is the Weights 4 draft, and `matchpoint-combined:d944e836cd4d78` is Weights 3.

## Top-5 panel (20 fresh seeds per rival, plus 8 against RevRange 264127f)

| Bot | gate | RevRange | poker-ours | burnerdrew | Blueprint | RevRange 264127f | Points / 108 | Losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weights 3 | 0/16/4 | 0/6/14 | 0/7/13 | 0/17/3 | 3/17/0 | 0/3/5 | 36.0 | 39 |
| Weights 4 draft | 0/17/3 | 14/5/1 | 0/7/13 | 0/18/2 | 1/18/1 | 6/2/0 | 54.5 | 20 |
| Guard v4 | 0/18/2 | 14/6/0 | 0/8/12 | 0/17/3 | 2/18/0 | 6/2/0 | 56.5 | 17 |
| **matchpointv5** | 0/19/1 | 16/4/0 | 0/15/5 | 0/20/0 | 2/18/0 | 6/2/0 | **63.0** | **6** |

Cells are wins/draws/losses. [Summary](top5-summary.txt) · [All matches](top5.jsonl)

## Round robin among the top tier

The other five bots also played one another: 8 fresh seeds per pairing, in the same format. Each row shows a bot's average points per match against each opponent.

| Rank | Bot | Expected points (of 5) |
| --- | --- | --- |
| 1 | Blueprint | 2.89 |
| 2 | poker-ours (BlueprintMix) | 2.81 |
| 3 | **matchpointv5** | **2.80** |
| 4 | burnerdrew | 2.75 |
| 5 | gate | 2.46 |
| 6 | RevRange | 1.29 |

With Guard v4 in the same seat, MatchPoint ranked 5th (2.47); with Weights 3 it ranked last (1.73). [Full table](standings.txt) · [Round-robin matches](round-robin.jsonl)

## Field check

These matches reuse the 78 seeds of the earlier holdout, against the weak field and the strong rivals in the shared-core format. They check that the stop-loss does not give up wins that Weights 3 already had.

| Bot | Points / 78 | Losses | Weak field (48 matches) |
| --- | --- | --- | --- |
| Weights 3 | 53.5 | 11 | 43.5 |
| Weights 4 draft | 62.0 | 5 | 44.5 |
| Guard v4 | 65.0 | 3 | 46.0 |
| **matchpointv5** | **64.0** | **2** | 44.5, no losses |

[Summary](field-check-summary.txt) · [All matches](field-check.jsonl)

## Other strong bots (real format)

| Bot | CFR Blueprint (team variant) | DefeatEquity (latest) | DefeatEquity 0f992c5 | Points |
| --- | --- | --- | --- | --- |
| Weights 3 | 0/8/2 | 4/2/0 | 3/3/0 | 13.5/22 |
| Guard v4 | 0/6/4 | 4/2/0 | 3/3/0 | 12.5/22 |
| **matchpointv5** | 0/9/1 | 4/2/0 | 3/3/0 | **14.0/22** |

[All matches](extended.jsonl)

## Limits

- These are paired tests on the team's arena host, not official leaderboard scores.
- Several rivals are nondeterministic, so single matches are noisy. Compare the totals.
- The round-robin rows for the other five bots come from their own matches, played on different seeds from MatchPoint's.
- [blueprint-bot-vs-problem-opponents.jsonl](blueprint-bot-vs-problem-opponents.jsonl) records the measurement that motivated the Blueprint fallback: the Blueprint bot scored 19.0/27 against the opponents MatchPoint draws or loses to.
