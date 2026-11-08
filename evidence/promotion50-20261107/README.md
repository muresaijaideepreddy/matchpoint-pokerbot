# MatchPoint: 50-fixture promotion test

**Decision: promote SizeMix.** The rule frozen before play was to promote only on strictly more official tournament points and no SizeMix forfeits. A tie or loss keeps the previous version.

Each version played 50 full matches under the official referee in isolated, CPU-only Linux containers. All 37 distinct public opponent snapshots appeared once; 13 of them were randomly selected for a second match without using previous performance. Each fixture had a fresh seed; both versions faced the same opponent image and the same 1,000 deals in both seats. This is 100 matches and 200,000 played hands.

| Version | Wins | Draws | Losses | Points / 50 | Mean BB per duplicate deal |
|---|---:|---:|---:|---:|---:|
| Published v1 | 42 | 8 | 0 | 46 | +2.939 |
| SizeMix | 43 | 7 | 0 | 46.5 | +2.954 |

SizeMix earned more points than v1 in 1 fixtures, fewer in 0, and the same in 49. Its paired chip margin changed by +0.015 BB per duplicate deal (approximate 95% interval across the 50 fixture differences: -0.066 to +0.097). The release decision uses points.

[All 50 scores](fixtures.csv) · [Frozen opponent/seed/image manifest](panel.json) · [Referee outcomes and fault counts](decision.json) · [Per-deal returns](returns.jsonl.gz)

Both versions contain identical CFR150k and Compact96 trained weights. SizeMix changes only how observed preflop bet sizes select a policy. This benchmark does not establish a tournament ranking.
