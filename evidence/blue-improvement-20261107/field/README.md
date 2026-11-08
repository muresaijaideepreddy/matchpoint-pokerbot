# Public opponent comparison

Each fixture covers 1,000 duplicate deals played in both seats, or 2,000 hands per version. Weights 2 and the challenger faced identical cards and pinned opponent images under the organizer’s CPU and time limits.

Public GitHub searches and the organizer’s fork network yielded 491 repository leads. The test includes 46 distinct currently public team strategies, four published dummy opponents and one shared starter representing 13 identical forks. Archived ShortStackSage was tested separately because its repository returned HTTP 404 during this search. [Sources and pinned revisions](discovery.json)

| Opponents | Bot | Wins | Draws | Losses | Points | BB per duplicate deal |
|---|---|---:|---:|---:|---:|---:|
| 46 current public teams | Weights 2 | 38 | 7 | 1 | 41.5/46 | +2.402 |
| 46 current public teams | Challenger | 41 | 5 | 0 | 43.5/46 | +2.964 |
| 5 reference bots | Weights 2 | 5 | 0 | 0 | 5/5 | +7.099 |
| 5 reference bots | Challenger | 5 | 0 | 0 | 5/5 | +7.099 |
| Archived ShortStackSage | Weights 2 | 1 | 0 | 0 | 1/1 | +1.389 |
| Archived ShortStackSage | Challenger | 1 | 0 | 0 | 1/1 | +2.133 |

The challenger earned +2 points versus Weights 2 across the 46 current teams. 8 match verdicts changed. In this group 45/46 challenger fixtures and 46/46 Weights 2 fixtures finished with both bots free of referee faults. Across the 45 common fixtures where both versions and their opponents were fault-free, the challenger earned 43 points and Weights 2 earned 40.5.

| Opponent (pinned version) | Weights 2 points | Challenger points |
|---|---:|---:|
| [AllenChirayil/team-mustangs](https://github.com/AllenChirayil/team-mustangs/tree/766a71b2108d3645499c18182981c0c43bb12bc3) | 1 | 0.5 |
| [Utsavdotcom/aura-farmer](https://github.com/Utsavdotcom/aura-farmer/tree/e09a3a05fb3d31feac6125fae2bfd5fe8c621a1f) | 1 | 0.5 |
| [650JoshLee/quad-j-jish](https://github.com/650JoshLee/quad-j-jish/tree/8a21119372e2cf71cda87b932c3311f15068f5aa) | 0.5 | 1 |
| [Dushyant-Betala/heads-up-bundle](https://github.com/Dushyant-Betala/heads-up-bundle/tree/ba64c267aa2e8c442e5e5f824539c400faea7a64) | 0 | 0.5 |
| [harishkumar123tamu/outliers](https://github.com/harishkumar123tamu/outliers/tree/085caedf7af6efdb8d76eb9c262404cf94fc6b3f) | 0.5 | 1 |
| [junehwang0930-lang/PokerBot-Team-quad-j](https://github.com/junehwang0930-lang/PokerBot-Team-quad-j/tree/48c22e690b095684496616cb447072b308d75162) | 0.5 | 1 |
| [shaswat-singh23/datathon-poker](https://github.com/shaswat-singh23/datathon-poker/tree/05c1a8ecadbb1186c94a670796905fff9d73fc8c) | 0.5 | 1 |
| [zuhairrr07/Poker-Bot](https://github.com/zuhairrr07/Poker-Bot/tree/16965ebd30dbc7a79bcbcc6f695b4c65fceaf727) | 0.5 | 1 |

## 25-team public proxy

The selection rule picks the 25 lowest Weights 2 chip margins on the earlier 46-team public screen. Two independent fresh matches per opponent determined the proxy scores. The displayed score out of 25 equals the official points earned across 50 matches divided by two. The comparison target was 24.25/25.

| Bot | Wins | Draws | Losses | Official fixture points | Proxy points / 25 |
|---|---:|---:|---:|---:|---:|
| Weights 2 | 32 | 17 | 1 | 40.5/50 | 20.25/25 |
| Challenger | 40 | 10 | 0 | 45/50 | 22.5/25 |

Across the 46 proxy fixtures with both runs fault-free, the challenger earned 41.5 points and Weights 2 earned 36.5. The public proxy measures these 25 pinned strategies. An official leaderboard score requires organizer results.

## Requested opponents

| Public bot or repository | Broad field W2 → challenger | Proxy W2 → challenger |
|---|---:|---:|
| [RangeRunner](https://github.com/aashrithtutuku02-stack/rangerunner-plo5/tree/f452b09e8ebd6fd6eb9d17035f41e5b626712fcb) | 1 → 1 / 1 | 2 → 2 / 2 |
| [JustAsAwesomeAgent (akshajag)](https://github.com/akshajag/testpokerbot/tree/910ff15c18baf8bf3fa81cdaaf5463bfec135791) | 1 → 1 / 1 | 1.5 → 2 / 2 |
| [Adaptive Equity](https://github.com/andrewv426/timecasters-pokerbot/tree/54118d0c191efe5328bb36c7ff246b771e6d0002) | 1 → 1 / 1 | 2 → 2 / 2 |
| [JustAsAwesomeAgent (ArmaanMediratta)](https://github.com/ArmaanMediratta/heads-up-bundle/tree/183342c61239b9df5db22e6fb2cf426d5e2df758) | 1 → 1 / 1 | 2 → 1.5 / 2 |
| [Blueprint](https://github.com/Dushyant-Betala/heads-up-bundle/tree/ba64c267aa2e8c442e5e5f824539c400faea7a64) | 0 → 0.5 / 1 | 0.5 → 1 / 2 |
| [td26 repository (ShortDeckPLO-v1)](https://github.com/Fargield/td26-poker-bot-v2/tree/4ce02282e8e764766d183130d3c44b8af5d2760a) | 1 → 1 / 1 | 2 → 2 / 2 |
| [PocketEdge v2](https://github.com/tylerpham89/poker-bot-heads-up-/tree/113dba584a2f48727436f0ad8b98d0271044932a) | 1 → 1 / 1 | 2 → 2 / 2 |
| [ShortStackSage (archive)](blue-field-supplement-v1/panel.json) | 1 → 1 / 1 | Outside proxy |

The td26 repository advertises ShortDeckPLO-v1 as its agent name. Dushyant-Betala’s public entry advertises Blueprint. Searches did not establish a separate public challenge entry named BlueprintMix, CFR Blueprint, or DefeatEquity-v5.

[All 52 field fixtures](blue-field-v1/challenger-all-matched.csv) · [All 50 proxy fixtures](blue-proxy25-plan/challenger-scores.csv) · [Frozen proxy roster and rule](blue-proxy25-plan/selection.json) · [Public identity matches](requested-aliases.json)

Each panel folder includes the frozen opponent and deal manifest, the two arm summaries, and every paired deal return.

The same challenger image `sha256:0b5c3793397cc94e05f1a2dc6fc900a18f8d9021e1859320e21fd4d9591b9a99` and Weights 2 image `sha256:b8144b50b34c55a09ed6238f291120d511c139fc0c49539eec9653ebcdbacdfe` were used across all five panels.
