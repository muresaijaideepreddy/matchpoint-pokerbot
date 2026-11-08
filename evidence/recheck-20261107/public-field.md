# MatchPoint public-field test — 7 November 2026

37 distinct, currently public opponent strategies, two fresh seeds, 2,000 played hands per match. One big blind per duplicate deal is half a big blind per played hand. All entrants faced the same opponent images and cards.

| Entrant | Wins / draws / losses | Match points | Mean BB/deal (95% interval) |
|---|---:|---:|---:|
| Published Compact96 + CFR150k | 63 / 10 / 1 | 68/74 | +2.741 [+2.509, +2.973] |
| Balanced router + CFR450k | 60 / 13 / 1 | 66.5/74 | +2.614 [+2.391, +2.837] |
| SizeMix (release + preflop sizing rule) | 64 / 9 / 1 | 68.5/74 | +2.711 [+2.482, +2.940] |

Paired change in mean BB/deal for the broadly routed CFR450k challenger: **-0.127** (95% interval -0.232 to -0.022). We average the opponents for each deal before estimating uncertainty so reusing cards does not inflate the sample size.

Against the published release directly, the broadly routed CFR450k challenger averaged **+0.494 BB/deal** (95% interval +0.042 to +0.946) on 4,000 different duplicate deals.

SizeMix’s paired field change versus the release was **-0.030 BB/deal** (95% interval -0.107 to +0.046). Both use the same value and self-play weights. SizeMix changes its choice of expert only when enough observed preflop actions show frequent half-pot opens and pot-sized reraises.

**Tournament candidate: SizeMix.** It earned the most points in this field and reduced a confirmed loss against Blueprint. The half-point lead over the published release is small, and the release had the higher observed average chip margin. The larger CFR450k model is retained as a challenger.

The fresh 450k training run added 300,000 self-play traversals and 27,233,518 decision nodes. In a 4,000-hand diagnostic, 9,394 of 10,185 decisions against Blueprint went to the value expert. With genuinely different half-pot and pot-sized raises available, Blueprint chose half-pot on 1,211/1,595 opens (76%) but only 69/599 reraises (12%). The narrow rule switches to the release’s self-play expert on this public betting pattern. [Training evidence](extra-training.json) · [Routing diagnosis](routing-diagnosis.json)

## Independent follow-up on the closest opponents

These four fresh-seed matches per opponent were reserved for the final SizeMix test. They are excluded from the field table.

| Opponent | Release BB/deal | SizeMix BB/deal | Same-deal change (95% interval) |
|---|---:|---:|---:|
| Dushyant-Betala/heads-up-bundle | -1.291 | -0.571 | +0.720 [+0.190, +1.249] |
| burnerdrew2-prog/tamu-heads-up-bot | -0.027 | -0.292 | -0.264 [-0.644, +0.115] |

## Per-opponent scores

Two tournament points possible for each version in this table. Opponent links are pinned to the exact tested revision. Rows are sorted by the challenger’s chip margin.

| Public opponent | Release points | 450k points | Release BB/deal | 450k BB/deal | SizeMix points | SizeMix BB/deal |
|---|---:|---:|---:|---:|---:|---:|
| [burnerdrew2-prog/tamu-heads-up-bot](https://github.com/burnerdrew2-prog/tamu-heads-up-bot/tree/c0114922f94845414288074843e78f6fec9749c9) | 1 | 1 | +0.101 | -0.669 | 1 | -0.317 |
| [Dushyant-Betala/heads-up-bundle](https://github.com/Dushyant-Betala/heads-up-bundle/tree/ba64c267aa2e8c442e5e5f824539c400faea7a64) | 0.5 | 0.5 | -0.782 | -0.651 | 0.5 | -0.465 |
| [650JoshLee/quad-j-jish](https://github.com/650JoshLee/quad-j-jish/tree/8a21119372e2cf71cda87b932c3311f15068f5aa) | 1 | 1 | +0.629 | +0.554 | 1 | +0.629 |
| [junehwang0930-lang/PokerBot-Team-quad-j](https://github.com/junehwang0930-lang/PokerBot-Team-quad-j/tree/48c22e690b095684496616cb447072b308d75162) | 1 | 1 | +0.342 | +0.715 | 1.5 | +1.127 |
| [AllenChirayil/team-mustangs](https://github.com/AllenChirayil/team-mustangs/tree/766a71b2108d3645499c18182981c0c43bb12bc3) | 2 | 1.5 | +2.037 | +0.823 | 2 | +2.037 |
| [SahithReddy-GitHub/poker-bot-tamu-datathon](https://github.com/SahithReddy-GitHub/poker-bot-tamu-datathon/tree/cf49c1ab68f0b1113218b1d955957cee37712d35) | 1.5 | 1 | +0.958 | +0.911 | 1.5 | +1.079 |
| [ArmaanMediratta/heads-up-bundle](https://github.com/ArmaanMediratta/heads-up-bundle/tree/32d0e2832f035b9aa7351b01713b0d46a4e8ae5d) | 2 | 2 | +1.098 | +1.143 | 1.5 | +1.100 |
| [akshajag/testpokerbot](https://github.com/akshajag/testpokerbot/tree/910ff15c18baf8bf3fa81cdaaf5463bfec135791) | 2 | 1.5 | +1.200 | +1.171 | 2 | +1.437 |
| [harishkumar123tamu/outliers](https://github.com/harishkumar123tamu/outliers/tree/085caedf7af6efdb8d76eb9c262404cf94fc6b3f) | 2 | 1.5 | +1.275 | +1.184 | 1.5 | +1.075 |
| [oskarexists/heads_up_datathon_2026](https://github.com/oskarexists/heads_up_datathon_2026/tree/5571729ae285bd2e534e0b09c77a83a95ad2899e) | 1 | 1.5 | +0.806 | +1.321 | 2 | +1.173 |
| [dunlapjack/poker-bot](https://github.com/dunlapjack/poker-bot/tree/809521b76fb6ff8356a57e949a8379206ef875d0) | 2 | 2 | +1.546 | +1.327 | 2 | +1.591 |
| [ujjwalbana07/poker-bot](https://github.com/ujjwalbana07/poker-bot/tree/1125fd67b7b62fe051862fe4c1b7bf9976d08da8) | 2 | 2 | +1.891 | +1.339 | 2 | +1.799 |
| [Aiyshwariya/poker-bot](https://github.com/Aiyshwariya/poker-bot/tree/2ba65a83546143c9ea78d50064dcea0b968026a1) | 2 | 2 | +1.825 | +1.393 | 2 | +0.941 |
| [shaswat-singh23/datathon-poker](https://github.com/shaswat-singh23/datathon-poker/tree/ebd2caf4974ef1673223e8d584e1efb6ba65358f) | 2 | 2 | +2.174 | +1.411 | 2 | +2.220 |
| [LaserRyan/Heads-Up-Poker](https://github.com/LaserRyan/Heads-Up-Poker/tree/f051f4b34e315056370510cccb005a6275c45cab) | 2 | 2 | +2.073 | +1.541 | 2 | +1.694 |
| [Dhanush006/ssd-heads-up-poker](https://github.com/Dhanush006/ssd-heads-up-poker/tree/0f4f12c67c5ee116ade1ff4ae3eec6a34c08b668) | 2 | 2 | +1.793 | +1.594 | 2 | +1.793 |
| [tylerpham89/poker-bot-heads-up-](https://github.com/tylerpham89/poker-bot-heads-up-/tree/113dba584a2f48727436f0ad8b98d0271044932a) | 2 | 2 | +1.790 | +1.600 | 2 | +1.792 |
| [aashrithtutuku02-stack/rangerunner-plo5](https://github.com/aashrithtutuku02-stack/rangerunner-plo5/tree/b15a8402286cb93f901c84f3eabd2911be5784ea) | 2 | 2 | +1.429 | +1.634 | 2 | +1.620 |
| [andrewv426/timecasters-pokerbot](https://github.com/andrewv426/timecasters-pokerbot/tree/54118d0c191efe5328bb36c7ff246b771e6d0002) | 2 | 2 | +1.858 | +1.797 | 2 | +1.858 |
| [Utsavdotcom/aura-farmer](https://github.com/Utsavdotcom/aura-farmer/tree/070ce721b1a512f6fc949d1c72536dd7ab237624) | 2 | 2 | +2.192 | +1.824 | 2 | +1.668 |
| [Fargield/td26-poker-bot-v2](https://github.com/Fargield/td26-poker-bot-v2/tree/4ce02282e8e764766d183130d3c44b8af5d2760a) | 2 | 2 | +1.416 | +1.893 | 2 | +1.416 |
| [karammasad/pokerMAKK](https://github.com/karammasad/pokerMAKK/tree/c7ad8caf6253a66874d350cfa29c4f5f51d21cde) | 2 | 2 | +2.256 | +2.168 | 2 | +2.256 |
| [DiegoAC06/heads-up-green-fn](https://github.com/DiegoAC06/heads-up-green-fn/tree/56fce2949c1cbd7dfb23f6d668c715e3af6a0829) | 2 | 2 | +2.292 | +2.229 | 2 | +2.282 |
| [awojinrin/xplore-heads-up-poker-bot](https://github.com/awojinrin/xplore-heads-up-poker-bot/tree/20784b7b73ad7b1d6fb5bc4b08c518ff719c184a) | 2 | 2 | +2.279 | +2.377 | 2 | +2.114 |
| [shriharish1310/tamu-datathon-poker-bot](https://github.com/shriharish1310/tamu-datathon-poker-bot/tree/fd5c428475eee98edb7c38bd1e7fc0870ddbf42d) | 2 | 2 | +2.804 | +2.440 | 2 | +2.574 |
| [shreynath/PitBossEquity](https://github.com/shreynath/PitBossEquity/tree/6b2921de93674e8600659540e307df5894b9e259) | 2 | 2 | +3.232 | +2.570 | 2 | +3.232 |
| [JadonLee8/datathon26-submission](https://github.com/JadonLee8/datathon26-submission/tree/285efeae764d17efbadbc76ac5c5cc7ebc0e3719) | 2 | 2 | +3.020 | +2.683 | 2 | +3.020 |
| [PayneAM/heads-up-bundle](https://github.com/PayneAM/heads-up-bundle/tree/16d7fa032ccd2f616528d41d007b86a9de1414f9) | 2 | 2 | +3.278 | +2.739 | 2 | +2.879 |
| [jane-vo/tamu-datathon26](https://github.com/jane-vo/tamu-datathon26/tree/b681d6e5b997d883a5198645a62f6f8aa8f00891) | 2 | 2 | +2.954 | +3.195 | 2 | +3.136 |
| [snr2753423/heads-up-bot](https://github.com/snr2753423/heads-up-bot/tree/5f3a10297d526cc681af1b19aa995e4d38dd6656) | 2 | 2 | +3.389 | +3.708 | 2 | +3.421 |
| [zeyad-commits/heads_up](https://github.com/zeyad-commits/heads_up/tree/33eb79e16af039227cc1d0c4c6a568aeccc93487) | 2 | 2 | +3.950 | +4.144 | 2 | +3.950 |
| [alterj07/pokerbot](https://github.com/alterj07/pokerbot/tree/bc1f37338bc87e08c4dee9eb527de78e95c7feb5) | 2 | 2 | +4.229 | +4.279 | 2 | +4.153 |
| [kswamyget/pokerbot](https://github.com/kswamyget/pokerbot/tree/8d5a5556a5089ba15e24dfcd85b24b93ff69d549) | 2 | 2 | +5.376 | +5.313 | 2 | +5.417 |
| [danjoelsahaya/MoonLily-HeadsUp](https://github.com/danjoelsahaya/MoonLily-HeadsUp/tree/0e884ac7ce4c00bc5d00a0e4adf20e6098db2374) | 2 | 2 | +7.822 | +7.727 | 2 | +7.822 |
| [Nave3425/poker-bot](https://github.com/Nave3425/poker-bot/tree/252a8f3f96e0217997a6a0a1667d882514dde0cc) | 2 | 2 | +8.110 | +7.960 | 2 | +7.999 |
| [ruby22-gif/team-blackjack](https://github.com/ruby22-gif/team-blackjack/tree/8f15f20654f2e776f06ea69d9219546bd64bba64) | 2 | 2 | +8.697 | +8.361 | 2 | +8.697 |
| [dimi1729/2026_datathon_poker](https://github.com/dimi1729/2026_datathon_poker/tree/71e76a458640f42e62fdd7d98ea6571532e00cf5) | 2 | 2 | +10.092 | +10.980 | 2 | +10.092 |

## Speed

For the release, median time per 2,000-hand match was 14.5 s. Median request wall time was 3.35 s for the release and 10.53 s for its opponents. The largest time charged to the release in one half was 0.106 s out of 8 s. The laptop neural-action optimization reduced median selection time by 25.2% in its isolated benchmark and reproduced the same 10,000-hand output. The isolated archive comparisons used the pinned release runtime. [Speed measurements](speed.json)

## Scope

The comparison uses the organizer’s HTTP referee in isolated Linux containers with one CPU, no GPU, no internet, an eight-second time bank per half, and fresh bot processes for each half. Every opponent used its publicly shipped default strategy, including any fallback the repository itself selected. The zeyad-commits snapshot lacks its optional strategy package and played its published fallback; dimi1729 defaults to a random strategy.

Public repository and fork searches cannot reveal private submissions. Identical starter forks, byte-identical bot copies, archived-only snapshots and old strategy variants are excluded from the current-field average. A first attempt to run RangeRunner without Numba did not start; the scored comparison uses the verified native runtime on both sides. Internal self-play opponents, repeat tests selected after seeing a result, and older revisions are recorded separately in the JSON.

The confidence intervals describe duplicate-deal variation for this fixed field. They do not estimate a tournament rank. Match points follow the referee’s win/draw/loss verdict for each match separately.

[Per-match scores and referee timing](complete-score.json) · [Per-deal returns (gzip JSONL)](complete-returns.jsonl.gz) · [Per-opponent table (CSV)](public-field.csv)
