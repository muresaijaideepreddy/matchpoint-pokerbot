# Stricter opponent classifier

Only the allowed distance between an observed betting profile and a learned opponent profile changed: 0.070 to 0.020. Both versions faced ten pinned public team bots on two freshly generated match seeds each. Every outcome below was recomputed from all 1,000 duplicate returns in its official referee match. All forty matches were fault-free.

| Model | Wins / draws / losses | Points / 20 | BB per duplicate deal |
|---|---:|---:|---:|
| frozen | 15 / 5 / 0 | 17.5 | +2.736 |
| confidence020 | 16 / 3 / 1 | 17.5 | +2.119 |

**Keep saved Weights 3.** Both policies tied on points. The stricter classifier recovered two wins against dimi1729 and an extra half point against MiniBurger, but gave up 1.5 points against aura-farmer, including a loss. It also reduced measured chip returns against several other familiar opponent styles. The initial two-opponent diagnosis was insufficient to justify a global threshold change.

| Public opponent | Weights 3 points / 2 | Strict points / 2 | Strict minus original BB/deal |
|---|---:|---:|---:|
| 650joshlee | 2 | 2 | -0.294 |
| junehwang0930-lang | 2 | 2 | -0.225 |
| harishkumar123tamu | 2 | 2 | -0.401 |
| sahithreddy-github | 2 | 2 | -0.856 |
| aashrithtutuku02-stack | 2 | 2 | -1.415 |
| dimi1729 | 1 | 2 | +0.960 |
| zuhairrr07 | 1.5 | 2 | +0.628 |
| burnerdrew2-prog | 1 | 1 | +0.775 |
| utsavdotcom | 2 | 0.5 | -4.544 |
| tylerpham89 | 2 | 2 | -0.792 |

[All per-match results and exact model hashes](summary.json). The unchanged verified archive remains in weights/weights-3.zip.
