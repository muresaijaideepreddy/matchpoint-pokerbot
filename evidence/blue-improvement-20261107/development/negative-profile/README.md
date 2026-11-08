# Missing opponent-profile experiment

**Keep published Weights 3.** This candidate recovered a half point against dimi1729 but lost a half point against MiniBurger. Both versions finished at 21/24 development points, and the candidate's average chip return was slightly lower. It failed the acceptance rule frozen before evaluation, so no independent confirmation or new release was started.

## Change and test design

Weights 3 has eighteen learned betting profiles, but no dimi1729 profile. Earlier diagnostic traces showed dimi1729 sometimes selecting the PocketEdge and MiniBurger responses. A previous global distance-threshold reduction recovered performance against dimi1729 while hurting other opponents.

This experiment retained the existing 0.070 distance threshold and appended one **disabled** profile learned from two fresh Weights 3 versus dimi1729 matches: 4,000 played hands, four half-match profiles. Every existing model row remained unchanged. If the new profile is nearest, the classifier declines to select a field response and resumes the existing Blueprint-GPU-latch or Weights 2 fallback path. It does not identify an opponent by its submitted name, and it does not force the Weights 2 path in every case.

Evaluation then used twelve pinned public strategies, two new match seeds per strategy, and both versions on exactly matched cards. These 48 official referee matches contain 96,000 played hands. Training and evaluation seeds are disjoint. The organizer's CPU-only play limits apply; this adjustment needs no GPU at inference.

The predeclared development gate required at least a half-point overall improvement, no losses or faults, no point regressions against dimi1729, aura-farmer, Blueprint or RangeRunner, and no decrease in average chip return. An independent holdout would still have been required before promotion.

| Version | Wins / draws / losses | Points / 24 | BB per duplicate deal |
|---|---:|---:|---:|
| Frozen Weights 3 | 18 / 6 / 0 | 21 | +2.642354 |
| Added disabled profile | 18 / 6 / 0 | 21 | +2.613229 |

Both bots and every opponent were fault-free in all 48 matches. Maximum charged entrant time per half was 0.066406 seconds for Weights 3 and 0.069083 seconds for the candidate, within the eight-second half bank. Zero observed losses in this development panel is not a guarantee about future matches.

## Per-opponent results

Each row covers two matches per version. Chip changes are candidate minus Weights 3, in BB per duplicate deal.

| Public repository owner | Weights 3 points / 2 | Candidate points / 2 | Chip change |
|---|---:|---:|---:|
| 650JoshLee | 2 | 2 | -0.08550 |
| junehwang0930-lang | 2 | 2 | +0.32425 |
| harishkumar123tamu | 2 | 2 | -0.48900 |
| SahithReddy-GitHub | 2 | 2 | +0.14225 |
| aashrithtutuku02-stack / RangeRunner | 2 | 2 | 0.00000 |
| dimi1729 | 1.5 | 2 | +1.17275 |
| zuhairrr07 / MiniBurger | 1.5 | 1 | -0.74175 |
| burnerdrew2-prog | 1 | 1 | -0.03650 |
| Utsavdotcom / aura-farmer | 2 | 2 | 0.00000 |
| tylerpham89 / PocketEdge | 2 | 2 | -0.12450 |
| Dushyant-Betala / Blueprint | 1 | 1 | -0.66075 |
| shaswat-singh23 | 2 | 2 | +0.14925 |

The measured dimi1729 recovery supports the missing-profile diagnosis, but simple nearest-profile rejection also displaced useful routes. The aggregate result does not establish an improvement. Future work could investigate stronger evidence requirements before declining a response, or features conditioned on the betting situation; neither was tested here.

## Verification and artifacts

- [Independent verification](verified.json) recomputes every outcome from the 1,000 retained duplicate returns in each match, verifies their correspondence with full referee reports, checks seed and image identity, and checks model-row preservation.
- [Comparison and exact candidate identity](summary.json), [acceptance rule](acceptance.json), [training provenance](training.json), and [training plan](training-plan.json).
- The `first/` and `second/` folders retain each panel, both summaries, all raw returns, and all 48 compressed referee reports. `training/` retains the two complete training traces.
- [File hashes](files-sha256.json) cover the exported evidence. The independently generated verification and this README are subsequent additions.

The unchanged release archive is `weights/weights-3.zip`, SHA-256 `d944e836cd4d78552b5b2be92b682d9f5768ff3702f9e42c708071e212a9658a`. The separate experimental model hash is `3484536e806b72ea959417c02bad1fdfbbed63149e431020c5b98b2ce0b858c2`, and its Docker image is `sha256:692eca449f4f6290a65209221e3ca138859eade4d5c5b7dfe80ad85188c84171`.

From the repository root, reproduce the independent local audit with:

```text
python experiments/negative_profile/verify_evidence.py evidence/blue-improvement-20261107/development/negative-profile
```

Experiment code is in `experiments/research/negative_profile_development.py` and `experiments/negative_profile/`. The remote runner needs the existing staged Swirl Docker images, manifests, referee workspace, `trace_public.py`, and `recheck_round2.py`; it is not a self-contained training command for a clean clone. The local verification command above uses the checked-in evidence, saved Weights 3 archive, and repository dependencies.
