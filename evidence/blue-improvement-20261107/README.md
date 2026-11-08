# MatchPoint Weights 3

The frozen candidate recorded **130 wins, 52 draws and zero losses in 182 matches** (364,000 played hands). All candidate calls finished without referee errors, timeouts or forfeits. Results use exact public opponent revisions; they do not establish a rank against private tournament submissions.

[Runnable CPU bundle](../../weights/weights-3.zip) · [Live page](http://127.0.0.1:8765/#blueprint-live) · [Verified archive and match manifest](verified-release.json)

| Independent test | Weights 2 | Weights 3 |
|---|---:|---:|
| Blueprint, 40 matches (win/draw/loss) | 0 / 28 / 12 | **5 / 35 / 0** |
| RangeRunner, 40 matches (win/draw/loss) | 37 / 3 / 0 | **38 / 2 / 0** |
| Distinct current public teams, points / 46 | 41.5 | **43.5** |
| 25-team public proxy, points / 25 | 20.25 | **22.5** |

The two named rivals were evaluated on the same 40 fresh seeds for each policy. Weights 3 averaged +0.509 BB per duplicate deal against Blueprint (95% seed-level interval +0.355 to +0.664) and +3.259 against RangeRunner (+2.953 to +3.565). The within-match 95% confidence rule still produced many Blueprint draws. [Complete confirmation and variance analysis](confirmation.md)

The proxy selects the 25 hardest of the 46 public teams using Weights 2's earlier screening margin, then plays two new matches per team. **The requested 24.25/25 target was not met.** The public proxy is distinct from the organizer's leaderboard. Every challenger field match was fault-free on our side; AllenChirayil's opponent process exhausted its time bank in one broad-field fixture and one proxy fixture. [All public comparisons and source snapshots](field/README.md)

## What changed

Weights 3 keeps the Compact96, CFR150k and aggression fallbacks from Weights 2. It adds a CPU lookup policy trained on an RTX 5090 against Blueprint's public betting strategy. The run used **100,663,296 independently shuffled deals and 1,207,959,552 simulated betting trajectories** in 564.1 seconds. Multiple simulated betting trajectories reuse the cards in a training deal. This model represents ordered betting history, hand shape and how draws change hand strength. All-in training targets average the remaining board possibilities to reduce label noise. [Training provenance](development/gpu-100m.json) · [Blueprint architecture and source links](blueprint-model.md)

An early detector uses observed opening and reraising sizes, never an opponent name. A separate field-response classifier enables twelve specialists selected from three development fixtures each. The primary final archive was frozen before the 40-match confirmations and public-field results were read. Continued experiments with larger abstractions and risk penalties remain separate from this verified version.

## Run and reproduce

Extract the archive into its own directory, install `requirements.txt`, and start `python agent.py` with Python 3.12+. No GPU, model server or Internet connection is needed during play. The same archive passed the Windows CPU HTTP smoke test and exercised both trained policy routes across all four streets. The actual 80 primary Linux confirmation matches used less than 0.534 seconds of charged thinking time in any half, against the eight-second limit. [Windows verification](windows-smoke.json)

Archive SHA-256: `d944e836cd4d78552b5b2be92b682d9f5768ff3702f9e42c708071e212a9658a`. Weights 1 and Weights 2 remain unchanged.

From the repository, regenerate the exact archive with:

```powershell
.venv\Scripts\python.exe experiments/field_response/combine.py --gpu experiments/blue_gpu/MatchPoint-BlueGPU-Large.zip --out runs/reproduced-weights3.zip
.venv\Scripts\python.exe experiments/research/verify_blue_final.py evidence/blue-improvement-20261107
.venv\Scripts\python.exe experiments/research/verify_weights3.py
```

The regenerated archive matched the tested SHA-256 byte for byte. [Protocol and separation of training, selection and final tests](protocol.md)

## Completed follow-up experiments

The variance-aware GPU models finished another 20,971,520 independent training deals and 251,658,240 trajectories **per objective**, with shared cards across four objectives. A packaging failure at the Docker image-reference step was fixed before evaluation. All sixty development matches then completed cleanly. The lightest risk penalty cut observed volatility but tied saved Weights 3 at 7.5/12 points; higher penalties scored worse, including one genuine loss at the strongest penalty. **Keep Weights 3.** [Full variance-aware results](development/gpu-risk/README.md)

A stricter classifier threshold solved one diagnosed false match, then faced ten opponent styles on two new seeds each. Saved Weights 3 scored 15 wins/5 draws/0 losses and the stricter model 16 wins/3 draws/1 loss: both 17.5/20 points. The change did not earn promotion. The archive and the original 182-match confirmation remain unchanged. [Classifier comparison](development/confidence/README.md)
