# MatchPoint weights

Each ZIP contains a runnable CPU bot with all of its trained models. No GPU or Internet connection is needed while playing.

| Download | Release | How it plays |
| --- | --- | --- |
| [weights-1.zip](weights-1.zip) | [v1.0.0](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.0.0) | Original MatchPoint policy. |
| [weights-2.zip](weights-2.zip) | [v1.1.0](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.1.0) | SizeMix also uses the opponent’s opening and reraising bet sizes to select a policy. |
| [weights-3.zip](weights-3.zip) | [v1.2.0](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.2.0) | Adds a GPU-trained response to Blueprint-like betting and selected responses learned from public opponents. |
| [matchpointv5.zip](matchpointv5.zip) | — | Weights 3 with a per-half stop-loss: a GPU-trained counter to range-reading opponents, then the balanced Blueprint strategy. This is the repository root. [Tests](../evidence/matchpointv5-20261108/README.md) |

Versions 1 and 2 use the same Compact96 and CFR150k weights. Version 3 includes those fallbacks and additional learned policies. Its SHA-256 is `d944e836cd4d78552b5b2be92b682d9f5768ff3702f9e42c708071e212a9658a`. On 40 new matches per rival it scored 5 wins/35 draws/0 losses against Blueprint and 38 wins/2 draws/0 losses against RangeRunner. [Verified scores](../evidence/blue-improvement-20261107/confirmation.md) · [Windows CPU check](../evidence/blue-improvement-20261107/windows-smoke.json)

Extract the ZIP you want to run into its own directory. From that directory, use Python 3.12+ to start the bot.

```sh
python -m pip install -r requirements.txt
python agent.py
```

The repository-root `agent.py` runs matchpointv5. To compare the bundles locally from the repository root, extract them into separate folders first:

```sh
python -m zipfile -e weights/weights-2.zip runs/test-weights2
python -m zipfile -e weights/weights-3.zip runs/test-weights3
python tools/http_match.py runs/test-weights3/agent.py runs/test-weights2/agent.py --deals 1000 --seed weights3-check-a --json runs/weights3-vs-weights2.json
```

This runs 2,000 played hands, restarting both bots between the two seats. Use several different seeds when comparing strategies; one head-to-head match does not establish performance against the field. [Paired public-opponent results](../evidence/blue-improvement-20261107/README.md)

[SHA256SUMS](SHA256SUMS) records all four bundles. Weights 3's risk-penalty and stricter-classifier follow-up experiments did not improve match points, so those changes are excluded.
