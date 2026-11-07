# Experimental poker policies

Extract `MatchPoint-Research-Models.zip` into the root of this repository. The archive places 16 supplemental NumPy checkpoints in `experiments/research/weights/`. Each model's family, exact byte size and SHA-256 are in [checkpoints.json](checkpoints.json). The tournament entrant and its default checkpoints are documented in the [project README](../../README.md).

## Run the models

Use Python 3.12 or newer and install the project's runtime dependencies. CPU inference uses NumPy. From the repository root, run

```sh
python -m pip install -r requirements.txt
python experiments/research/checkpoints.py
python experiments/research/checkpoints.py --model scaled-residual96
python experiments/research/checkpoints.py --model tabensemble --row path/to/replay.jsonl
```

The first command after installation checks the hash of every supplemental checkpoint. For action-value policies it scores a reproducible public observation and checks the model's selected action is legal. For the CFR strategies it loads the strategy table and reports the number of supported information sets. The demonstration supplies fixed equity, opponent rates and a base action. An actual game estimates these inputs from the acting player's information and prior public actions.

On the demonstration decision, `scaled-residual96` predicts gains of approximately −2.627, 0.000, +0.454 and +0.545 chips for fold, check/call, half-pot raise and pot raise. The selected action is check/call because the largest predicted gain is below the two-chip override margin. A replay row supplies that decision's own cards, board, stack sizes, betting history, base action, equity estimate, opponent aggression, observed fold rate and observed style.

## Recorded comparisons

| Checkpoint | Original grouped-validation RMSE | 24,835-decision comparison RMSE |
| --- | ---: | ---: |
| Compact96, shipped VALUE expert | 25.095 | 20.806 |
| Residual64 | 25.057 | 20.750 |
| Residual96 | 25.039 | 20.745 |
| Scaled residual96 | 25.000 | 20.697 |

Errors are in chips per legal action advantage, relative to the base action on the same deal. On a separately sealed 66,565-decision evaluation, scaled residual96 scored **27.057** versus **27.145** for Compact96. Its paired RMSE difference was −0.088 chips with a deal-bootstrap 95% interval of [−0.113, −0.064]. In a separate seven-opponent practice field, both policies earned 14/14 points. The pooled direct chip margin favored the shipped Compact96/CFR150k bot. Full comparisons are in [gpu_evaluation.json](../../gpu_evaluation.json).

The residual networks learn corrections to explicit pot, equity, call-cost and raise-cost priors. The other experiments include a four-member tabular neural ensemble with numeric embeddings, fold-anchored and suit-expert models, compact and card-feature MLPs, and CFR strategies. Their inference implementations share the repository's legal action and card representations. The tabular ensemble uses an original implementation inspired by [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html).

## Generate and fit

The residual generator and ordinary inference require the project's runtime dependencies. The two neural fitters also require PyTorch. The original GPU fits used PyTorch 2.10.0 with CUDA 12.8. Both fitters accept `--device cpu` for a smaller local run.

```sh
python experiments/research/generate_allin_replay.py --verify
python experiments/research/generate_allin_replay.py --profiles station pot half random value tight incumbent population --deals 15000 --iterations 2 --workers 6 --sample-later --initial-checkpoint models/laptop_mc.json --seed swirl-rb-scale-02 --eval-deals 0 --dataset-dir work/scaled-replay --checkpoint work/scaled-table.json
python experiments/research/fit_residual.py --replay work/original-replays --train-only work/scaled-replay --width 96 --output-dir work/scaled-fit
python experiments/research/fit_tabensemble.py --replay work/runout-replays --test-replay work/independent-replay --output-dir work/tab-fit
```

`work/original-replays` contains JSONL shards from the original three ordinary replay corpora. Use the project's `tools/train_rl.py` for original-format replays. The supplemental archive distributes models and code; the exact raw corpora are a separate input for reproducing the historical fits. For the tabular ensemble study, `tools/average_allins.py` prepared the original runout-averaged training replay. Each fitter writes a NumPy `policy.npz` and checks it against PyTorch predictions before accepting it.

The scaled generator produced 480,000 simulated hands and 1,289,302 added decisions. When all betting ends before the river, it computes training payouts from 128 sampled preflop runouts, all 253 flop runouts or all 22 turn runouts given both simulated hands. The additional shard namespace is used only for training. The original 436,043 training and 108,443 validation decisions keep their deal-grouped split, with raw validation labels. At decision time the model sees only the acting hand and public observations.
