# jai_poker

jai_poker is team **hanuman**'s bot for the TAMU Datathon heads-up short-deck PLO5 tournament. It answers `hanuman / jai_poker`.

It plays with a deep reinforcement-learning policy: a small neural network (MLP) that reads the cards, pot and betting history and picks one of the four legal actions. The weights are in `nn_policy.npz`, and `nn_policy.py` runs them with NumPy only, on CPU.

The training started from the team's earlier deep-RL population, then continued on the GPU. The opponents were:

- the earlier population;
- behaviour-cloned copies of the strongest rival bots;
- CFR self-play;
- a set of exploitable styles (calling station, nit, maniac, tight-aggressive, barreller).

The training code is not part of this branch.

If the network ever fails to load or answer, the bot falls back to the bundled CFR blueprint (`blueprint2.pkl`), so it always returns a legal action.

## Saved weights

`nn_policy.npz` holds the weights the bot plays with. Each released version is also kept in `weights/`, together with a JSON file that records its source checkpoint, SHA-256 and test results. The Docker image leaves out `weights/`.

| Version | Weights | Source | Results |
| --- | --- | --- | --- |
| **jaiv1** (played now) | `weights/jaiv1.npz` | step 2,500 | Tournament format: 61 wins, 59 draws, 0 losses in 120 matches. Draws vs the Blueprint, 5/6/0 vs the earlier deep-RL bot |
| jaiv2 | `weights/jaiv2.npz` | step 6,000 | Trainer's duplicate evaluation, BB per hand: +0.46 vs jaiv1, +0.43 vs the earlier deep-RL population, +0.22 vs the Blueprint clone. Not yet tested in tournament format. It fails `test_nn_mode_never_wastes_the_nuts`: out of position it checks a river royal flush every time. |

To play another version, copy its file over `nn_policy.npz`.

## Run it

```sh
pip install -r requirements-dev.txt
python agent.py                          # the bot, on port 5008
python baseline-bots/baseline_random.py  # an opponent, on port 5009
python judge_engine.py --deals 1000      # referee both halves
```

Before submitting, check the bot:

```sh
python -m pytest -q
./tools/check_build.sh   # builds and smoke-tests the bot like the tournament does
```

## Files

| File | What it is |
| --- | --- |
| `agent.py` | Tournament entry point: the web server and the `PARTICIPANT` / `AGENT_NAME` names |
| `strategy.py` | Decision logic. Mode `nn`, the deep-RL policy, is the default. `BOT_MODE` selects other modes for local experiments only |
| `nn_policy.py`, `nn_policy.npz` | The deep-RL policy and its weights |
| `plo_engine.py`, `betting.py`, `preflop_equity.npy`, `preflop_range.npz` | Hand evaluation, equity and features for the policy |
| `blueprint2.pkl`, `rnr_*.pkl`, `params_tuned.json` | CFR blueprint (the fallback) and the non-default modes |
| `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/` | Tournament rules, helpers, local referee and practice opponents |
| `docker/`, `tools/` | The official build files and checks (unchanged) |
