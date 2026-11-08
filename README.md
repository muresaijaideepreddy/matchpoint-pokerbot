# jai_poker

jai_poker is team **hanuman**'s bot for the TAMU Datathon heads-up short-deck PLO5 tournament. It answers `hanuman / jai_poker`.

It plays with a deep reinforcement-learning policy. Each network is a small MLP that reads the cards, pot and betting history and picks one of the four legal actions. `nn_policy.npz` holds three such networks from three training runs. At the start of each hand the bot picks one of them at random, and that network plays the whole hand. `nn_policy.py` runs them with NumPy only, on CPU.

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
| **jai_abc** (played now) | `weights/jai_abc.npz` | jaiA step 3,000 + jaiB step 2,500 (jaiv1) + jaiC step 3,500 | 66-bot round robin: 1st, 58.0/65, 0 losses. Field test (60 rivals × 4 seeds): 55.62/60, 0 losses. Drew jaiv1 head to head |
| jaiv1 | `weights/jaiv1.npz` | jaiB step 2,500 | 66-bot round robin: 2nd, 56.5/65, 0 losses. Field test: 54.88/60, 1 loss (to the Blueprint) |
| jaiv2 | `weights/jaiv2.npz` | jaiB step 6,000 | 66-bot round robin: 5th, 54.5/65, 0 losses. Field test: 55.00/60, 0 losses. Beat jaiv1 head to head 10/30/0. It fails `test_nn_mode_never_wastes_the_nuts`: out of position it checks a river royal flush every time. |

All three used at most 2.55 s of the 8 s time bank in any half, with no timeouts or errors.

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
