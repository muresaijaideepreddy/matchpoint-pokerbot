# MatchPoint | Heads-up short-deck PLO5

**Latest tested bundle: [Weights 3](weights/weights-3.zip)** ([v1.2.0 release](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.2.0)). It adds GPU-trained lookup policies with CPU-only inference. Across 182 retained matches it recorded **130 wins, 52 draws, zero losses and zero entrant faults**. On paired tests it improved the 25-team public proxy from **20.25/25 to 22.5/25**; this is not an official leaderboard score. [Full comparison and limits](evidence/blue-improvement-20261107/README.md)

To test this version, **extract `weights/weights-3.zip` into its own directory** and run its `agent.py`. The repository-root `agent.py` still runs Weights 2. [Weights 1 and 2 remain available](weights/README.md), unchanged. The ZIP includes every model and source file needed to play; no GPU or model server is required.

The [37-opponent field report](evidence/recheck-20261107/public-field.md) compares the released policy with new challengers on the same two sets of 1,000 duplicate deals per opponent. It links the exact public opponent revisions and reports both match points and big blinds per duplicate deal.

SizeMix earned **46.5/50** official match points versus **46/50** for v1.0.0 on the same public opponents and fresh deals (200,000 played hands). The update changes preflop opponent recognition and uses the same trained weights. [50-fixture results](evidence/promotion50-20261107/README.md) · [Download SizeMix](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.1.0)

MatchPoint is a Python bot for heads-up short-deck five-card pot-limit Omaha. The v1.0.0 release won a 2,000-hand practice match against Quad-J under the organizer's Linux referee by **1,748 chips (+0.437 big blinds per hand)**. Both bots finished with zero errors, skipped decisions or timeouts. [Match evidence](evidence/) · [Download the bot and training data](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases)

The game engine, referee, Docker recipes and starter opponents come from the TAMU Datathon Poker Bot Challenge. This fork adds the MatchPoint strategy, hand evaluation tables, training tools, checkpoints and experiments. [Organizer's starter repository](https://github.com/tamu-datathon-org/heads-up-bundle)

Cloning this repository is sufficient to run the tournament bot. The release also includes the simulated training and evaluation replays, alternative model weights, and checksums. The research guide shows how to load the alternative models, including the residual network that achieved the lowest error on the blind test. [Research guide](experiments/research/README.md) · [Release assets](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.0.0)

The tournament starts `agent.py`. It serves MatchPoint SizeMix with the Compact96 value model trained on Swirl and the 150,000-traversal self-play policy. `rivercount.py` preserves the original bot for head-to-head testing. Both use the tournament's exact two-hole-card and three-board-card rule.

MatchPoint observes how often its opponent opens, reraises and folds. Against opponents who consistently open and reraise before the flop, it uses a policy learned through self-play. A compact neural policy estimates the value of each legal action against other common opponents. A Monte Carlo policy table handles opponents that raise almost every chance. After observing at least 25 opening raises and 10 reraises before the flop, SizeMix can also choose self-play against opponents who frequently open half-pot and reraise pot-sized. Only observations from the current half affect the choice.

## Run on a laptop

Use Python 3.12+ and install the development requirements in your virtual environment.

~~~sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python agent.py
~~~

The server reads `PORT` and defaults to 5008. The following command starts both bots on private loopback ports, restarts them between halves, and runs the official referee over 2,000 hands.

~~~sh
python tools/http_match.py agent.py rivercount.py --deals 1000 --seed local-match --json runs/local-match.json
~~~

Pass `--entry-script` to launch each bot as `python ENTRY.py`, the same command used by the tournament image. The local referee generates a fresh authentication token per bot and connects via loopback.

For faster comparisons, use the in-process runners. The strategy comparison expresses results in big blinds per played hand. The referee's big blinds per duplicate deal is twice that value. Each tournament match awards 1 point for a win, 0.5 for a chop and 0 for a loss. The in-process comparison excludes HTTP and the official time bank.

~~~sh
python tools/compare_agents.py --candidate agent.py --incumbent rivercount.py --seeds test-a test-b --deals 1000
python tools/league.py agent.py rivercount.py models/laptop_neural.npz models/cfr_40k.npz --seeds cup-a cup-b
~~~

## Live training dashboard

To watch an SSH training run from this laptop, start the status page in a separate terminal. Supply the same host key file used for the training connection.

~~~sh
python tools/training_status.py --target USER@GPU --remote-dir pokerbot-training/RUN --known-hosts /path/to/known_hosts
~~~

Open `http://127.0.0.1:8765/`. While the page is open it reads progress, errors and GPU usage over SSH, and displays the train and validation curves. Match comparisons come from completed reports in the local `runs` directory. The page shows the last successful observation time when the connection drops.

## GPU model comparison

The Swirl experiments trained a 64-unit neural model on the original replay, then generated 96,000 fresh hands. The expanded replay covers 544,486 decisions. We tested compact neural networks, two networks with additional card and betting-history features, and shallow boosted decision trees. The train/validation split groups all decisions from one physical deal together. Every challenger used the same self-play and aggression experts as the previous MatchPoint.

On two full matches per opponent, Compact96 recorded 14 wins across seven practice styles. Frozen MatchPoint recorded 13 wins and a chop on the same deals. Compact96 gained 0.255 BB/hand (paired 95% interval +0.133 to +0.378). On two separate confirmation seeds against value, tight and polarized opponents, the gain was 0.263 BB/hand (+0.125 to +0.402) and both bots earned six points. Direct matches against the frozen version chopped on all four seeds.

Compact96 won all 12 matches against pinned public bots. Frozen MatchPoint won 11 and chopped once on the corresponding seeds. Ten historical reference reports lack per-deal scores and several opponents adjusted their play for elapsed computation time. The 12-match record uses duplicate scoring without the external clock. With the 40,000-traversal self-play policy, Compact96 beat Quad-J at +0.624 BB/hand in a full Linux container match (95% interval +0.279 to +0.969). Both players had zero errors and zero skipped requests.

Switching Compact96 to the 150,000-traversal policy preserved all 12 public wins and improved the harder four-opponent field from 7 to 8 points out of 12. Its paired gain there was +0.208 BB/hand (+0.054 to +0.363). All three direct matches between the two configurations chopped. In the same one-CPU container test against Quad-J, the 150,000-traversal version won at +0.771 BB/hand (+0.441 to +1.100) with zero errors or skipped decisions. See `gpu_evaluation.json` for the selected checkpoint, opponent versions and further comparisons.

The referee gives each bot 15 seconds to start, one second per HTTP request, and eight seconds of charged time per 1,000-hand half after a one-millisecond allowance per request. A bot must choose an action present in `observation.legal_actions`. When the bank runs out, the referee checks or folds. Five errors forfeit a half. The timed tests used the organizer's Dockerfiles with one CPU and 4 GiB per bot, a read-only filesystem and no external network.

## Previous MatchPoint checkpoint

The following laptop results describe the 64-unit checkpoint shipped before the Swirl experiments. Its weights are in `models/laptop_neural.npz`; `evaluation.json` records its historical source and model hashes.

Across six 2,000-hand matches against the frozen RiverCount bot, MatchPoint averaged **+0.318 big blinds per played hand**, with two wins, four chops and zero losses. Four of those matches used a separate set of confirmation seeds. On those seeds, hands ending in folds contributed +0.500 BB/hand and showdowns contributed −0.194 BB/hand, for a net +0.306 BB/hand. Both bots received the same cards in opposite seats.

Two six-bot round robins produced 30 full matches. MatchPoint earned 5.5 of 10 available points and shared first place with the standalone 40,000-traversal self-play policy and the 11-rule Monte Carlo policy. We also evaluated a 192-unit neural network on the same opponent lineup as the compact 64-unit model. The bigger model earned 0.424 fewer BB/hand on average across seven styles, with a paired 95% interval from −0.601 to −0.247. MatchPoint uses the compact model.

MatchPoint played one full practice match against each of 12 publicly available team snapshots. It recorded **10 wins, 2 chops, 0 losses** (11 of 12 points). The chops were Quad-J and RangeRunner. On fresh deals it beat Quad-J (+0.477 BB/hand) and ReveillePLO (+1.207 BB/hand). A regularized 128-unit challenger also won those two matches, with a paired difference of +0.078 BB/hand versus MatchPoint across both opponents (95% interval −0.169 to +0.324). This 12-opponent comparison tested the v1.0.0 policy.

These public matches used the official game and duplicate scoring over local process pipes. The external time bank was disabled. ShortStackSage used its own fast fallback on about 79% of decisions; RangeRunner used its Windows Numba backend and recorded more than eight seconds of computation per half. Both details affect how closely their tests reproduce tournament play.

In a separate 2,000-hand HTTP match against RiverCount, the official referee launched fresh Python 3.12 processes for both halves. MatchPoint became ready in 1.612 and 1.585 seconds (limit 15 seconds). It used 0.174 and 0.192 seconds of its eight-second time bank, with zero errors or skipped decisions across 5,312 requests. It earned +0.419 BB/hand and the match chopped. This test ran on the laptop with training workers stopped.

In a full HTTP match against the public ReveillePLO snapshot, MatchPoint won at **+1.071 BB/hand** (95% interval +0.797 to +1.344). Its startups took 6.487 and 6.739 seconds. Its charged time was 5.643 and 5.518 seconds, with zero errors or skipped decisions. ReveillePLO exhausted its clock in both halves and the referee skipped 755 and 763 of its calls. Quad-J completed one timed half and missed the next startup deadline. The 12-opponent practice record above excludes these timed runs.

`evaluation.json` records the checkpoint hashes, match seeds, per-opponent results and timed referee statistics. Public opponents are pinned practice snapshots. Their code and weights stay outside the handoff.

## Train on CPU or GPU

The simulator branches from an observed decision, plays out every legal action on the same hidden deal and records terminal chips. The model sees only information available to the acting player. The neural trainer groups both seats and all streets of each deal together before splitting training and validation examples.

The following commands reproduce the compact offline learning experiment. The simulator uses CPU workers. PyTorch is required only for fitting the neural checkpoint. The tournament server uses NumPy.

~~~sh
python tools/train_rl.py --profiles incumbent station half pot random tight value --deals 2000 --workers 2 --min-samples 36 --confidence-z 2.1 --retain-z 2.8 --seed laptop-rl-train-20261107 --checkpoint runs/reproduction-table.json --dataset-dir runs/laptop-replay --eval-deals 0
python tools/fit_neural.py --dataset-dir runs/laptop-replay --output runs/fitted.npz --device cpu --epochs 12 --batch-size 512
python tools/compare_agents.py --candidate runs/fitted.npz --incumbent rivercount.py --seeds fitted-a fitted-b
~~~

On a GPU server with CUDA-enabled PyTorch, use `--device cuda` for the fitting step. `--detailed` adds suit-invariant card and betting-history inputs, `--width` sets the number of units in each hidden layer, and `--disk-backed` stages large replay tensors in temporary files. The model is exported to a NumPy checkpoint and checked against PyTorch predictions before use.

For an additional GPU experiment, extract `MatchPoint.zip` and `TrainingReplay.zip` into the same directory. The replay contains 331,714 counterfactual decisions from 124,000 simulated hands. With CUDA-enabled PyTorch installed, the following trains a challenger on both replay sources and plays it against the shipped policy on fresh deals.

~~~sh
python tools/fit_neural.py --dataset-dir runs/laptop-replay --dataset-dir runs/league-replay --output runs/gpu-challenger.npz --device cuda --width 128 --detailed --weight-decay 0.01 --patience 4 --epochs 32
python tools/compare_agents.py --candidate runs/gpu-challenger.npz --incumbent agent.py --seeds gpu-check-a gpu-check-b gpu-check-c gpu-check-d --deals 1000
~~~

The self-play trainer uses external-sampling Monte Carlo counterfactual regret minimization (MCCFR). It groups similar private equities and public betting situations into information sets. At each sampled deal it compares the traverser's legal actions and updates regrets, then exports average action frequencies. A lookup and a seeded draw select the action during a match. Use a fresh output filename for a new experiment, or `--resume` with the same configuration and a larger iteration target.

~~~sh
python tools/train_cfr.py --iterations 40000 --burnin 1000 --samples 128 --allin-samples 64 --save-every 5000 --output runs/cfr-reproduction.npz
python tools/compare_agents.py --candidate runs/cfr-reproduction.npz --incumbent rivercount.py --seeds cfr-a cfr-b
~~~

A CPU or CUDA **training** host must allow Python jobs. An OpenAI-compatible chat or embedding endpoint can review text. Tournament play is offline and all shipped policies run on CPU.

To reproduce Compact96, extract `TrainingReplay.zip` and `FreshReplay.zip` from the GitHub release into the project directory. Together they contain 544,486 training and validation examples. The released checkpoint uses the best validation epoch from the following configuration.

~~~sh
python tools/fit_neural.py --device cuda --dataset-dir runs/laptop-replay --dataset-dir runs/league-replay --dataset-dir fresh/replay --output runs/compact96-reproduction.npz --width 96 --epochs 48 --patience 6 --batch-size 1024 --learning-rate 0.0006 --weight-decay 0.015 --margin 2 --seed 7721
~~~

## Files
| File | What it is |
|---|---|
| `agent.py` | Tournament entry point. Set `PARTICIPANT` to the registered team name before submission. |
| `matchpoint.py` | MatchPoint's strategy selection and referee callbacks. |
| `rivercount.py` | Original RiverCount strategy for comparison. |
| `models/` | Frozen, CPU-readable policy checkpoints. Keep the shipped model files with the entry point. |
| `equity.py` | Omaha showdown evaluation and postflop equity. |
| `board_scores.npz` and `equity_helpers.npz` | Completed-board scores, five-card ranking tables, sampled runouts and exact river masks. Regenerate offline with `python equity.py`. |
| `preflop.bin.z` | Precomputed preflop equity for all 376,992 five-card holdings. Keep this file alongside `agent.py`. The optional offline generator is `tools/generate_preflop.py`. |
| `game.py` | The rules and game state (editing this has no effect on the evaluation platform's match referee's game). |
| `evaluator.py` | Cards and hand ranking helper tool (same as above, but its helpers are useful). |
| `judge_engine.py` | A local judge that referees a match between two running bots and prints the result. |
| `baseline-bots/` | Directory of dummy practice opponents, built from the same template as `agent.py`. |
| `test_agent.py` | Tests for your bot. Add your own at the bottom. |
| `requirements.txt` | Add all dependencies your bot requires here. Keep in mind that your Docker image will be built with this. Also must be in the root of your repo (don't move it). |
| `requirements-dev.txt` | Dependencies for local testing. |
| `.dockerignore` | Files left out of your bot's Docker image. The image size is limited, so list big files your bot doesn't need while playing. |
| `docker/` | The Dockerfiles the tournament builds your bot with (don't edit them; edits have no effect on the real build). |
| `tools/check_build.sh` | Builds your bot exactly like the tournament does and smoke-tests it. **Run this before you submit.** |
| `tools/smoke_test.py` | The quick "does this bot work?" check that `check_build.sh` (and the tournament build) runs inside your image. |

## 1. The game in one paragraph
The variant of poker that your agent must play is *Heads-Up Short Deck 5-card Pot-Limit Omaha*.

* Heads-up: 1v1 Poker against other Datathon participant agents in a tournament.
* Short Deck: 36-card deck (6 through A). Two short-deck rules: a **flush beats a full house**, and **three of a kind beats a straight**.
* 5-card Pot-Limit Omaha (PLO): Your hand is the best five cards made from **exactly 2** of your 5 hole (initially dealt) cards and **exactly 3** board/community cards. Every turn you choose one of four actions: `FOLD`, `CHECK_CALL`, `RAISE_HALF` (half pot) or `RAISE_POT` (full pot). Pot-Limit means your raise must be a function of the current pot.

Each player starts every hand with 80 chips (40 big blinds). The tournament will pit your agent against other Datathon participant submitted agents in a 1v1 format. The winner of a match will be determined after 2000 hands (1000 duplicate deals) between agents.

## 2. Set up
Your agent will initially require certain dependencies to run. Install them as such, you may need to open a venv.
```
pip install -r requirements-dev.txt
```

## 3. Strategy callbacks

The judge calls these functions in `agent.py`.

| Function | When it runs | What to do |
|---|---|---|
| `new_half(config, memory)` | Start of each 1000-hand half | Set up anything you want to track |
| `choose_action(obs, memory)` | Every time it is your turn | Return one of `obs.legal_actions` |
| `on_hand_end(result, memory)` | After every hand | Optional: learn from the result |

`Observation` in `game.py` defines the visible cards, pot, amount to call, legal actions and action history.

Keep anything you want to remember in the `memory` dict. It is wiped at the start of each half. In the tournament, each half also runs in a **brand-new copy of your server**, so nothing at all carries over between halves. Locally, one process can play both halves and only `memory` is wiped (other global variables are **not**). Using `memory` makes your bot behave the same in both places.

## 4. Play a match
Use three terminals:
```
python agent.py                             # your bot, on port 5008
python baseline-bots/baseline_random.py     # an opponent, on port 5009
python judge_engine.py --deals 200          # the judge: both halves, 200 deals each (default 1000)
```
The judge prints a report that ends with `RESULT: AGENT1_WIN`, `RESULT: AGENT2_WIN` or `RESULT: CHOP` (or `RESULT: BOTH_FORFEIT` if both bots forfeited). A bot that forfeits either half loses the match, whatever the chips say, just like in the tournament. If your bot crashes, the report shows your traceback.

Every baseline bot listens on port 5009 by default. To run two at once, give one another port, e.g. `PORT=5010 python baseline-bots/baseline_always_pot.py`, and point the judge at it with `--agent2 5010`.

When you run it yourself, `agent.py` listens on every network interface, so other machines on the same network (e.g. hackathon Wi-Fi) can reach it. Set `AGENT_TOKEN=<some secret>` when you start it if that worries you; the judge then needs the same value in `AGENT1_TOKEN` (or `AGENT2_TOKEN`).

Useful judge options:
- `--agent1 5008 --agent2 5009` picks the bots by port (or a full URL like `http://localhost:5008`).
- `--seed 7` uses different cards. The same seed always gives the same cards. The actual tournament uses a secret, randomly generated seed for every match.
- `--deals 1000` plays a full-length match (about 10 seconds with simple bots).
- `--json out.json --logs` saves the full result and every hand.

**Playing halves separately (exactly like the tournament).** Restart both bots between halves to match the tournament's fresh servers:
```
python judge_engine.py --half 1 --seed 7 --json h1.json   # agent1 in seat 0
# ...restart both bots...
python judge_engine.py --half 2 --seed 7 --json h2.json   # agent1 in seat 1
python judge_engine.py combine h1.json h2.json            # pair the halves -> verdict
```
A single half ends with `RESULT: HALF_DONE`. Its chip count is mostly card luck, so only the combined result decides the match.

## 5. Test and check your bot
First install the testing dependencies:
```
pip install -r requirements-dev.txt
```
Then check your bot for tournament compatibility:
```
python -m pytest -q
```

Then check that it **builds and runs as a Docker image**, exactly like the tournament builds it (needs [Docker](https://docs.docker.com/get-docker/); on Windows use WSL or Git Bash):
```
./tools/check_build.sh
```
It builds the base image and your bot, checks the image size, then starts your bot with match restrictions (no network, read-only files) and plays a few sample hands. It ends with `ALL CHECKS PASSED` or tells you what to fix. The first run downloads PyTorch, so it takes a few minutes.

## 6. Submit
1. Commit and **push** your bot to your **public** GitHub repo.
2. Get the full commit ID: `git rev-parse HEAD` (40 characters), or copy it from the commit's page on GitHub.
3. Submit your team name, repo URL and that commit ID in the form.

The tournament builds **exactly that commit**. Anything you push afterwards is ignored unless you edit your form response with a new commit ID.

**If your build fails** (for example, the smoke test fails or the image is too big), your bot can't play. Run `./tools/check_build.sh` on the exact commit you submit.

## 7. How a match is scored
- **Duplicate format:** a match is two halves of 1000 deals on the **same cards**. In half 1 you sit in one seat; in half 2 you sit in the other seat and get the cards your opponent had. This cancels most of the card luck. Look into duplicate dealing for more information.
- **Each half is played separately.** The tournament runs each half in its own fresh environment, with no shared storage and no network access. You can't use what you saw in one half (like your opponent's cards at showdown) in the other half.
- **Score:** your average result in big blinds per deal over both halves, with a 95% confidence interval.
- **Verdict:** you **win** (1 point) if the whole interval is above 0, **lose** (0 points) if it is below 0, and **chop** (0.5 points each) if it includes 0.

## 8. Rules your bot must follow
- **Time:** **8 seconds of thinking per half** (1000 hands), which is roughly 4–7 ms per decision on average. Network time doesn't count: the first 1 ms of every request is free. A single decision may take at most **1 second**. If your time runs out, your bot plays check-or-fold for the rest of the half.
- **Startup.** Your server must answer within **15 seconds** of starting. RiverCount prepares its evaluation tables once per server at startup.
- **Errors:** crashing, returning something that is not an action, or making an illegal move each count as one error. After **5 errors in a half**, your bot **forfeits** that half, and **a forfeit loses the whole match** (0 points), whatever the chip count. A bot that never answers `/new-half` forfeits too. Every crash inside `on_hand_end()` counts as its own error, and finished hands arrive in batches, so one bug there can forfeit you within a few hands. Run `python -m pytest -q` to catch these. (Running out of time bank is **not** an error: your bot just checks or folds for the rest of the half.)
- **Port:** the tournament sets the `PORT` environment variable. Keep the last lines of `agent.py` that read it, otherwise your agent's port is hardcoded and won't run on our evaluation pipeline.
- **Hand results arrive late:** to save time, the judge delivers finished hands together with your next decision request. `on_hand_end()` still runs once per hand, always before `choose_action()` sees the next hand.
- **Keep the server plumbing working.** If you change the routes or the reply format, the judge can't talk to your bot, and every failed call counts as an error.
- **No outside connections:** your bot has no network access during a match, and files it writes are thrown away after each half.
- **Files:** your bot's files are read-only during a match. Only `/tmp` is writable (512 MB), and it's wiped after each half.
- **Libraries:** any library pip can install is allowed. List them in `requirements.txt`. The only limit is size: **your image may add at most 3 GB on top of the base image.** Some tips:
  - The base image already includes **numpy** and **PyTorch (CPU version)**. List `torch` without a version to reuse it; other versions also work but count toward your 3 GB.
  - **There is no GPU.** If you ask for a different PyTorch version, the build installs the CPU version automatically. Don't use GPU packages (anything with a `[cuda]` extra, `jax[cuda]`, `tensorflow[and-cuda]`): they're huge and useless here.
  - **TensorFlow** works (`tensorflow`, about 2 GB), but takes most of your 3 GB.
  - **FastAPI and uvicorn are fixed** at the versions in `requirements.txt`. Asking for other versions makes the build fail.
  - Big libraries load slowly. Your server must answer within 15 seconds of starting, so check that `import`s plus loading your model fit.
- **Threads:** the base image limits PyTorch/numpy to 1 thread (`OMP_NUM_THREADS=1`), because many bots share the machine. Your bot gets about 1 CPU.
- **Model files:** commit them to your repo normally. Each file must be under GitHub's 100 MB limit. **Git LFS files are not downloaded.**
