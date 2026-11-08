# MatchPoint weights

Each ZIP contains a runnable bot with its trained models in `models/`.

| Download | Release | How it plays |
| --- | --- | --- |
| [weights-1.zip](weights-1.zip) | [v1.0.0](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.0.0) | Original MatchPoint policy. |
| [weights-2.zip](weights-2.zip) | [v1.1.0](https://github.com/muresaijaideepreddy/matchpoint-pokerbot/releases/tag/v1.1.0) | SizeMix also uses the opponent’s opening and reraising bet sizes to select a policy. |

Both versions use the same Compact96 and CFR150k weights. Extract the ZIP you want to run into its own directory. From that directory, use Python 3.12 to start the bot.

```sh
python -m pip install -r requirements.txt
python agent.py
```
