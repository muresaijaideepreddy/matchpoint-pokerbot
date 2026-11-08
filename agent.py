"""Tournament entry point. Run python agent.py to start MatchPoint."""

import os
from pathlib import Path
import runpy

from bot_server import create_app

PARTICIPANT = "PokerBot"  # Replace with the registered team name before submitting.

if os.environ.get("RIVERCOUNT_MODEL"):
    # Preserve the original checkpoint-selection interface for experiments.
    # A fresh module dictionary also honors new settings in a local benchmark.
    selected = runpy.run_path(str(Path(__file__).with_name("rivercount.py")))
    AGENT_NAME = selected["AGENT_NAME"]
    new_half = selected["new_half"]
    choose_action = selected["choose_action"]
    on_hand_end = selected["on_hand_end"]
else:
    from matchpoint import AGENT_NAME
    from size_mix_router import choose_release_action as choose_action, new_half, on_hand_end

app = create_app(globals())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
