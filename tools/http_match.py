"""Start two loopback-only bot servers and referee a duplicate match on a laptop.

    python tools/http_match.py challenger.py agent.py --seed local-duel --deals 1000
"""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import warmup
from game import AgentConfig, NUM_DEALS, STARTUP_TIMEOUT_SECONDS
from judge_engine import HttpAgentHandle, combine_halves, format_report, play_half


@contextmanager
def server(path, config, *, entry_script=False):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = secrets.token_hex(16)
    env = dict(os.environ, AGENT_TOKEN=token, PORT=str(port), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    entry = path
    if path.suffix in (".json", ".npz"):
        is_cfr = False
        if path.suffix == ".npz":
            with ZipFile(path) as checkpoint:
                is_cfr = "strategy_sum.npy" in checkpoint.namelist()
        if is_cfr:
            env["CFR_MODEL"] = str(path)
            entry = ROOT / "cfr_policy.py"
        else:
            env["RIVERCOUNT_MODEL"] = str(path)
            entry = ROOT / "rivercount.py"
    python = sys.executable
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        # Launch the interpreter itself so our process handle also stops the server.
        env["__PYVENV_LAUNCHER__"] = python
        python = sys._base_executable
    if entry_script:
        args = [python, str(entry)]
    else:
        args = [python, "-m", "uvicorn", f"{entry.stem}:app", "--app-dir", str(entry.parent),
                "--host", "127.0.0.1", "--port", str(port), "--no-access-log", "--log-level", "warning"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    output = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(args, cwd=entry.parent, env=env, creationflags=flags,
                                   stdout=output, stderr=subprocess.STDOUT)
    except OSError:
        output.close()
        raise
    handle = HttpAgentHandle(f"http://127.0.0.1:{port}", config, token)
    try:
        if not handle.wait_healthy(STARTUP_TIMEOUT_SECONDS):
            output.seek(0)
            detail = output.read().decode("utf-8", errors="replace")[-2000:].strip()
            message = f"{path.name} missed the {STARTUP_TIMEOUT_SECONDS:g}s startup limit (exit={process.poll()})"
            raise RuntimeError(f"{message}\n{detail}".rstrip())
        yield handle
    finally:
        handle._close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--seed", default="local-duel")
    parser.add_argument("--deals", type=int, default=NUM_DEALS)
    parser.add_argument("--entry-script", action="store_true",
                        help="start bots as python ENTRY.py, as in the tournament image; each script controls its bind address")
    parser.add_argument("--json", type=Path, help="save the referee verdict and time-bank statistics")
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.first, args.second)]
    if args.deals <= 0 or not all(p.is_file() and p.suffix in (".py", ".json", ".npz") for p in paths):
        parser.error("provide two bot entry points or policy checkpoints and a positive number of deals")
    config = AgentConfig(num_hands=args.deals)
    warmup()
    halves = []
    print(f"{paths[0].name} vs {paths[1].name}", flush=True)
    for half in (1, 2):
        print(f"Starting half {half} on {args.deals} deals", flush=True)
        with server(paths[0], config, entry_script=args.entry_script) as left, server(
            paths[1], config, entry_script=args.entry_script
        ) as right:
            halves.append(play_half(left, right, args.seed, half, args.deals))
    result = combine_halves(*halves)
    print(format_report(result), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(result.to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
