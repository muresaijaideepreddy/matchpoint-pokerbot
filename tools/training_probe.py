"""Read a training run and print one JSON snapshot. Usage: python3 - RUN_DIR"""

import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


NUMBER = r"[+-]?(?:(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
EPOCH = re.compile(rf"^Epoch\s+(\d+)/(\d+)\s+train action gain RMSE\s+({NUMBER})\s+chips"
                   rf"(?:\s+validation(?: RMSE)?\s+({NUMBER}))?", re.I)
ROWS = re.compile(r"^Training on [^:]+:\s*([\d,]+) training rows,\s*([\d,]+) held-out rows", re.I)
RESULT = re.compile(rf"^Held-out action gain RMSE\s+({NUMBER})\s+chips\..*?"
                    rf"mean replay gain\s+({NUMBER})\s+chips/decision", re.I)
SHARD = re.compile(r"^Pass\s+\d+:\s*[^,]+,\s*([\d,]+) paired decisions\b")
CHECKSUM = re.compile(r"^[a-fA-F0-9]{64}\s+\*?(.+\.npz)\s*$")
ERROR = re.compile(r"^(?:Traceback \(most recent call last\):|(?:[\w.]+(?:Error|Exception)|"
                   r"KeyboardInterrupt|SystemExit)(?::|$)|(?:ERROR|FATAL):|CUDA is unavailable\b|"
                   r"(?:.*?: )?(?:Killed|Segmentation fault)(?:\s|$))|"
                   r"(?:: FAILED(?: open or read)?$|checksums? did NOT match|command not found$|"
                   r"No space left on device|CUDA out of memory)", re.I)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_WINDOW = 512 * 1024
FRESH_SHARDS = 48


def small_integer(path):
    try:
        if path.is_file():
            with path.open("rb") as stream:
                data = stream.read(65).strip()
            if len(data) < 64:
                return int(data)
    except (OSError, ValueError):
        pass
    return None


def log_lines(path):
    """Parse ordinary logs intact; cap reads on unexpectedly large logs."""
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        length = stream.seek(0, 2)
        stream.seek(0)
        if length <= 2 * MAX_WINDOW:
            raw = stream.read(2 * MAX_WINDOW)
        else:
            head = stream.read(MAX_WINDOW).rpartition(b"\n")[0]
            stream.seek(length - MAX_WINDOW)
            tail = stream.read(MAX_WINDOW).partition(b"\n")[2]
            raw = head + b"\n[intermediate log lines omitted]\n" + tail
    return [ANSI.sub("", text).strip() for text in raw.decode("utf-8", errors="replace").splitlines()
            if text.strip()]


def proc_stat(pid):
    """Linux process state, parent PID, and start time in ticks."""
    try:
        fields = Path("/proc", str(pid), "stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def live_pid(pid_file):
    pid = small_integer(pid_file)
    if pid is None or not 0 < pid < 2**31:
        return None
    if Path("/proc/self/stat").exists():
        proc = proc_stat(pid)
        if proc is not None:
            if proc[0] in {"Z", "X", "x", "T", "t"}:
                return None
            try:
                uptime = float(Path("/proc/uptime").read_text().split()[0])
                started = time.time() - uptime + proc[2] / os.sysconf("SC_CLK_TCK")
                if started > pid_file.stat().st_mtime + 5:
                    return None  # the launcher exited and its PID has since been reused
            except (OSError, ValueError):
                pass
            return pid
    if os.name == "nt":
        # os.kill(pid, 0) can terminate a Windows process. Only acquire a wait handle.
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return None
        try:
            return pid if kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102 else None
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except PermissionError:
        return pid
    except OSError:
        return None
    return pid


def int_text(value):
    return int(value.replace(",", ""))


def float_text(value):
    result = float(value.replace(",", ""))
    return result if math.isfinite(result) else None


def read_run(directory, run_id, label):
    fresh = run_id == "fresh"
    log = directory / "training.log"
    pid = live_pid(directory / "training.pid")
    status = small_integer(directory / "exit.code")
    errors = []
    try:
        lines = log_lines(log)
    except OSError as error:
        lines = []
        errors.append(f"Cannot read training.log: {error.strerror}")
    phase = ("Starting" if fresh else "Loading training data") if pid else "Waiting to start"
    replay_shards = 240 if run_id == "rmse-scaled" else FRESH_SHARDS
    unit, done, total = "shards" if fresh else "epochs", 0, replay_shards if fresh else None
    epochs = {}
    exported = train_rows = validation_rows = rmse = gain = gpu_name = None
    weights = checksum = None
    latest_error = -1
    for index, line in enumerate(lines):
        if " GPU " in line and line.startswith("PyTorch "):
            gpu_name = line.split(" GPU ", 1)[1].strip() or None
        if line.startswith("Fresh replay simulation:"):
            phase, unit, done, total = "Generating replay", "shards", 0, replay_shards
        elif SHARD.match(line):
            phase, unit, total = "Generating replay", "shards", replay_shards
            done = min(replay_shards, done + 1)
        elif line.startswith("Saved ") and " supported overrides to " in line:
            phase = "Replay written"
        elif line.startswith("GPU fit:"):
            phase, unit, done, total = "Loading replay for training", "epochs", 0, None
        elif (m := ROWS.match(line)):
            train_rows, validation_rows = map(int_text, m.groups())
            phase, unit, done, total = "Training network", "epochs", 0, None
        elif (m := EPOCH.match(line)):
            done, total = int(m[1]), int(m[2])
            train, valid = float_text(m[3]), float_text(m[4]) if m[4] else None
            if train is None or (m[4] is not None and valid is None):
                errors.append(f"Non-finite loss reported at epoch {done}.")
            epochs[done] = {"epoch": done, "total": total, "train": train, "validation": valid}
            phase, unit = "Training network", "epochs"
            if valid is not None:
                rmse = valid
        elif (m := re.match(r"^Exporting epoch (\d+)\b", line)):
            exported, phase = int(m[1]), "Exporting checkpoint"
        elif line.startswith("Verified NumPy export"):
            phase = "Checking held-out decisions"
        elif (m := RESULT.match(line)):
            rmse, gain = map(float_text, m.groups())
            phase = "Held-out evaluation complete"
        elif line.startswith("Saved ") and line.endswith(".npz"):
            weights, phase = line[6:].strip(), "Checkpoint saved"
            checksum = None
        elif (m := CHECKSUM.match(line)) and weights and Path(m[1]).name == Path(weights).name:
            checksum, phase = index, "Checkpoint verified"
        if ERROR.search(line):
            errors.append(line[:480])
            latest_error = index

    if status is not None and status != 0:
        state, phase = "failed", "Training failed"
        errors.append(f"Launcher exited with code {status}.")
    elif status == 0 or (checksum is not None and checksum > latest_error and run_id == "initial"):
        state = "completed"
        if checksum is not None:
            phase = "Checkpoint verified"
        elif weights is None:
            phase = "Launcher finished"
    elif latest_error >= 0:
        state, phase = "failed", "Training failed"
    elif pid:
        state = "running"
    elif lines or errors or (directory / "training.pid").is_file():
        state = "stopped"
    else:
        state = "queued"
    measured = [v for v in epochs.values() if v["validation"] is not None]
    best = exported if exported is not None else (min(measured, key=lambda e: e["validation"])["epoch"]
                                                 if measured else None)
    row = {"id": run_id, "label": label,
           "phase": phase, "state": state, "epochs": sorted(epochs.values(), key=lambda e: e["epoch"]),
           "best_epoch": best, "train_rows": train_rows, "validation_rows": validation_rows,
           "rmse": rmse, "replay_gain": gain, "logs": [line[:600] for line in lines[-30:]],
           "errors": errors[-8:], "progress": {"done": done, "total": total, "unit": unit},
           "gpu_memory_mib": None, "gpu_utilization": None}
    try:
        row["last_log_update"] = log.stat().st_mtime
    except OSError:
        row["last_log_update"] = None
    return row, pid if state == "running" else None, gpu_name


def number(raw):
    try:
        return int(raw.strip())
    except ValueError:
        return None


def smi(exe, args):
    try:
        result = subprocess.run([exe, *args], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, errors="replace", timeout=1.8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def ancestor(pid, launchers):
    """Map a GPU worker to its launcher even when the child changed directory."""
    for _ in range(64):
        if pid in launchers:
            return launchers[pid]
        stat = proc_stat(pid)
        if not stat or stat[1] <= 1 or stat[1] == pid:
            break
        pid = stat[1]
    return None


def observe_gpu(rows, launchers, reported_name):
    device = {"name": reported_name, "memory_used_mib": None, "memory_total_mib": None,
              "utilization_pct": None}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return device
    query = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"
    fmt = "--format=csv,noheader,nounits"
    cards = [r for r in csv.reader(smi(exe, [query, fmt]).splitlines(), skipinitialspace=True)
             if len(r) == 6 and number(r[0]) is not None]
    if not cards:
        return device
    owners = {}
    if launchers:
        apps = smi(exe, ["--query-compute-apps=gpu_uuid,pid,used_gpu_memory", fmt])
        for fields in csv.reader(apps.splitlines(), skipinitialspace=True):
            if len(fields) != 3 or (pid := number(fields[1])) is None:
                continue
            run = ancestor(pid, launchers)
            if run is not None:
                owners[(fields[0].strip(), pid)] = (run, number(fields[2]))
    active_cards = {uuid for uuid, _ in owners}
    index, uuid, name, used, capacity, utilization = next((r for r in cards if r[1] in active_cards), cards[0])
    device.update(name=name.strip(), memory_used_mib=number(used), memory_total_mib=number(capacity),
                  utilization_pct=number(utilization))
    local = {pid: owner for (gpu, pid), owner in owners.items() if gpu == uuid}
    if not local:
        return device
    # pmon's SM percentage is per process, unlike utilization.gpu above.
    per_pid = {}
    for line in smi(exe, ["pmon", "-i", index.strip(), "-s", "u", "-c", "1"]).splitlines():
        fields = line.split()
        if len(fields) >= 4 and number(fields[0]) == number(index) and fields[2] in {"C", "C+G"}:
            per_pid[number(fields[1])] = number(fields[3])
    for run in rows:
        workers = [pid for pid, (owner, _) in local.items() if owner == run["id"]]
        memory = [local[pid][1] for pid in workers]
        if workers and all(n is not None for n in memory):
            run["gpu_memory_mib"] = sum(memory)
        if len(workers) == 1:
            run["gpu_utilization"] = per_pid.get(workers[0])
    return device


def field_evaluations(directory, run_id):
    log = directory / "field.log"
    report = directory / "field.json"
    if not log.is_file() and not report.is_file():
        return []
    def row(label, state, detail, score=None):
        return dict(run_id=run_id, label=label, state=state, detail=detail,
                    mean=score["bb_per_hand"] if score else None, ci95=score["ci95"] if score else None)
    label = "Seven-opponent field"
    try:
        if report.is_file() and report.stat().st_size <= MAX_WINDOW:
            with report.open(encoding="utf-8") as stream:
                record = json.load(stream)
            seeds = record["seeds"]
            reference_path = str(record.get("incumbent", "")).replace("\\", "/")
            if reference_path.endswith("/cfr-router/compact96_cfr150k.py"):
                reference = "Compact96 + CFR150k"
            elif reference_path.endswith("/compact96/field_agent.py"):
                reference = "Compact96"
            else:
                reference = "MatchPoint"
            scores = {"AGENT1_WIN": 1, "CHOP": .5, "AGENT2_WIN": 0}
            field = [result for seed in seeds for result in seed["opponents"].values()]
            points = [sum(scores[r[name + "_verdict"]] for r in field) for name in ("candidate", "incumbent")]
            direct = sum(scores[seed["verdict"]] for seed in seeds)
            return [row(label, "completed", f"Candidate {points[0]:g}/{len(field)} points · "
                        f"{reference} {points[1]:g}/{len(field)}. Paired gain over {reference}.",
                        record["overall"]["field_difference"]),
                    row(f"Head-to-head vs {reference}", "completed",
                        f"Candidate {direct:g}/{len(seeds)} points · {2 * record['deals_per_seed']:,} hands per match.",
                        record["overall"]["head_to_head"])]
        lines = log_lines(log)
        code = small_integer(directory / "field.exit")
        failed = code not in (None, 0) or any(line.startswith("Traceback ") for line in lines)
        done = sum(bool(re.match(r"^vs .+: candidate [+-].*paired difference", line)) for line in lines)
        latest = lines[-1][:220] if lines else "Starting matches"
        return [row(label, "failed" if failed else "running", f"{done}/14 opponent comparisons reported. {latest}")]
    except (OSError, ValueError, KeyError, TypeError):
        return [row(label, "failed", "Unable to read the field report.")]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 - RUN_DIR")
    root = Path(sys.argv[1]).expanduser()
    entries = [(root, "initial", "Initial GPU fit")]
    if (root / "fresh").is_dir():
        entries.append((root / "fresh", "fresh", "Fresh replay and fit"))
    used_ids = {"initial", "fresh"}
    safe = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if (not safe.fullmatch(directory.name) or directory.is_symlink()
                    or not directory.is_dir() or directory.name in {"initial", "fresh"}):
                continue
            metadata = directory / "run.json"
            try:
                if metadata.is_symlink() or not metadata.is_file():
                    continue
                with metadata.open("rb") as stream:
                    raw = stream.read(4097)
                if len(raw) > 4096:
                    continue
                info = json.loads(raw)
                run_id, label = info["id"], info["label"]
                if (not isinstance(run_id, str) or not safe.fullmatch(run_id) or run_id in used_ids
                        or not isinstance(label, str) or not label.strip()):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            entries.append((directory, run_id, label.strip()[:100]))
            used_ids.add(run_id)
    rows, launchers, reported_name, evaluations = [], {}, None, []
    for directory, run_id, label in entries:
        row, launcher, name = read_run(directory, run_id, label)
        rows.append(row)
        evaluations.extend(field_evaluations(directory, run_id))
        if launcher is not None:
            launchers[launcher] = run_id
        reported_name = name or reported_name
    device = observe_gpu(rows, launchers, reported_name)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(json.dumps({"observed_at": now, "device": device, "runs": rows, "evaluations": evaluations},
                     allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
