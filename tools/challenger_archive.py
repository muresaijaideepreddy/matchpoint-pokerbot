"""Shared reproducible ZIP layout for optional MatchPoint challengers."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
RELEASE_SHA = "86febd66ba061556561ecc69b9f78015c73aa031557867738824e0a7cd2aa50a"
RELEASE_FILES = (
    ".dockerignore", "board_scores.npz", "bot_server.py", "cfr_policy.py",
    "docker/Dockerfile.agent", "docker/Dockerfile.base", "equity.py", "equity_helpers.npz",
    "evaluator.py", "game.py", "hand_potential.py", "judge_engine.py", "matchpoint.py",
    "models/laptop_mc.json", "models/swirl_compact96.npz", "neural_policy.py",
    "preflop.bin.z", "requirements.txt", "rivercount.py", "rl_policy.json", "rl_policy.py",
    "tools/check_build.sh", "tools/smoke_test.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}; expected {expected}")
    return data


def write_archive(files, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        output = Path(temporary.name)
    try:
        with ZipFile(output, "w") as archive:
            for name in sorted(files):
                item = ZipInfo(name, date_time=(2026, 11, 7, 0, 0, 0))
                item.create_system = 3
                item.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                archive.writestr(item, files[name], compress_type=ZIP_DEFLATED, compresslevel=9)
        output.replace(target)
    finally:
        output.unlink(missing_ok=True)
    print(f"{digest(target.read_bytes())}  {target}")
