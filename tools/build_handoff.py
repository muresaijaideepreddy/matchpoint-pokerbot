"""Package the tournament project and, optionally, the locally generated training replay."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def package(output, files):
    files = sorted(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as fp:
        staged = Path(fp.name)
    try:
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
            for path in files:
                name = path.relative_to(ROOT).as_posix()
                archive.write(path, name)
                if path.suffix == ".sh":
                    script = archive.getinfo(name)
                    script.create_system = 3
                    script.external_attr = 0o100755 << 16
        with ZipFile(staged) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Archive CRC failed for {bad}")
        staged.replace(output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"{output} ({len(files)} files, {output.stat().st_size:,} bytes) SHA-256 {digest}")
    finally:
        staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="also package the local training JSONL shards separately")
    args = parser.parse_args()
    # Positive list: test opponents from other repositories live under runs/.
    # That directory is never traversed for code, assets or model weights.
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("requirements*.txt"))
    files.extend(ROOT / name for name in ("README.md", "pytest.ini", ".gitignore", ".dockerignore",
                                         "rl_policy.json", "preflop.bin.z", "board_scores.npz", "equity_helpers.npz"))
    for pattern in ("tools/*.py", "tools/*.sh", "tools/training_status.html", "models/*.npz", "models/*.json", "baseline-bots/*.py",
                    "docker/Dockerfile.*"):
        files.extend((path for path in ROOT.glob(pattern) if path.is_file()))
    if (ROOT / "evaluation.json").is_file():
        files.append(ROOT / "evaluation.json")
    if (ROOT / "gpu_evaluation.json").is_file():
        files.append(ROOT / "gpu_evaluation.json")
    package(ROOT / "dist" / "MatchPoint.zip", files)
    if args.replay:
        shards = []
        for name in ("laptop-replay", "league-replay"):
            paths = sorted((ROOT / "runs" / name).glob("pass-*/*.jsonl"))
            if not paths:
                raise ValueError(f"No replay shards found in runs/{name}")
            for path in paths:
                with path.open(encoding="utf-8") as stream:
                    first = json.loads(stream.readline())
                if "deal_id" not in first or "returns" not in first:
                    raise ValueError(f"Unexpected replay format in {path}")
            shards.extend(paths)
        package(ROOT / "dist" / "TrainingReplay.zip", shards)


if __name__ == "__main__":
    main()
