"""Collect the isolated run's pinned inputs, reports, and all duplicate returns."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

parent = Path(sys.argv[1]).resolve()
source = parent / "negative-profile-v1"
out = source / "export"
out.mkdir(exist_ok=True)
for name in ("summary.json", "negative-image.json", "training.json", "acceptance.json", "field_response.npz"):
    shutil.copyfile(source / name, out / name)
shutil.copyfile(source / "training/plan.json", out / "training-plan.json")
for block, name in (("first", "negative-profile-eval-a"), ("second", "negative-profile-eval-b")):
    target = out / block
    target.mkdir(exist_ok=True)
    shutil.copyfile(parent / name / "panel.json", target / "panel.json")
    for arm in ("frozen", "negative"):
        dest = target / arm
        dest.mkdir(exist_ok=True)
        for file in ("summary.json", "returns.jsonl.gz", "entrant.json"):
            shutil.copyfile(parent / name / arm / file, dest / file)
        report_dir = dest / "referee"
        report_dir.mkdir(exist_ok=True)
        for report in sorted((parent / name / arm).glob("*/attempt-*/clock-linux/gpu-compact96-linux-public.json")):
            slug = report.parents[2].name
            with gzip.open(report_dir / (slug + ".json.gz"), "wb") as output:
                output.write(report.read_bytes())
traces = out / "training"
traces.mkdir(exist_ok=True)
for report in sorted((source / "training").glob("frozen-vs-*/clock-linux/gpu-compact96-linux-public.json")):
    with gzip.open(traces / (report.parents[1].name + ".json.gz"), "wb") as output:
        output.write(report.read_bytes())
manifest = {path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.rglob("*")) if path.is_file() and path.name != "files-sha256.json"}
(out / "files-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
archive = source / "evidence.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
    for path in sorted(out.rglob("*")):
        if path.is_file(): target.write(path, path.relative_to(out).as_posix())
print(json.dumps(dict(path=str(archive), bytes=archive.stat().st_size, files=len(manifest),
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
