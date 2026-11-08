"""Independently verify model preservation, seed separation, clocks and all outcomes."""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as st
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from judge_engine import decide_verdict, CHOP_Z


def read(path): return json.loads(path.read_bytes())
def sha(data): return hashlib.sha256(data).hexdigest()


def main(folder):
    manifest = read(folder / "files-sha256.json")
    for name, digest in manifest.items(): assert sha((folder / name).read_bytes()) == digest, name
    model = folder / "field_response.npz"
    training = read(folder / "training.json")
    assert sha(model.read_bytes()) == training["candidate_sha256"]
    with ZipFile(ROOT / "weights/weights-3.zip") as archive:
        original = archive.read("experiments/field_response/field_response.npz")
        assert sha(original) == training["original_sha256"]
        runtime = archive.read("experiments/field_response/runtime.py").decode()
        assert "diff[nearest]<.070" in runtime
    with np.load(io.BytesIO(original), allow_pickle=False) as before, np.load(model, allow_pickle=False) as after:
        for key in before.files:
            assert np.array_equal(before[key], after[key] if key == "keys" else after[key][:-1]), key
        assert not after["enabled"][-1] and not np.any(after["q"][-1]) and not np.any(after["visits"][-1])
        assert np.allclose(after["styles"][-1], np.mean(training["half_profiles"], axis=0))
    plan = read(folder / "training-plan.json")
    trace_seeds = set(plan["seeds"])
    assert len(trace_seeds) == 2
    trace_files = list((folder / "training").glob("*.json.gz"))
    assert len(trace_files) == 2
    actual_seeds = set()
    for path in trace_files:
        data = gzip.decompress(path.read_bytes())
        report = json.loads(data)
        assert report["seed"] in trace_seeds and report["seed"] not in actual_seeds
        actual_seeds.add(report["seed"])
        assert report["staging"]["entrant"]["image_id"] == plan["parent_image"]
        assert report["staging"]["opponent"]["image_id"] == plan["opponent"]["image_id"]
        assert {half["half"] for half in report["halves"]} == {1, 2}
        assert all({hand["deal_index"] for hand in half["logs"]} == set(range(1000)) for half in report["halves"])
        source = next(row for row in training["training_sources"] if row["seed"] == report["seed"])
        assert sha(data) == source["sha256"] and report["both_completed_cleanly"]
        assert sum(len(half["logs"]) for half in report["halves"]) == 2000
    assert actual_seeds == trace_seeds
    rows = {"frozen": {}, "negative": {}}
    max_bank = {"frozen": 0., "negative": 0.}
    for block in ("first", "second"):
        panel = read(folder / block / "panel.json")
        expected = {row["slug"]: row for row in panel["opponents"]}
        assert len(expected) == 12
        for arm in rows:
            target = folder / block / arm
            summary = read(target / "summary.json")
            assert summary["completed"] == summary["total"] == 12 and not summary["errors"]
            saved = {row["slug"]: row for row in summary["rows"]}
            with gzip.open(target / "returns.jsonl.gz", "rt") as stream: raw = list(map(json.loads, stream))
            assert len(raw) == 12
            for item in raw:
                slug = item["slug"]
                seed = item["seed"]
                assert seed not in trace_seeds and seed not in rows[arm]
                assert seed == expected[slug]["seed"] == saved[slug]["seed"]
                report = json.loads(gzip.decompress((target / "referee" / (slug + ".json.gz")).read_bytes()))
                assert report["seed"] == seed and report["match"]["match_seed"] == seed
                assert report["staging"]["entrant"] == summary["entrant"]
                assert report["staging"]["opponent"] == expected[slug]["manifest"]
                values = item["paired_bb"]
                assert len(values) == 1000 and values == report["match"]["paired_bb"]
                mean = st.mean(values)
                radius = CHOP_Z * st.stdev(values) / math.sqrt(len(values))
                verdict, points, _ = decide_verdict(mean - radius, mean + radius)
                assert points["agent1"] == saved[slug]["points"] == report["match"]["points"]["agent1"]
                assert verdict == saved[slug]["verdict"] == report["match"]["verdict"]
                assert abs(mean - saved[slug]["bb_per_deal"]) < 1e-10
                for role in ("entrant", "opponent"):
                    processes = [p for p in report["processes"] if p["role"] == role]
                    assert len(processes) == 2 and all(p["startup_pass"] for p in processes)
                    actual = {key: sum(p["agent_stats"][key] for p in processes)
                              for key in ("errors", "skipped_calls", "forfeited", "timed_out")}
                    assert actual == saved[slug]["stats"][role]
                    if role == "entrant":
                        max_bank[arm] = max(max_bank[arm], *(p["agent_stats"]["time_seconds"] for p in processes))
                        assert all(p["startup_s"] <= p["startup_limit_s"] for p in processes)
                assert report["both_completed_cleanly"] == saved[slug]["clean"]
                rows[arm][seed] = saved[slug]
    assert rows["frozen"].keys() == rows["negative"].keys()
    result = {}
    for arm, values in rows.items():
        points = [row["points"] for row in values.values()]
        result[arm] = dict(matches=len(values), wins=points.count(1), draws=points.count(.5), losses=points.count(0),
                           points=sum(points), bb_per_deal=st.mean(row["bb_per_deal"] for row in values.values()),
                           fault_counts={role: {key: sum(row["stats"][role][key] for row in values.values())
                            for key in ("errors", "skipped_calls", "forfeited", "timed_out")} for role in ("entrant", "opponent")})
    summary = read(folder / "summary.json")
    for arm in result:
        for key, value in summary["scores"][arm].items(): assert abs(value - result[arm][key]) < 1e-10
    protected = set(read(folder / "acceptance.json")["no_target_point_regression"])
    aggregates = {}
    for arm, values in rows.items():
        for row in values.values():
            owner = row["repo"].split("/")[0].lower()
            aggregates.setdefault(owner, {"frozen": 0., "negative": 0.})[arm] += row["points"]
    passes = (result["negative"]["points"] >= result["frozen"]["points"] + .5 and
              result["negative"]["losses"] == 0 and
              result["negative"]["bb_per_deal"] >= result["frozen"]["bb_per_deal"] and
              all(not n for arm in result.values() for role in arm["fault_counts"].values() for n in role.values()) and
              all(row["clean"] for arm in rows.values() for row in arm.values()) and
              all(aggregates[owner]["negative"] >= aggregates[owner]["frozen"] for owner in protected))
    assert passes == summary["passes_development_gate"]
    assert all(seconds < 8 for seconds in max_bank.values())
    record = dict(scores=result, raw_outcomes_verified=48, training_matches_verified=2,
                  existing_rows_unchanged=True, training_evaluation_seeds_disjoint=True,
                  max_entrant_half_bank_seconds=max_bank,
                  passes_development_gate=summary["passes_development_gate"], decision=summary["decision"])
    (folder / "verified.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main(Path(sys.argv[1]).resolve())
