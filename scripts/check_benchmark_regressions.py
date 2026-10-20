#!/usr/bin/env python3
"""Check exact offline score fixtures; record timings without flaky time limits."""

import argparse
import json
import math
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from matheel.datasets import write_pair_dataset
from matheel.evaluation import score_pair_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "benchmark_regression.json"


def run_regression(fixture_path=DEFAULT_FIXTURE):
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    runs = []
    differences = []
    with tempfile.TemporaryDirectory(prefix="matheel-regression-") as temporary:
        dataset = write_pair_dataset(
            Path(temporary) / "pairs", files=pd.DataFrame(fixture["files"]),
            pairs=pd.DataFrame(fixture["pairs"]), metadata={"task_type": "plagiarism"},
        )
        for method in fixture["methods"]:
            started = time.perf_counter()
            scored = score_pair_dataset(dataset, similarity_options=method["options"])
            elapsed = time.perf_counter() - started
            actual = [float(value) for value in scored["similarity_score"]]
            expected = method.get("expected_scores")
            if expected is not None:
                if len(actual) != len(expected):
                    differences.append(f"{method['name']}: score count changed")
                else:
                    for index, (left, right) in enumerate(zip(actual, expected)):
                        if not math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12):
                            differences.append(f"{method['name']} pair {index}: expected {right}, received {left}")
            runs.append({"name": method["name"], "scores": actual, "elapsed_seconds": elapsed,
                         "pairs": len(actual), "options": method["options"]})
    return {"schema_version": 1, "fixture_name": fixture["name"], "passed": not differences,
            "differences": differences, "runs": runs,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {name: version(name) for name in ("matheel", "numpy", "pandas", "rapidfuzz")}},
            "timing_policy": "Informational wall times; compare trends on equivalent runners. Score parity is enforced."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=Path("benchmark-regression.json"))
    parser.add_argument("--update-baseline", action="store_true",
                        help="Explicitly replace expected scores after reviewing a deliberate scoring change.")
    args = parser.parse_args()
    report = run_regression(args.fixture)
    if args.update_baseline:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        for method, run in zip(fixture["methods"], report["runs"]):
            method["expected_scores"] = run["scores"]
        args.fixture.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        report["baseline_updated"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for difference in report["differences"]:
        print(difference)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['runs'])} configurations; report={args.output}")
    if not report["passed"] and not args.update_baseline:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
