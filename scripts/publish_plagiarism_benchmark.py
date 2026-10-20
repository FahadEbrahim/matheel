#!/usr/bin/env python3
"""Validate completed benchmark results and stage portable app/release artifacts.

The score ZIP must be outside the repository. Source submissions, local protocols,
worker checkpoints, and logs are never publication inputs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shutil
import tempfile
import zipfile

import pandas as pd

from matheel.benchmark_protocol import method_configuration_id, validate_benchmark_protocol, validate_benchmark_result
from matheel.leaderboard import normalize_leaderboard_manifest
if __package__:
    from scripts.run_plagiarism_benchmark import source_fingerprint
else:
    from run_plagiarism_benchmark import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFINITION = ROOT / "benchmarks" / "plagiarism-v1"
DEFAULT_DESTINATION = ROOT / "gradio_app" / "benchmark_data" / "plagiarism-v1"
BASENAME = "matheel_plagiarism_v1"
_PUBLIC_SUFFIXES = (".json", "_per_dataset.csv", "_aggregate.csv", ".html", "_details.html", "_reproducibility.json")
_SCORE_COLUMNS = {
    "left_id", "right_id", "label", "similarity_score", "query_id", "document_id",
    "query_file_id", "document_file_id", "relevance", "group", "group_id", "assignment",
    "assignment_id", "family", "family_id", "language", "split",
}
_LOCAL_PATH = re.compile(r"(?:/Users/|/private/(?:tmp|var)/|/home/|[A-Za-z]:\\\\)")


def _assert_portable(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_portable(str(key))
            _assert_portable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str):
        text = value.strip()
        if Path(text).is_absolute() or PureWindowsPath(text).is_absolute() or text.startswith(("~/", "file://")) or _LOCAL_PATH.search(text):
            raise ValueError("Public benchmark artifacts must not contain machine-local paths.")


def _expected_methods(protocol, methods_path):
    algorithms = json.loads(Path(methods_path).read_text(encoding="utf-8"))["algorithms"]
    manifest = normalize_leaderboard_manifest({
        "datasets": [{"name": entry["name"], "task": entry["task_family"], "path": entry["name"]} for entry in protocol["datasets"]],
        "algorithms": algorithms,
    })
    return {method_configuration_id(method): method["name"] for method in manifest["algorithms"]}


def publish_benchmark_results(results_dir, *, score_zip, destination=DEFAULT_DESTINATION,
                              protocol_path=DEFAULT_DEFINITION / "protocol.json",
                              methods_path=DEFAULT_DEFINITION / "methods.json"):
    """Publish only a complete declared method set, retaining honest timeout rows."""
    results = Path(results_dir)
    result_path = results / f"{BASENAME}.json"
    if not result_path.is_file():
        raise FileNotFoundError("Completed merged benchmark result is not available; nothing was published.")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    validate_benchmark_protocol(protocol)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    score_dir = results / "scores"
    validated = validate_benchmark_result(payload, protocol=protocol, score_artifact_dir=score_dir)
    expected = _expected_methods(protocol, methods_path)
    if set(payload["metadata"]["method_configurations"]) != set(expected):
        raise ValueError("Completed results must contain every declared method exactly once.")
    for row in payload["per_dataset"]:
        if expected[row["algorithm_id"]] != row["algorithm_name"]:
            raise ValueError("Result method names do not match the declared configurations.")
    executions = payload["metadata"].get("supervised_executions", [])
    if len(executions) != len(expected) or {entry.get("algorithm_id") for entry in executions} != set(expected):
        raise ValueError("Public results require supervised execution provenance for every declared method.")
    failure_times = {entry["algorithm_id"]: entry["elapsed_seconds"] for entry in payload["metadata"].get("execution_failures", [])}
    current_source = source_fingerprint()
    for execution in executions:
        if execution.get("matheel_source_sha256") != current_source:
            raise ValueError("Completed results were scored with different Matheel source code.")
        elapsed = execution.get("method_wall_seconds", failure_times.get(execution["algorithm_id"]))
        if elapsed is None or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
            raise ValueError("Every supervised method requires a finite nonnegative wall time.")
    _assert_portable(protocol)
    _assert_portable(payload)
    index = payload["metadata"].get("score_artifacts", [])
    indexed = {(entry["algorithm_id"], entry["dataset_name"]) for entry in index}
    if len(indexed) != len(index) or len({entry["filename"] for entry in index}) != len(index):
        raise ValueError("Score index contains duplicate method/dataset entries or filenames.")
    completed = {(row["algorithm_id"], row["dataset_name"]) for row in payload["per_dataset"] if row["score"] is not None}
    if not completed.issubset(indexed):
        raise ValueError("Every completed method/dataset score requires an audited score CSV.")
    for entry in index:
        frame = pd.read_csv(score_dir / entry["filename"], dtype=str, keep_default_na=False)
        if not set(frame.columns).issubset(_SCORE_COLUMNS):
            raise ValueError("Score CSVs may contain identifiers, labels/groups, and scores only.")
        for column in frame:
            for value in frame[column]:
                _assert_portable(value)
    source_artifacts = [results / f"{BASENAME}{suffix}" for suffix in _PUBLIC_SUFFIXES]
    for path in source_artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"Required completed artifact is missing: {path.name}")
        if path.suffix == ".json":
            _assert_portable(json.loads(path.read_text(encoding="utf-8")))
        elif _LOCAL_PATH.search(path.read_text(encoding="utf-8")):
            raise ValueError("Public report artifacts contain machine-local paths.")
    for table in ("per_dataset", "aggregate"):
        expected_table = pd.DataFrame(payload[table])
        actual_table = pd.read_csv(results / f"{BASENAME}_{table}.csv", keep_default_na=False, na_values=[""])
        if set(expected_table.columns) != set(actual_table.columns):
            raise ValueError("Public CSV columns do not match the validated result JSON.")
        actual_table = actual_table.reindex(columns=expected_table.columns)
        expected_table = expected_table.where(expected_table.notna(), float("nan"))
        actual_table = actual_table.where(actual_table.notna(), float("nan"))
        try:
            pd.testing.assert_frame_equal(expected_table, actual_table, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
        except AssertionError as exc:
            raise ValueError("Public CSV tables do not match the validated result JSON.") from exc
    zip_path = Path(score_zip).resolve()
    target = Path(destination).resolve()
    if zip_path.is_relative_to(ROOT) or zip_path.is_relative_to(target):
        raise ValueError("Score ZIP must be written outside the repository and published app directory.")
    # All result/protocol/coverage/file validation completes before any output is changed.
    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-publication-", dir=target.parent) as temporary:
        staged = Path(temporary)
        for path in source_artifacts:
            shutil.copyfile(path, staged / path.name)
        (staged / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
        score_index = {"protocol_id": protocol["protocol_id"], "benchmark_id": protocol["benchmark_id"], "score_artifacts": index}
        (staged / "score_index.json").write_text(json.dumps(score_index, indent=2, sort_keys=True), encoding="utf-8")
        staged_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("score_index.json", json.dumps(score_index, indent=2, sort_keys=True))
                bundle.writestr("protocol.json", json.dumps(protocol, indent=2, sort_keys=True))
                for entry in sorted(index, key=lambda item: item["filename"]):
                    bundle.write(score_dir / entry["filename"], arcname=f"scores/{entry['filename']}")
            summary = {
                **validated, "declared_methods": len(expected), "score_files": len(index),
                "score_zip": {"filename": zip_path.name, "sha256": hashlib.sha256(staged_zip.read_bytes()).hexdigest()},
                "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(staged.iterdir())},
            }
            (staged / "publication.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            target.mkdir(parents=True, exist_ok=True)
            for path in staged.iterdir():
                shutil.copyfile(path, target / path.name)
            staged_zip.replace(zip_path)
        finally:
            staged_zip.unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--score-zip", required=True)
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument("--protocol", default=str(DEFAULT_DEFINITION / "protocol.json"))
    parser.add_argument("--methods", default=str(DEFAULT_DEFINITION / "methods.json"))
    args = parser.parse_args()
    summary = publish_benchmark_results(args.results_dir, score_zip=args.score_zip,
                                        destination=args.destination, protocol_path=args.protocol,
                                        methods_path=args.methods)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
