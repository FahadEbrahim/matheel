"""Audit portable release evidence without datasets, model weights or detectors."""

import hashlib
import json
from pathlib import Path

import pytest

from matheel.benchmark_analysis import load_benchmark_analysis
from matheel.benchmark_protocol import _validate_public_metadata, validate_benchmark_result


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "gradio_app/benchmark_data/plagiarism-071"


def test_release_edition_inventory_and_exact_published_source_proof():
    inventory = json.loads((ROOT / "edition-manifest.json").read_text())
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    assert set(inventory["files"]) == actual - {"edition-manifest.json"}
    for relative, expected in inventory["files"].items():
        content = (ROOT / relative).read_bytes()
        assert len(content) == expected["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
    edition = json.loads((ROOT / "edition.json").read_text())
    original_plan = {key: value for key, value in edition.items() if key != "total_wall_seconds"}
    assert hashlib.sha256(json.dumps(original_plan, indent=2, sort_keys=True, allow_nan=False).encode()).hexdigest() == (
        "cbd62e2eb48cbf01d471cf91e95fdca9c8c9635f6c40f2f1da45dd2ce47b04b8"
    )
    _validate_public_metadata(edition)
    assert edition["source"]["version"] == "0.7.1"
    assert edition["source"]["git_commit"] == "64ca1364a08f76904559e19504524f80671b0479"
    assert edition["release_wheel"]["sha256"] == "b53e215f5f95aa7198eec229a3c30127ec4f38d1a1b7ca77460b7e705ca3473f"
    assert edition["release_wheel"]["all_python_modules_match_scoring_checkout"] is True
    files = edition["source"]["files"]
    assert "external_detectors/jplag.py" in files and "external_detectors/dolos.py" in files
    assert hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == edition["source"]["python_files_sha256"]
    assert set(edition["runner_source"]["files"]) == {
        "scripts/run_release_benchmarks.py", "scripts/run_plagiarism_benchmark.py",
        "scripts/publish_plagiarism_benchmark.py",
    }
    assert edition["prior_v1_unchanged"] is True
    assert edition["dataset_labels_changed"] is False
    assert edition["original_method_definitions_sha256"] == hashlib.sha256(
        (REPOSITORY / "benchmarks/plagiarism-v1/methods.json").read_bytes()).hexdigest()
    assert edition["original_protocol_id"] == json.loads(
        (REPOSITORY / "benchmarks/plagiarism-v1/protocol.json").read_text())["protocol_id"]


def test_portable_native_failure_notes_match_published_errors_without_reinterpretation():
    notes = json.loads((REPOSITORY / "benchmarks/plagiarism-071/native_failures.json").read_text())
    _validate_public_metadata(notes)
    expected = {}
    for identity in ("detectors-java", "detectors-cpp", "detectors-python"):
        result = json.loads((ROOT / identity / "matheel_plagiarism_v1.json").read_text())
        for row in result["per_dataset"]:
            if row.get("error"):
                expected[(identity, row["algorithm_id"], row["dataset_name"])] = row["error"]
    actual = {(row["edition"], row["algorithm_id"], row["dataset_name"]): row["error"]
              for row in notes["rows"]}
    assert len(notes["rows"]) == len(actual)
    assert actual == expected
    assert all(row["status"] == "failed" for row in notes["rows"])


@pytest.mark.parametrize("identity", ["all-methods", "detectors-java", "detectors-cpp", "detectors-python"])
def test_each_published_edition_matches_its_predeclared_scope_and_score_audit(identity):
    edition = json.loads((ROOT / "edition.json").read_text())
    declared = next(item for item in edition["editions"] if item["id"] == identity)
    result_path = ROOT / identity / "matheel_plagiarism_v1.json"
    result = json.loads(result_path.read_text())
    _validate_public_metadata(result)
    assert validate_benchmark_result(result)["valid"] is True
    protocol = result["metadata"]["benchmark_protocol"]
    assert protocol["protocol_id"] == declared["protocol_id"]
    assert protocol["execution_policy"] == declared["execution_policy"]
    assert result["metadata"]["method_configurations"] == declared["method_configurations"]
    assert all(config["matheel_version"] == "0.7.1" for config in declared["method_configurations"].values())
    assert [entry["name"] for entry in protocol["datasets"]] == declared["dataset_views"]
    original = json.loads((REPOSITORY / "benchmarks/plagiarism-v1/protocol.json").read_text())
    original_views = {entry["name"]: entry for entry in original["datasets"]}
    assert all(entry == original_views[entry["name"]] for entry in protocol["datasets"])
    index = json.loads((ROOT / identity / "score_index.json").read_text())
    assert index["score_artifacts"] == result["metadata"]["score_artifacts"]
    for artifact in index["score_artifacts"]:
        assert artifact["rows"] == original_views[artifact["dataset_name"]]["expected_result_count"]
        assert set(artifact["columns"]).issubset({
            "left_id", "right_id", "label", "similarity_score", "query_id", "document_id",
            "query_file_id", "document_file_id", "relevance", "row_index", "group", "group_id",
            "assignment", "assignment_id", "family", "family_id", "language", "split",
        })
    for row in result["per_dataset"]:
        if row.get("error"):
            assert row["score"] is None and row["rank"] is None and not row["eligible"]
            assert row.get("status") != "not_applicable"
    analysis = load_benchmark_analysis(result_path)
    assert analysis["rounds"] == 500 and analysis["seed"] == 7
    assert analysis["protocol_id"] == declared["protocol_id"]
