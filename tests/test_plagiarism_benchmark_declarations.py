"""Offline contracts for the checked-in public benchmark definition."""

import inspect
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from matheel.benchmark_protocol import method_configuration_id, validate_benchmark_protocol, validate_benchmark_result
from matheel.leaderboard import normalize_leaderboard_manifest
from matheel.lexical import available_lexical_metrics
from matheel.similarity import calculate_similarity
from scripts.publish_plagiarism_benchmark import _assert_portable

DEFINITION = Path(__file__).resolve().parents[1] / "benchmarks" / "plagiarism-v1"
PUBLICATION = DEFINITION.parents[1] / "gradio_app" / "benchmark_data" / "plagiarism-v1"
RESULT_BASENAME = "matheel_plagiarism_v1"


@pytest.fixture
def committed_publication():
    result_path = PUBLICATION / f"{RESULT_BASENAME}.json"
    if not result_path.is_file():
        pytest.skip("Completed benchmark artifacts have not been committed yet.")
    # Once the canonical result exists, missing companion files are a failure.
    return {
        "protocol": json.loads((DEFINITION / "protocol.json").read_text()),
        "result": json.loads(result_path.read_text()),
        "index": json.loads((PUBLICATION / "score_index.json").read_text()),
        "publication": json.loads((PUBLICATION / "publication.json").read_text()),
    }


def test_public_protocol_has_frozen_complete_declared_task_views():
    protocol = json.loads((DEFINITION / "protocol.json").read_text())
    assert validate_benchmark_protocol(protocol) is protocol
    _assert_portable(protocol)
    assert protocol["benchmark_id"] == "Matheel-Plagiarism-v1"
    assert protocol["domain"] == "plagiarism"
    assert protocol["threshold_policy"] == {"mode": "fixed", "value": 0.5}
    assert protocol["execution_policy"]["max_method_seconds"] == 1800
    assert len(protocol["datasets"]) == 16
    assert len({entry["name"] for entry in protocol["datasets"]}) == 16
    pairs = [entry for entry in protocol["datasets"] if entry["task_family"] == "pair"]
    retrieval = [entry for entry in protocol["datasets"] if entry["task_family"] == "retrieval"]
    assert len(pairs) == 5
    assert sum(entry["content"]["counts"]["pairs"] for entry in pairs) == 2284
    assert len(retrieval) == 11
    assert sum(entry["content"]["counts"]["queries"] for entry in retrieval) == 1180
    for entry in protocol["datasets"]:
        assert "path" not in entry and "validation_path" not in entry
        assert entry["domain"] == "plagiarism"
        assert entry["source_url"].startswith("https://")
        assert entry["source_revision"] not in {"main", "master", "latest"}
        assert all(re.fullmatch(r"sha256:[a-f0-9]{64}", checksum) for checksum in entry["source_checksum"])
        assert entry["expected_result_count"] > 0
        assert set(entry["similarity_options"]) == {"code_language"}
    soco = [entry for entry in retrieval if entry["name"].startswith("soco14_")]
    assert len(soco) == 10
    assert all(entry["selection"]["candidate_policy"] == "exclude_same_file" for entry in soco)
    assert sum(entry["content"]["counts"]["queries"] for entry in soco) == 825


def test_all_29_methods_are_offline_validated_unique_configurations_with_pinned_models():
    protocol = json.loads((DEFINITION / "protocol.json").read_text())
    algorithms = json.loads((DEFINITION / "methods.json").read_text())["algorithms"]
    _assert_portable(algorithms)
    config = normalize_leaderboard_manifest({"datasets": [
        {"name": entry["name"], "task": entry["task_family"], "path": entry["name"]}
        for entry in protocol["datasets"]
    ], "algorithms": algorithms})
    methods = config["algorithms"]
    assert len(methods) == 29
    assert len({method_configuration_id(method) for method in methods}) == 29
    semantic = []
    lexical = set()
    for method in methods:
        options = method["similarity_options"]
        inspect.signature(calculate_similarity).bind("left", "right", **options)
        assert options.get("preprocess_mode", "none") == "none"
        assert options.get("chunking_method", "none") == "none"
        weights = options["feature_weights"]
        if weights.get("semantic", 0) > 0:
            semantic.append(method)
            assert options["vector_backend"] in {"sentence_transformers", "model2vec", "multivector"}
            assert re.fullmatch(r"[a-f0-9]{40}", method["model_revision"])
            assert "/" in options["model_name"]
            assert options["device"] == "cpu"
            assert options["max_token_length"] == 256
        if len(weights) == 1:
            lexical.update(set(weights).intersection(available_lexical_metrics()))
    assert len(semantic) == 5
    assert {method["similarity_options"]["vector_backend"] for method in semantic} == {"sentence_transformers", "model2vec", "multivector"}
    assert lexical == set(available_lexical_metrics())


def test_documented_lexical_presets_and_python_benchmark_example_execute_offline(tmp_path, monkeypatch):
    import pandas as pd
    from matheel.datasets import write_pair_dataset
    from matheel.leaderboard_presets import get_leaderboard_algorithm_preset
    documentation = (DEFINITION.parents[1] / "docs" / "benchmark_protocol.md").read_text()
    json_blocks = re.findall(r"```json\n(.*?)\n```", documentation, re.DOTALL)
    example_methods = json.loads(json_blocks[1])["algorithms"]
    for method in example_methods:
        if isinstance(method, str):
            assert get_leaderboard_algorithm_preset(method)["name"] == method
    data = tmp_path / "pairs"
    write_pair_dataset(data, files=pd.DataFrame([
        {"file_id": "a", "text": "same"}, {"file_id": "b", "text": "same"}, {"file_id": "c", "text": "different"},
    ]), pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1}, {"left_id": "a", "right_id": "c", "label": 0},
    ]))
    context = {"manifest": {"datasets": [{"name": "pairs", "task": "pair", "path": str(data),
        "source_url": "https://example.org/synthetic-data", "source_revision": "a" * 40, "split": "test"}]}}
    python_blocks = re.findall(r"```python\n(.*?)\n```", documentation, re.DOTALL)
    monkeypatch.chdir(tmp_path)
    exec(compile(python_blocks[0], "benchmark_protocol.md", "exec"), context)
    assert context["report"]["aggregate"]["eligible"].all()
    assert (tmp_path / "results" / "benchmark.json").exists()


def test_committed_results_match_the_canonical_protocol_and_all_declared_methods(committed_publication, monkeypatch):
    protocol = committed_publication["protocol"]
    payload = committed_publication["result"]
    validated = validate_benchmark_result(payload, protocol=protocol)
    assert validated["valid"] is True
    assert validated["methods"] == 29
    assert payload["metadata"]["benchmark_protocol"] == protocol
    assert json.loads((PUBLICATION / "protocol.json").read_text()) == protocol
    assert payload["metadata"]["protocol_id"] == protocol["protocol_id"]

    # Reconstruct historical identities with their recorded release, independent
    # of the package version currently running these artifact-contract checks.
    recorded_versions = {entry["matheel_version"]
                         for entry in payload["metadata"]["method_configurations"].values()}
    assert recorded_versions == {"0.7.0"}
    monkeypatch.setattr("matheel.__version__", recorded_versions.pop())
    declarations = json.loads((DEFINITION / "methods.json").read_text())["algorithms"]
    methods = normalize_leaderboard_manifest({
        "datasets": [{"name": entry["name"], "task": entry["task_family"], "path": entry["name"]}
                     for entry in protocol["datasets"]],
        "algorithms": declarations,
    })["algorithms"]
    expected = {method_configuration_id(method): method["name"] for method in methods}
    assert len(expected) == 29
    assert set(payload["metadata"]["method_configurations"]) == set(expected)
    assert {entry["algorithm_id"] for entry in payload["metadata"]["supervised_executions"]} == set(expected)
    assert len(payload["metadata"]["supervised_executions"]) == 29
    for row in payload["per_dataset"]:
        assert row["algorithm_name"] == expected[row["algorithm_id"]]
        assert row["protocol_id"] == protocol["protocol_id"]
        assert row["domain"] == row["task_type"] == "plagiarism"
    publication = committed_publication["publication"]
    for key, value in validated.items():
        assert publication[key] == value
    assert publication["declared_methods"] == 29


def test_committed_publication_hashes_index_and_portable_file_inventory(committed_publication):
    protocol = committed_publication["protocol"]
    payload = committed_publication["result"]
    index = committed_publication["index"]
    publication = committed_publication["publication"]
    expected_files = {
        f"{RESULT_BASENAME}.json", f"{RESULT_BASENAME}_aggregate.csv",
        f"{RESULT_BASENAME}_per_dataset.csv", f"{RESULT_BASENAME}.html",
        f"{RESULT_BASENAME}_details.html", f"{RESULT_BASENAME}_reproducibility.json",
        "protocol.json", "score_index.json",
    }
    assert set(publication["artifacts"]) == expected_files
    # There is one canonical copy, with no cached sources, scores, or checkpoints.
    assert {path.name for path in PUBLICATION.iterdir()} == expected_files | {"publication.json"}
    assert all(path.is_file() for path in PUBLICATION.iterdir())
    for filename, digest in publication["artifacts"].items():
        assert re.fullmatch(r"[a-f0-9]{64}", digest)
        assert hashlib.sha256((PUBLICATION / filename).read_bytes()).hexdigest() == digest
    for table in ("per_dataset", "aggregate"):
        expected = pd.DataFrame(payload[table])
        actual = pd.read_csv(
            PUBLICATION / f"{RESULT_BASENAME}_{table}.csv",
            keep_default_na=False, na_values=[""],
        )
        assert set(actual.columns) == set(expected.columns)
        pd.testing.assert_frame_equal(
            expected.where(expected.notna(), float("nan")),
            actual.reindex(columns=expected.columns).where(actual.notna(), float("nan")),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
        )
    for path in PUBLICATION.iterdir():
        contents = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            _assert_portable(json.loads(contents))
        for private_root in (str(DEFINITION.parents[1]), "/Users/", "/private/tmp/", "/tmp/", "/home/"):
            assert private_root not in contents, f"Private path in {path.name}"
        assert not re.search(r"[A-Za-z]:\\+", contents), f"Windows path in {path.name}"

    assert index["protocol_id"] == protocol["protocol_id"]
    assert index["benchmark_id"] == protocol["benchmark_id"]
    assert index["score_artifacts"] == payload["metadata"]["score_artifacts"]
    artifacts = index["score_artifacts"]
    assert publication["score_files"] == len(artifacts)
    assert len({entry["filename"] for entry in artifacts}) == len(artifacts)
    indexed = {(entry["algorithm_id"], entry["dataset_name"]) for entry in artifacts}
    assert len(indexed) == len(artifacts)
    completed = {(row["algorithm_id"], row["dataset_name"])
                 for row in payload["per_dataset"] if row["score"] is not None}
    assert completed.issubset(indexed)
    assert re.fullmatch(r"[a-f0-9]{64}", publication["score_zip"]["sha256"])
    filename = publication["score_zip"]["filename"]
    assert Path(filename).name == filename and filename.endswith(".zip")


def test_committed_results_preserve_failed_and_incomplete_method_status(committed_publication):
    payload = committed_publication["result"]
    for row in payload["per_dataset"]:
        if row["score"] is None:
            assert row["eligible"] is False
            assert row["rank"] is None
        else:
            assert row["eligible"] is True
            assert row["rank"] is not None
            assert 0 <= row["score"] <= 1
        if row.get("error"):
            assert row["score"] is None and row["eligible"] is False
    for row in payload["aggregate"]:
        if row["eligible"]:
            assert row["complete_coverage"] is True
            assert row["coverage"] == 1
            assert row["dataset_count"] == row["expected_dataset_count"]
            assert row["mean_score"] is not None and row["rank"] is not None
        else:
            assert row["mean_score"] is None and row["median_score"] is None
            assert row["rank"] is None
    for failure in payload["metadata"].get("execution_failures", []):
        rows = [row for row in payload["per_dataset"] if row["algorithm_id"] == failure["algorithm_id"]]
        assert rows and all(row["score"] is None and row["rank"] is None for row in rows)
