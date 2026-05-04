import json

import pandas as pd
import pytest

import matheel
from matheel.benchmark_cache import (
    benchmark_cache_key,
    benchmark_cache_key_for_run,
    benchmark_cache_paths,
    benchmark_dependency_versions,
    clear_benchmark_cache,
    load_benchmark_cache_result,
    write_benchmark_cache_result,
)


def test_benchmark_cache_helpers_are_exported_from_package_root():
    assert matheel.benchmark_cache_key is benchmark_cache_key
    assert matheel.benchmark_cache_key_for_run is benchmark_cache_key_for_run
    assert matheel.benchmark_cache_paths is benchmark_cache_paths
    assert matheel.benchmark_dependency_versions is benchmark_dependency_versions
    assert matheel.clear_benchmark_cache is clear_benchmark_cache
    assert matheel.load_benchmark_cache_result is load_benchmark_cache_result
    assert matheel.write_benchmark_cache_result is write_benchmark_cache_result


def test_benchmark_cache_key_is_deterministic_and_path_safe(tmp_path):
    algorithm_path = tmp_path / "algo.py"
    algorithm_path.write_text("def score_pair(code_a, code_b):\n    return 1.0\n", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")

    run_config = {
        "run_name": "custom",
        "options": {"algorithm_path": str(algorithm_path), "threshold": 0.5},
    }
    first = benchmark_cache_key_for_run(
        source_dir,
        run_config,
        dependency_versions={"matheel": "test"},
        seed=7,
    )
    second = benchmark_cache_key_for_run(
        source_dir,
        run_config,
        dependency_versions={"matheel": "test"},
        seed=7,
    )

    assert first == second
    assert len(first["key"]) == 64
    assert "/" not in first["key"]
    assert first["components"]["run_config"]["options"]["algorithm_path"] == "algo.py"
    assert first["components"]["extra"]["algorithm_fingerprint"]["algorithm_source_fingerprint"]["sha256"]


def test_benchmark_cache_key_changes_when_config_changes():
    source_fingerprint = {"source_type": "directory", "sha256": "abc"}
    baseline = benchmark_cache_key(
        source_fingerprint,
        {"run_name": "baseline", "options": {"threshold": 0.5}},
        dependency_versions={"matheel": "test"},
    )
    changed = benchmark_cache_key(
        source_fingerprint,
        {"run_name": "baseline", "options": {"threshold": 0.7}},
        dependency_versions={"matheel": "test"},
    )

    assert baseline["key"] != changed["key"]


def test_benchmark_cache_round_trips_results_and_attrs(tmp_path):
    cache_key = benchmark_cache_key(
        {"source_type": "directory", "sha256": "abc"},
        {"run_name": "baseline", "options": {"threshold": 0.5}},
        dependency_versions={"matheel": "test"},
    )
    results = pd.DataFrame(
        [{"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 1.0}]
    )
    results.attrs["elapsed_seconds"] = 0.25
    results.attrs["feature_set"] = "levenshtein"

    paths = write_benchmark_cache_result(tmp_path / "cache", cache_key, results, metadata={"run_name": "baseline"})
    cached = load_benchmark_cache_result(tmp_path / "cache", cache_key["key"])

    assert paths["metadata"].exists()
    assert paths["results"].exists()
    assert cached is not None
    cached_results, metadata = cached
    pd.testing.assert_frame_equal(cached_results, results)
    assert cached_results.attrs["elapsed_seconds"] == 0.25
    assert cached_results.attrs["cache_status"] == "hit"
    assert metadata["metadata"]["run_name"] == "baseline"


def test_benchmark_cache_clear_removes_cache_directory(tmp_path):
    cache_key = benchmark_cache_key(
        {"source_type": "directory", "sha256": "abc"},
        {"run_name": "baseline", "options": {}},
        dependency_versions={"matheel": "test"},
    )
    results = pd.DataFrame([{"similarity_score": 1.0}])
    write_benchmark_cache_result(tmp_path / "cache", cache_key, results)

    assert clear_benchmark_cache(tmp_path / "cache") is True
    assert load_benchmark_cache_result(tmp_path / "cache", cache_key["key"]) is None
    assert clear_benchmark_cache(tmp_path / "cache") is False


def test_benchmark_cache_rejects_invalid_keys(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        benchmark_cache_paths(tmp_path / "cache", "../bad")


def test_benchmark_cache_metadata_does_not_store_absolute_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")
    run_config = {
        "run_name": "baseline",
        "options": {"algorithm_path": str(tmp_path / "algorithm.py")},
    }
    payload = benchmark_cache_key_for_run(
        source_dir,
        run_config,
        dependency_versions={"matheel": "test"},
    )
    results = pd.DataFrame([{"similarity_score": 1.0}])
    paths = write_benchmark_cache_result(tmp_path / "cache", payload, results)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    assert str(tmp_path) not in json.dumps(metadata)
