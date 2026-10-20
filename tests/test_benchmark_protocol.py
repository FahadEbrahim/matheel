import copy
import json

import pandas as pd
import pytest
from click.testing import CliRunner

from matheel.benchmark_protocol import (
    dataset_content_fingerprint,
    freeze_benchmark_protocol,
    load_benchmark_protocol,
    method_configuration_id,
    run_benchmark_protocol,
    validate_benchmark_result,
)
from matheel.benchmark_registry import compare_benchmark_runs, register_benchmark_run
from matheel.cli import main
from matheel.datasets import write_pair_dataset
from matheel.leaderboard import leaderboard_payload, run_leaderboard


def _dataset(root, prefix=""):
    write_pair_dataset(root, files=pd.DataFrame([
        {"file_id": "a", "text": prefix + "print(1)", "suffix": ".py", "assignment": prefix + "a"},
        {"file_id": "b", "text": prefix + "print(1)", "suffix": ".py", "assignment": prefix + "a"},
        {"file_id": "c", "text": prefix + "unrelated long program", "suffix": ".py", "assignment": prefix + "b"},
    ]), pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1},
        {"left_id": "a", "right_id": "c", "label": 0},
    ]), metadata={"task_type": "plagiarism"})
    return root


def _manifest(root):
    return {"domain": "plagiarism", "seed": 7, "pair_metrics": ["f1", "auroc"], "datasets": [
        {"name": "pairs", "task": "pair", "path": str(root), "source_url": "https://example.org/versioned-data",
         "source_revision": "a" * 40, "split": "test"}
    ]}


def _methods():
    return [{"name": "Levenshtein", "feature_weights": {"levenshtein": 1.0}}]


def test_freeze_evaluate_and_validate_round_trip(tmp_path):
    data = _dataset(tmp_path / "data")
    protocol = freeze_benchmark_protocol(_manifest(data), output_path=tmp_path / "protocol.json")
    loaded = load_benchmark_protocol(tmp_path / "protocol.json")
    report, artifacts = run_benchmark_protocol(loaded, _methods(), output_dir=tmp_path / "artifacts")
    payload = json.loads(artifacts["json"].read_text())
    assert validate_benchmark_result(payload)["valid"]
    assert report["aggregate"]["eligible"].all()
    assert report["aggregate"]["coverage"].eq(1).all()
    assert report["per_dataset"]["elapsed_seconds"].ge(0).all()
    assert payload["metadata"]["protocol_id"] == protocol["protocol_id"]
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["metadata"]["benchmark_protocol"]["threshold_policy"] == {"mode": "fixed", "value": 0.5}


def test_fingerprint_and_protocol_identity_are_portable(tmp_path):
    left = _dataset(tmp_path / "left")
    right = _dataset(tmp_path / "right")
    assert dataset_content_fingerprint(left, "pair") == dataset_content_fingerprint(right, "pair")
    assert freeze_benchmark_protocol(_manifest(left))["protocol_id"] == freeze_benchmark_protocol(_manifest(right))["protocol_id"]


def test_modified_frozen_data_cannot_be_evaluated(tmp_path):
    data = _dataset(tmp_path / "data")
    protocol = freeze_benchmark_protocol(_manifest(data))
    (data / "pairs.csv").write_text((data / "pairs.csv").read_text().replace("a,c,0", "a,c,1"))
    with pytest.raises(ValueError, match="content changed"):
        run_benchmark_protocol(protocol, _methods())


def test_source_text_mutation_invalidates_frozen_content(tmp_path):
    data = _dataset(tmp_path / "data")
    protocol = freeze_benchmark_protocol(_manifest(data))
    files = pd.read_csv(data / "files.csv")
    assert "text" not in files
    source = data / files.iloc[0]["file_path"]
    source.write_text("changed source program")
    with pytest.raises(ValueError, match="content changed"):
        run_benchmark_protocol(protocol, _methods())


@pytest.mark.parametrize("revision", [None, "main", "HEAD", "latest"])
def test_freeze_requires_pinned_sources(tmp_path, revision):
    manifest = _manifest(_dataset(tmp_path / "data"))
    manifest["datasets"][0]["source_revision"] = revision
    with pytest.raises(ValueError, match="immutable source_revision"):
        freeze_benchmark_protocol(manifest)


def test_method_identity_uses_configuration_and_code_not_name(tmp_path):
    first = {"name": "one", "similarity_options": {"feature_weights": {"lcs": 1}}}
    assert method_configuration_id(first) == method_configuration_id({**first, "name": "two"})
    assert method_configuration_id(first) != method_configuration_id({**first, "similarity_options": {"feature_weights": {"levenshtein": 1}}})
    path = tmp_path / "method.py"
    path.write_text("def score_pair(code_a, code_b): return 1\n")
    code = {"algorithm_path": str(path)}
    before = method_configuration_id(code)
    path.write_text("def score_pair(code_a, code_b): return 0\n")
    assert before != method_configuration_id(code)


def test_missing_nonfinite_metric_is_unranked_for_entire_aggregate(tmp_path, monkeypatch):
    import matheel.leaderboard as module
    data = _dataset(tmp_path / "data")
    manifest = _manifest(data)
    other = {**manifest["datasets"][0], "name": "second"}
    manifest["datasets"].append(other)
    protocol = freeze_benchmark_protocol(manifest)
    original = module._evaluate_leaderboard_run
    def missing(config, dataset, method):
        rows = original(config, dataset, method)
        if dataset["name"] == "second":
            rows[0]["score"] = float("inf")
        return rows
    monkeypatch.setattr(module, "_evaluate_leaderboard_run", missing)
    report, _ = run_benchmark_protocol(protocol, _methods())
    result = report["aggregate"].set_index("metric")
    assert not result.loc["f1", "eligible"]
    assert pd.isna(result.loc["f1", "rank"])
    assert pd.isna(result.loc["f1", "mean_score"])
    assert result.loc["f1", "coverage"] == 0.5
    assert result.loc["auroc", "eligible"]
    assert validate_benchmark_result(report)["valid"]


def test_failed_method_retains_null_rows_without_partial_ranking(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text('def score_pair(code_a, code_b):\n    raise RuntimeError("unsupported")\n')
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    report, _ = run_benchmark_protocol(protocol, _methods() + [{"name": "broken", "algorithm_path": str(path)}])
    broken = report["aggregate"].query('algorithm_name == "broken"')
    assert broken["rank"].isna().all()
    assert not broken["eligible"].any()
    assert report["per_dataset"].query('algorithm_name == "broken"')["error"].str.contains("unsupported").all()


@pytest.mark.parametrize("mutation", ["fingerprint", "rank", "score", "coverage", "missing"])
def test_result_import_rejects_tampering(tmp_path, mutation):
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    report, _ = run_benchmark_protocol(protocol, _methods())
    payload = leaderboard_payload(report)
    if mutation == "fingerprint":
        payload["per_dataset"][0]["dataset_fingerprint"] = "b" * 64
    elif mutation == "rank":
        payload["aggregate"][0]["rank"] = 7
    elif mutation == "score":
        payload["aggregate"][0]["mean_score"] = 0.123
    elif mutation == "coverage":
        payload["aggregate"][0]["eligible"] = False
    else:
        payload["per_dataset"].pop()
    with pytest.raises(ValueError):
        validate_benchmark_result(payload)


def test_registry_refuses_different_protocols(tmp_path):
    manifest = _manifest(_dataset(tmp_path / "data"))
    left = freeze_benchmark_protocol(manifest)
    right = freeze_benchmark_protocol({**manifest, "threshold_policy": {"mode": "fixed", "value": 0.9}})
    first, _ = run_benchmark_protocol(left, _methods())
    second, _ = run_benchmark_protocol(right, _methods())
    registry = tmp_path / "registry.json"
    register_benchmark_run(registry, first)
    register_benchmark_run(registry, second)
    with pytest.raises(ValueError, match="different benchmark protocols"):
        compare_benchmark_runs(registry)


def test_legacy_registry_refuses_different_dataset_coverage(tmp_path):
    root = _dataset(tmp_path / "data")
    first, _ = run_leaderboard({"datasets": [{"name": "one", "task": "pair", "path": str(root)}], "algorithms": _methods()})
    second = copy.deepcopy(first)
    second["per_dataset"]["dataset_name"] = "different"
    registry = tmp_path / "registry.json"
    register_benchmark_run(registry, first)
    register_benchmark_run(registry, second)
    with pytest.raises(ValueError, match="different dataset coverage"):
        compare_benchmark_runs(registry)


def test_test_threshold_overrides_are_rejected(tmp_path):
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    with pytest.raises(ValueError, match="defined by the protocol"):
        run_benchmark_protocol(protocol, [{**_methods()[0], "threshold": 0.99}])


def test_embedding_methods_require_immutable_model_revision(tmp_path):
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    with pytest.raises(ValueError, match="immutable.*model_revision"):
        run_benchmark_protocol(protocol, [{"name": "embedding", "feature_weights": {"semantic": 1}, "vector_backend": "sentence_transformers", "model_name": "sentence-transformers/all-MiniLM-L6-v2"}])


def test_validation_calibration_is_disjoint_and_records_threshold(tmp_path):
    manifest = _manifest(_dataset(tmp_path / "test"))
    manifest["threshold_policy"] = {"mode": "validation_f1"}
    manifest["datasets"][0].update(validation_path=str(_dataset(tmp_path / "validation", prefix="# validation\n")), validation_split="validation")
    protocol = freeze_benchmark_protocol(manifest)
    report, _ = run_benchmark_protocol(protocol, _methods())
    assert report["metadata"]["thresholds"][0]["dataset_name"] == "pairs"
    assert report["per_dataset"]["selected_threshold"].notna().all()
    manifest["datasets"][0]["validation_path"] = str(tmp_path / "test")
    with pytest.raises(ValueError, match="source content overlap"):
        freeze_benchmark_protocol(manifest)


def test_benchmark_cli_round_trip(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(_dataset(tmp_path / "data"))))
    methods_path = tmp_path / "methods.json"
    methods_path.write_text(json.dumps({"algorithms": _methods()}))
    protocol_path = tmp_path / "protocol.json"
    runner = CliRunner()
    frozen = runner.invoke(main, ["benchmark", "freeze", str(manifest_path), "--output", str(protocol_path)])
    assert frozen.exit_code == 0, frozen.output
    evaluated = runner.invoke(main, ["benchmark", "evaluate", str(protocol_path), str(methods_path), "--output-dir", str(tmp_path / "results"), "--format", "json"])
    assert evaluated.exit_code == 0, evaluated.output
    validated = runner.invoke(main, ["benchmark", "validate", str(tmp_path / "results" / "benchmark.json")])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"]


def test_pinned_model_executes_snapshot_but_records_portable_configuration(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    import matheel.leaderboard as module
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    seen = []
    def snapshot_download(**options):
        seen.append(options)
        return str(model_path)
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    def score(manifest, dataset, algorithm):
        assert algorithm["similarity_options"]["model_name"] == str(model_path)
        return [module._leaderboard_row("pair", dataset, algorithm, metric, 0.75, 2) for metric in manifest["pair_metrics"]]
    monkeypatch.setattr(module, "_evaluate_leaderboard_run", score)
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    method = {"name": "pinned", "vector_backend": "sentence_transformers", "model_revision": "b" * 40, "model_name": "owner/model", "feature_weights": {"semantic": 1}}
    report, _ = run_benchmark_protocol(protocol, [method])
    payload = leaderboard_payload(report)
    assert seen[0]["revision"] == "b" * 40
    assert payload["manifest"]["algorithms"][0]["similarity_options"]["model_name"] == "owner/model"
    assert str(model_path) not in json.dumps(payload)
    assert validate_benchmark_result(payload)["valid"]


def test_retrieval_candidate_policy_is_frozen_and_self_matches_are_excluded(tmp_path):
    from matheel.datasets import write_retrieval_dataset
    root = tmp_path / "retrieval"
    write_retrieval_dataset(root, files=pd.DataFrame([
        {"file_id": "a", "text": "print(1)", "suffix": ".py"},
        {"file_id": "b", "text": "print(2)", "suffix": ".py"},
    ]), queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame([{"document_id": "self", "file_id": "a"}, {"document_id": "other", "file_id": "b"}]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "other", "relevance": 1}]))
    manifest = _manifest(root)
    manifest["datasets"][0].update(task="retrieval", selection={"candidate_policy": "exclude_same_file", "query_policy": "positive_judged"})
    protocol = freeze_benchmark_protocol(manifest)
    report, _ = run_benchmark_protocol(protocol, _methods())
    assert report["per_dataset"]["sample_count"].eq(1).all()
    assert report["aggregate"]["mean_score"].eq(1).all()
    qrels = pd.read_csv(root / "qrels.csv")
    qrels["document_id"] = "self"
    qrels.to_csv(root / "qrels.csv", index=False)
    with pytest.raises(ValueError, match="Positive self-qrels"):
        freeze_benchmark_protocol(manifest)


def test_resource_budget_is_positive_finite_and_changes_protocol_identity(tmp_path):
    manifest = _manifest(_dataset(tmp_path / "data"))
    first = freeze_benchmark_protocol(manifest)
    assert first["execution_policy"]["max_method_seconds"] == 1800
    second = freeze_benchmark_protocol({**manifest, "execution_policy": {"max_method_seconds": 60}})
    assert first["protocol_id"] != second["protocol_id"]
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            freeze_benchmark_protocol({**manifest, "execution_policy": {"max_method_seconds": invalid}})


def test_nested_local_paths_are_rejected_before_public_export(tmp_path):
    manifest = _manifest(_dataset(tmp_path / "data"))
    manifest["datasets"][0]["selection"] = {"nested": {"cache": str(tmp_path)}}
    with pytest.raises(ValueError, match="local absolute paths"):
        freeze_benchmark_protocol(manifest)


def test_supervised_timeout_results_are_unranked_and_merge_with_checkpoints(tmp_path):
    from matheel.benchmark_protocol import failed_benchmark_result, merge_benchmark_results
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    complete, _ = run_benchmark_protocol(protocol, _methods())
    timeout, _ = failed_benchmark_result(protocol, [{"name": "LCS", "feature_weights": {"lcs": 1}}], elapsed_seconds=1800)
    assert timeout["aggregate"]["rank"].isna().all()
    assert timeout["aggregate"]["mean_score"].isna().all()
    assert timeout["metadata"]["execution_failures"][0]["elapsed_seconds"] == 1800
    merged, _ = merge_benchmark_results(protocol, [complete, timeout])
    assert validate_benchmark_result(merged)["valid"]
    assert merged["aggregate"].query('algorithm_name == "LCS"')["rank"].isna().all()
    assert merged["aggregate"].query('algorithm_name == "Levenshtein"')["rank"].eq(1).all()
    with pytest.raises(ValueError, match="duplicate method"):
        merge_benchmark_results(protocol, [complete, complete])


def test_score_artifacts_capture_existing_scoring_once_without_code_or_local_paths(tmp_path, monkeypatch):
    import hashlib
    import matheel.leaderboard as module
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    calls = []
    original = module.evaluate_pair_dataset
    def evaluate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "evaluate_pair_dataset", evaluate)
    score_dir = tmp_path / "scores"
    report, _ = run_benchmark_protocol(protocol, _methods(), score_artifact_dir=score_dir)
    assert len(calls) == 1
    artifact = report["metadata"]["score_artifacts"][0]
    contents = (score_dir / artifact["filename"]).read_text()
    assert str(tmp_path) not in contents
    assert "print(1)" not in contents
    assert artifact["rows"] == 2
    assert artifact["sha256"] == hashlib.sha256(contents.encode()).hexdigest()
    assert set(artifact["columns"]) == {"left_id", "right_id", "label", "similarity_score"}
    assert validate_benchmark_result(report, score_artifact_dir=score_dir)["valid"]
    (score_dir / artifact["filename"]).write_text(contents + "corrupted")
    with pytest.raises(ValueError, match="content fingerprint"):
        validate_benchmark_result(report, score_artifact_dir=score_dir)


def test_importable_custom_modules_have_distinct_fingerprinted_identities(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "benchmark_method_one.py").write_text("def score_pair(code_a, code_b): return 1.0\n")
    (tmp_path / "benchmark_method_two.py").write_text("def score_pair(code_a, code_b): return 0.0\n")
    first = method_configuration_id({"algorithm_path": "benchmark_method_one"})
    second = method_configuration_id({"algorithm_path": "benchmark_method_two"})
    assert first != second
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    report, _ = run_benchmark_protocol(protocol, [{"name": "one", "algorithm_path": "benchmark_method_one"}])
    configuration = next(iter(report["metadata"]["method_configurations"].values()))
    assert configuration["algorithm_module"] == "benchmark_method_one"
    assert configuration["algorithm_sha256"]
    assert str(tmp_path) not in json.dumps(leaderboard_payload(report))


@pytest.mark.parametrize("field,value", [("rank", 999), ("eligible", False), ("sample_count", -4), ("sample_count", 0), ("score", 2.0)])
def test_result_import_rejects_impossible_or_forged_per_dataset_values(tmp_path, field, value):
    from matheel.leaderboard import _aggregate_leaderboard
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    report, _ = run_benchmark_protocol(protocol, _methods())
    payload = leaderboard_payload(report)
    payload["per_dataset"][0][field] = value
    if field in {"sample_count", "score"}:
        aggregate = _aggregate_leaderboard(pd.DataFrame(payload["per_dataset"]), config={"strict_coverage": True, "datasets": protocol["datasets"]})
        aggregate["benchmark_id"] = protocol["benchmark_id"]
        aggregate["protocol_id"] = protocol["protocol_id"]
        payload["aggregate"] = json.loads(aggregate.to_json(orient="records"))
    with pytest.raises(ValueError):
        validate_benchmark_result(payload)


def test_dataset_overrides_cannot_bypass_method_model_revision_or_identity(tmp_path):
    manifest = _manifest(_dataset(tmp_path / "data"))
    manifest["datasets"][0]["similarity_options"] = {
        "feature_weights": {"semantic": 1}, "model_name": "owner/unpinned-model", "vector_backend": "sentence_transformers",
    }
    with pytest.raises(ValueError, match="only set code_language"):
        freeze_benchmark_protocol(manifest)


def test_official_semantic_methods_require_explicit_backend_before_snapshot_rewrite(tmp_path):
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    method = {"name": "auto", "feature_weights": {"semantic": 1}, "model_name": "owner/model", "model_revision": "a" * 40}
    with pytest.raises(ValueError, match="explicit vector_backend"):
        run_benchmark_protocol(protocol, [method])


def test_retrieval_finite_scores_require_all_frozen_candidate_comparisons(tmp_path):
    from matheel.datasets import write_retrieval_dataset
    from matheel.leaderboard import _aggregate_leaderboard
    root = tmp_path / "retrieval"
    write_retrieval_dataset(root, files=pd.DataFrame([
        {"file_id": "a", "text": "print(1)", "suffix": ".py"},
        {"file_id": "b", "text": "print(2)", "suffix": ".py"},
        {"file_id": "c", "text": "print(3)", "suffix": ".py"},
    ]), queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame([{"document_id": "self", "file_id": "a"}, {"document_id": "other", "file_id": "b"}, {"document_id": "negative", "file_id": "c"}]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "other", "relevance": 1}]))
    manifest = _manifest(root)
    manifest["datasets"][0].update(task="retrieval", selection={"candidate_policy": "exclude_same_file"})
    protocol = freeze_benchmark_protocol(manifest)
    assert protocol["datasets"][0]["expected_result_count"] == 2
    report, _ = run_benchmark_protocol(protocol, _methods())
    payload = leaderboard_payload(report)
    for row in payload["per_dataset"]:
        row["sample_count"] = 1
    aggregate = _aggregate_leaderboard(pd.DataFrame(payload["per_dataset"]), config={"strict_coverage": True, "datasets": protocol["datasets"]})
    aggregate["benchmark_id"], aggregate["protocol_id"] = protocol["benchmark_id"], protocol["protocol_id"]
    payload["aggregate"] = json.loads(aggregate.to_json(orient="records"))
    with pytest.raises(ValueError, match="complete frozen"):
        validate_benchmark_result(payload)


def test_content_fingerprint_preserves_exact_small_relevance_and_rejects_nonfinite(tmp_path):
    from matheel.datasets import write_retrieval_dataset
    root = tmp_path / "retrieval_precision"
    write_retrieval_dataset(root, files=pd.DataFrame([
        {"file_id": "q", "text": "query"}, {"file_id": "d", "text": "document"},
    ]), queries=pd.DataFrame([{"query_id": "q1", "file_id": "q"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "d"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 0.0}]))
    zero = dataset_content_fingerprint(root, "retrieval")
    qrels_path = root / "qrels.csv"
    qrels_path.write_text("query_id,document_id,relevance\nq1,d1,1e-12\n")
    small = dataset_content_fingerprint(root, "retrieval")
    assert zero != small
    qrels_path.write_text("query_id,document_id,relevance\nq1,d1,1.000000000001e-12\n")
    assert small != dataset_content_fingerprint(root, "retrieval")
    qrels_path.write_text("query_id,document_id,relevance\nq1,d1,nan\n")
    with pytest.raises(ValueError, match="finite"):
        dataset_content_fingerprint(root, "retrieval")


def test_merge_preserves_supervised_execution_provenance_for_every_method(tmp_path):
    from matheel.benchmark_protocol import merge_benchmark_results
    protocol = freeze_benchmark_protocol(_manifest(_dataset(tmp_path / "data")))
    first, _ = run_benchmark_protocol(protocol, _methods())
    second, _ = run_benchmark_protocol(protocol, [{"name": "LCS", "feature_weights": {"lcs": 1}}])
    first["metadata"]["supervised_execution"] = {"matheel_source_sha256": "a" * 64, "method_wall_seconds": 1.5}
    second["metadata"]["supervised_execution"] = {"matheel_source_sha256": "a" * 64, "method_wall_seconds": 2.5}
    merged, _ = merge_benchmark_results(protocol, [first, second])
    executions = merged["metadata"]["supervised_executions"]
    assert len(executions) == 2
    assert {entry["method_wall_seconds"] for entry in executions} == {1.5, 2.5}
    assert {entry["algorithm_id"] for entry in executions} == set(merged["metadata"]["method_configurations"])
