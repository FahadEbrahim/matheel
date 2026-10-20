import json
import zipfile

import pandas as pd
import pytest

from matheel.benchmark_protocol import failed_benchmark_result, freeze_benchmark_protocol, merge_benchmark_results, public_benchmark_protocol, run_benchmark_protocol
from matheel.datasets import write_pair_dataset
from scripts.publish_plagiarism_benchmark import BASENAME, ROOT, publish_benchmark_results
from scripts.run_plagiarism_benchmark import source_fingerprint


def _completed_run(tmp_path):
    root = tmp_path / "data"
    write_pair_dataset(root, files=pd.DataFrame([
        {"file_id": "a", "text": "print(1)"}, {"file_id": "b", "text": "print(1)"},
        {"file_id": "c", "text": "unrelated program"},
    ]), pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1},
        {"left_id": "a", "right_id": "c", "label": 0},
    ]))
    protocol = freeze_benchmark_protocol({"pair_metrics": ["f1", "auroc"], "datasets": [
        {"name": "pairs", "task": "pair", "path": str(root), "split": "test",
         "source_url": "https://example.org/data", "source_revision": "a" * 40},
    ]})
    methods = [{"name": "lev", "feature_weights": {"levenshtein": 1}}, {"name": "lcs", "feature_weights": {"lcs": 1}}]
    results = tmp_path / "results"
    complete, _ = run_benchmark_protocol(protocol, [methods[0]], score_artifact_dir=results / "scores")
    timeout, _ = failed_benchmark_result(protocol, [methods[1]], elapsed_seconds=1800)
    complete["metadata"]["supervised_execution"] = {"matheel_source_sha256": source_fingerprint(), "method_wall_seconds": 1.0}
    timeout["metadata"]["supervised_execution"] = {"matheel_source_sha256": source_fingerprint()}
    merge_benchmark_results(protocol, [complete, timeout], output_dir=results, basename=BASENAME)
    definition = tmp_path / "definition"
    definition.mkdir()
    protocol_path = definition / "protocol.json"
    protocol_path.write_text(json.dumps({**public_benchmark_protocol(protocol), "protocol_id": protocol["protocol_id"]}))
    methods_path = definition / "methods.json"
    methods_path.write_text(json.dumps({"algorithms": methods}))
    return results, protocol_path, methods_path


def _publish(tmp_path, results, protocol_path, methods_path, **kwargs):
    return publish_benchmark_results(results, destination=tmp_path / "public", score_zip=tmp_path / "release" / "scores.zip",
                                     protocol_path=protocol_path, methods_path=methods_path, **kwargs)


def test_publication_validates_completed_results_and_zips_only_indexed_scores(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    (results / "scores" / "unindexed.csv").write_text("incomplete,data\n")
    summary = _publish(tmp_path, results, protocol, methods)
    assert summary["declared_methods"] == 2
    assert summary["score_files"] == 1
    public = tmp_path / "public"
    assert (public / f"{BASENAME}.json").is_file()
    assert (public / "publication.json").is_file()
    assert not (public / "scores").exists()
    assert not (public / "checkpoints").exists()
    assert str(tmp_path) not in (public / f"{BASENAME}.json").read_text()
    with zipfile.ZipFile(tmp_path / "release" / "scores.zip") as bundle:
        assert "score_index.json" in bundle.namelist()
        assert len([name for name in bundle.namelist() if name.startswith("scores/")]) == 1
        assert not any("unindexed" in name for name in bundle.namelist())
        assert "protocol.local.json" not in bundle.namelist()


def test_publication_waits_for_merged_result_before_changing_outputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing was published"):
        _publish(tmp_path, tmp_path / "missing", tmp_path / "protocol", tmp_path / "methods")
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "release").exists()


def test_publication_rejects_score_tampering_before_copying(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    score = next((results / "scores").glob("*.csv"))
    score.write_text(score.read_text() + "corrupted")
    with pytest.raises(ValueError, match="fingerprint"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_publication_rejects_missing_declared_methods(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    declaration = json.loads(methods.read_text())
    declaration["algorithms"].append({"name": "jaro", "feature_weights": {"jaro": 1}})
    methods.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="every declared method"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_publication_rejects_local_path_metadata(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    path = results / f"{BASENAME}.json"
    payload = json.loads(path.read_text())
    payload["metadata"]["private_cache"] = str(tmp_path)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="machine-local"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_score_release_bundle_cannot_be_written_into_repository(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        publish_benchmark_results(results, destination=tmp_path / "public", score_zip=ROOT / "should-not-exist.zip",
                                  protocol_path=protocol, methods_path=methods)
    assert not (ROOT / "should-not-exist.zip").exists()


def test_publication_rejects_report_csv_that_disagrees_with_validated_json(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    path = results / f"{BASENAME}_aggregate.csv"
    table = pd.read_csv(path)
    table.loc[table["eligible"], "mean_score"] = 0.123
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="CSV tables"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_publication_requires_all_supervised_method_provenance(tmp_path):
    results, protocol, methods = _completed_run(tmp_path)
    path = results / f"{BASENAME}.json"
    payload = json.loads(path.read_text())
    payload["metadata"]["supervised_executions"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="supervised execution provenance"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_publication_rejects_source_text_even_with_updated_score_hash(tmp_path):
    import hashlib
    results, protocol, methods = _completed_run(tmp_path)
    result_path = results / f"{BASENAME}.json"
    payload = json.loads(result_path.read_text())
    artifact = payload["metadata"]["score_artifacts"][0]
    score_path = results / "scores" / artifact["filename"]
    frame = pd.read_csv(score_path)
    frame["source_text"] = "private source submission"
    frame.to_csv(score_path, index=False)
    artifact["sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    artifact["columns"] = list(frame.columns)
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identifiers, labels/groups"):
        _publish(tmp_path, results, protocol, methods)
    assert not (tmp_path / "public").exists()


def test_publication_command_runs_as_documented_script_from_another_directory(tmp_path):
    import subprocess
    import sys
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "publish_plagiarism_benchmark.py"), "--help"],
                               cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--score-zip" in completed.stdout
    assert "--results-dir" in completed.stdout
