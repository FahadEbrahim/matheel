import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from matheel.benchmark_protocol import (
    failed_benchmark_result, freeze_benchmark_protocol, method_configuration_id,
    run_benchmark_protocol, validate_benchmark_result,
)
from matheel.datasets import write_pair_dataset
from matheel.leaderboard import leaderboard_payload, normalize_leaderboard_manifest, run_leaderboard


def _manifest(tmp_path):
    data = write_pair_dataset(tmp_path / "data", files=pd.DataFrame([
        {"file_id": "a", "text": "same"}, {"file_id": "b", "text": "same"}, {"file_id": "c", "text": "other"},
    ]), pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1},
                           {"left_id": "a", "right_id": "c", "label": 0}]))
    return {"datasets": [{"name": name, "task": "pair", "path": str(data.root), "split": "test",
                           "source_url": "https://example.org/data", "source_revision": "a" * 40}
                          for name in ("first", "second")],
            "algorithms": [{"name": "bounded", "feature_weights": {"levenshtein": 1},
                            "dataset_exclusions": {"second": "Predeclared scope limitation"}}]}


def test_exclusions_are_unranked_and_never_enable_partial_averaging(tmp_path):
    manifest = _manifest(tmp_path)
    report, _ = run_leaderboard(manifest)
    rows = report["per_dataset"]
    assert rows.loc[rows["dataset_name"].eq("second"), "status"].tolist() == ["not_applicable"]
    assert rows.loc[rows["dataset_name"].eq("second"), "sample_count"].tolist() == [0]
    assert report["aggregate"]["status"].tolist() == ["not_applicable"]
    assert not report["aggregate"]["eligible"].any()
    assert report["aggregate"]["mean_score"].isna().all()
    assert report["aggregate"]["rank"].isna().all()
    assert report["aggregate"]["coverage"].tolist() == [0.5]


def test_task_exclusion_skips_execution_before_scoring(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest["algorithms"][0].pop("dataset_exclusions")
    manifest["algorithms"][0]["task_eligibility"] = {"pair": {"status": "not_applicable", "reason": "Retrieval-only configuration"}}
    monkeypatch.setattr("matheel.leaderboard._evaluate_leaderboard_run", lambda *args, **kwargs: pytest.fail("Excluded scorer must not run"))
    report, _ = run_leaderboard(manifest)
    assert report["per_dataset"]["status"].eq("not_applicable").all()
    assert report["per_dataset"]["elapsed_seconds"].eq(0).all()


def test_frozen_results_validate_declared_na_and_reject_posthoc_skips(tmp_path):
    manifest = _manifest(tmp_path)
    protocol = freeze_benchmark_protocol(manifest)
    report, _ = run_benchmark_protocol(protocol, manifest["algorithms"])
    payload = leaderboard_payload(report)
    assert validate_benchmark_result(payload)["valid"]
    row_index = next(index for index, row in enumerate(payload["per_dataset"]) if row["status"] == "not_applicable")
    for changes in ({"score": 0.5}, {"sample_count": 1}, {"not_applicable_reason": "Changed after scoring"},
                    {"status": "failed"}, {"error": "TimeoutError"}, {"elapsed_seconds": 1}):
        tampered = copy.deepcopy(payload)
        tampered["per_dataset"][row_index].update(changes)
        with pytest.raises(ValueError, match="Predeclared exclusions"):
            validate_benchmark_result(tampered)
    ordinary = next(row for row in payload["per_dataset"] if row["status"] == "completed")
    ordinary.update(status="not_applicable", not_applicable_reason="Hide a bad score", score=None)
    with pytest.raises(ValueError, match="predeclared"):
        validate_benchmark_result(payload)


def test_failure_result_keeps_predeclared_na_distinct_from_timeouts(tmp_path):
    manifest = _manifest(tmp_path)
    report, _ = failed_benchmark_result(freeze_benchmark_protocol(manifest), manifest["algorithms"])
    rows = report["per_dataset"].set_index("dataset_name")
    assert rows.loc["first", "status"] == "failed"
    assert rows.loc["first", "error"] == "TimeoutError"
    assert rows.loc["second", "status"] == "not_applicable"
    assert pd.isna(rows.loc["second", "error"])
    assert validate_benchmark_result(report)["valid"]


def test_task_eligibility_changes_identity_and_requires_a_reason(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["algorithms"][0].pop("dataset_exclusions")
    original = normalize_leaderboard_manifest(manifest)["algorithms"][0]
    manifest["algorithms"][0]["task_eligibility"] = {"retrieval": {"status": "not_applicable", "reason": "Quadratic reranking only"}}
    changed = normalize_leaderboard_manifest(manifest)["algorithms"][0]
    assert method_configuration_id(original) != method_configuration_id(changed)
    manifest["algorithms"][0]["task_eligibility"]["retrieval"].pop("reason")
    with pytest.raises(ValueError, match="requires a reason"):
        normalize_leaderboard_manifest(manifest)


def test_unknown_dataset_exclusion_is_not_silently_ignored(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["algorithms"][0]["dataset_exclusions"] = {"typo": "Not applicable"}
    with pytest.raises(ValueError, match="Unknown dataset exclusion"):
        normalize_leaderboard_manifest(manifest)


def test_fully_excluded_semantic_method_does_not_download_or_initialize_models(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    method = {"name": "Unavailable semantic method", "feature_weights": {"semantic": 1.0},
              "vector_backend": "sentence_transformers", "model_name": "offline/pinned", "model_revision": "a" * 40,
              "task_eligibility": {"pair": {"status": "not_applicable", "reason": "Unavailable for this task"}}}
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: pytest.fail("Excluded model must not download"))
    report, _ = run_benchmark_protocol(freeze_benchmark_protocol(manifest), [method])
    assert report["per_dataset"]["status"].eq("not_applicable").all()
    assert validate_benchmark_result(report)["valid"]


def test_frozen_exhaustive_protocol_rejects_shortlist_options(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["algorithms"][0]["retrieval_options"] = {"candidate_count": 1}
    with pytest.raises(ValueError, match="exhaustive candidate coverage"):
        run_benchmark_protocol(freeze_benchmark_protocol(manifest), manifest["algorithms"])


def test_published_v1_legacy_statuses_and_method_identities_still_validate():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1.json").read_text())
    assert validate_benchmark_result(payload)["methods"] == 29
