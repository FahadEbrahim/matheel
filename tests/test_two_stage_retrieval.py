import json

import pandas as pd
import pytest
from click.testing import CliRunner

from matheel.cli import main
from matheel.datasets import write_retrieval_dataset
from matheel.evaluation import evaluate_retrieval_dataset
from matheel.retrieval import (
    candidate_recall_diagnostics,
    evaluate_two_stage_retrieval_dataset,
    select_retrieval_candidates,
)


@pytest.fixture
def dataset(tmp_path):
    return write_retrieval_dataset(tmp_path / "data", files=pd.DataFrame([
        {"file_id": key, "text": text} for key, text in {
            "q": "alpha beta gamma", "r": "delta epsilon zeta", "zero": "no relevant documents",
            "a": "alpha beta gamma", "b": "unrelated", "c": "delta epsilon zeta", "d": "alpha gamma",
        }.items()
    ]), queries=pd.DataFrame([{"query_id": query, "file_id": file_id}
                             for query, file_id in (("q1", "q"), ("q2", "r"), ("q0", "zero"))]),
        corpus=pd.DataFrame([{"document_id": key, "file_id": key} for key in ("d", "c", "b", "a")]),
        qrels=pd.DataFrame([
            {"query_id": "q1", "document_id": "a", "relevance": 1},
            {"query_id": "q1", "document_id": "d", "relevance": 1},
            {"query_id": "q2", "document_id": "c", "relevance": 1},
            {"query_id": "q0", "document_id": "b", "relevance": 0},
        ]))


def test_shortlist_is_bounded_stable_and_does_not_see_labels(dataset):
    calls = []
    def constant(left, right, row):
        assert "relevance" not in row
        calls.append((row["query_id"], row["document_id"]))
        return 0.5
    candidates = select_retrieval_candidates(dataset, 2, scorer=constant)
    assert len(calls) == 12
    assert len(candidates) == 6
    assert candidates.groupby("query_id")["document_id"].apply(list).to_dict() == {
        "q0": ["a", "b"], "q1": ["a", "b"], "q2": ["a", "b"],
    }
    assert candidates.groupby("query_id")["candidate_rank"].apply(list).tolist() == [[1, 2]] * 3


def test_reranking_penalizes_missed_positives_using_full_qrels_and_every_query(dataset):
    expensive_calls = []
    def expensive(left, right, row):
        assert "relevance" not in row
        expensive_calls.append(row["document_id"])
        return 1.0
    scored, metrics = evaluate_two_stage_retrieval_dataset(
        dataset, 1, k=1, candidate_scorer=lambda *args: 0.0, scorer=expensive,
    )
    assert expensive_calls == ["a"] * 3
    assert metrics["first_stage_comparisons"] == 12
    assert metrics["reranker_comparisons"] == 3
    assert metrics["candidate_recall"] == 0.25
    assert metrics["candidate_recall_micro"] == pytest.approx(1 / 3)
    assert metrics["queries_without_positive_judgments"] == 1
    assert metrics["query_count"] == 3
    assert metrics["relevant_count"] == 3
    assert metrics["mean_average_precision"] == pytest.approx(1 / 6)
    assert metrics["recall_at_k"] == pytest.approx(1 / 6)
    assert set(scored["document_id"]) == {"a"}  # No synthetic zero-score tail.


def test_full_candidate_limit_matches_exhaustive_scores_and_metrics(dataset):
    options = {"feature_weights": {"gst": 1.0}, "gst_min_match_length": 2}
    exhaustive, expected = evaluate_retrieval_dataset(dataset, k=2, similarity_options=options)
    shortlisted, actual = evaluate_two_stage_retrieval_dataset(dataset, 100, k=2, similarity_options=options)
    keys = ["query_id", "document_id"]
    assert exhaustive.sort_values(keys)["similarity_score"].tolist() == shortlisted.sort_values(keys)["similarity_score"].tolist()
    assert all(actual[key] == value for key, value in expected.items())
    assert actual["candidate_recall"] == actual["candidate_recall_micro"] == 1.0


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_candidate_count_is_explicit_and_positive(dataset, value):
    with pytest.raises(ValueError, match="positive integer"):
        select_retrieval_candidates(dataset, value)


def test_candidate_diagnostics_reject_unknown_and_duplicate_ids(dataset):
    for rows in ([{"query_id": "unknown", "document_id": "a"}],
                 [{"query_id": "q1", "document_id": "a"}] * 2):
        with pytest.raises(ValueError):
            candidate_recall_diagnostics(dataset, rows)


def test_self_exclusion_preserves_queries_with_no_candidates(tmp_path):
    data = write_retrieval_dataset(tmp_path / "self", files=pd.DataFrame([{"file_id": "a", "text": "same"}]),
        queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame([{"document_id": "d", "file_id": "a"}]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "d", "relevance": 0}]))
    scored, metrics = evaluate_two_stage_retrieval_dataset(data, 1, exclude_self_matches=True)
    assert scored.empty
    assert metrics["query_count"] == 1 and metrics["result_count"] == 0
    assert metrics["first_stage_comparisons"] == metrics["reranker_comparisons"] == 0
    assert metrics["candidate_recall"] is None and metrics["mean_average_precision"] == 0.0
    data.qrels.loc[:, "relevance"] = 1
    with pytest.raises(ValueError, match="positive self"):
        select_retrieval_candidates(data, 1, exclude_self_matches=True)


def test_nonfinite_stage_scores_fail_instead_of_becoming_missing_candidates(dataset):
    with pytest.raises(ValueError, match="finite scores"):
        select_retrieval_candidates(dataset, 1, scorer=lambda *args: float("nan"))


def test_cli_writes_offline_pipeline_scores_and_full_diagnostics(dataset, tmp_path):
    scores, report = tmp_path / "scores.csv", tmp_path / "metrics.json"
    result = CliRunner().invoke(main, ["evaluate-two-stage-retrieval", str(dataset.root),
        "--candidate-count", "2", "--candidate-feature-weight", "lcs=1", "--feature-weight", "gst=1",
        "--scores-out", str(scores), "--metrics-out", str(report)])
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert len(pd.read_csv(scores)) == 6
    assert payload["metrics"]["retrieval_mode"] == "two_stage"
    assert len(payload["retrieval"]["candidate_diagnostics"]["per_query"]) == 3
    assert payload["retrieval"]["candidate_similarity_options"]["feature_weights"] == {"lcs": 1}


def test_streaming_multivector_stages_keep_roles_without_a_full_score_matrix(dataset, monkeypatch):
    import numpy as np
    from matheel import evaluation
    monkeypatch.setattr(evaluation._similarity, "load_backend_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation._similarity, "build_document_embeddings", lambda model, texts, **options: [
        np.array([[100.0 if options["is_query"] else 200.0, len(text)]]) for text in texts
    ])
    monkeypatch.setattr(evaluation, "_add_multivector_scores", lambda *args, **kwargs: pytest.fail("No dense Q*D matrix"))
    calls = []
    def semantic(left, right, **kwargs):
        calls.append((left[0, 0], right[0, 0]))
        return 0.75
    monkeypatch.setattr(evaluation._similarity, "semantic_similarity", semantic)
    options = {"feature_weights": {"semantic": 1.0}, "vector_backend": "multivector"}
    scored, metrics = evaluate_two_stage_retrieval_dataset(
        dataset, 1, candidate_similarity_options=options, similarity_options=options,
    )
    assert calls == [(100.0, 200.0)] * 15
    assert len(scored) == 3 and metrics["first_stage_comparisons"] == 12


def test_custom_leaderboard_records_pipeline_identity_and_diagnostics(dataset):
    from matheel.benchmark_protocol import method_configuration_id
    from matheel.leaderboard import normalize_leaderboard_manifest, run_leaderboard
    method = {"name": "LCS then GST", "feature_weights": {"gst": 1.0},
              "retrieval_options": {"candidate_count": 2, "candidate_similarity_options": {"feature_weights": {"lcs": 1.0}}}}
    manifest = {"datasets": [{"name": "example", "task": "retrieval", "path": str(dataset.root)}], "algorithms": [method]}
    report, _ = run_leaderboard(manifest)
    row = report["per_dataset"].iloc[0]
    assert row["retrieval_mode"] == "two_stage" and row["reranker_comparisons"] == 6
    assert row["candidate_recall"] >= 0
    changed = {**method, "retrieval_options": {"candidate_count": 1}}
    first = normalize_leaderboard_manifest(manifest)["algorithms"][0]
    second = normalize_leaderboard_manifest({**manifest, "algorithms": [changed]})["algorithms"][0]
    assert method_configuration_id(first) != method_configuration_id(second)
