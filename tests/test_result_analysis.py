import json

import pandas as pd
import numpy as np
import pytest
from click.testing import CliRunner

from matheel.cli import main
from matheel.result_analysis import analyze_scored_results, write_result_analysis
from matheel.result_analysis import _metrics, _pair_bootstrap_context, _weighted_pair_metrics


@pytest.fixture
def pairs():
    return pd.DataFrame([
        {"left_id": f"a{i}", "right_id": f"b{i}", "label": i % 2,
         "similarity_score": [.2, .9, .7, .3, .1, .8][i], "assignment": str(i // 2)}
        for i in range(6)
    ])


def test_pair_errors_and_grouped_paired_bootstrap_are_deterministic(pairs):
    report = analyze_scored_results(pairs, reference_rows=pairs, group_column="assignment", rounds=30)
    assert report == analyze_scored_results(pairs, reference_rows=pairs, group_column="assignment", rounds=30)
    assert report["group_count"] == 3
    assert {row["outcome"] for row in report["errors"]} == {
        "true_positive", "true_negative", "false_positive", "false_negative"}
    for row in report["metrics"]:
        assert row["delta"] == 0
        assert row["delta_ci_lower"] == row["delta_ci_upper"] == 0
        assert row["ci_lower"] <= row["ci_upper"]
    assert not any(row["decision_disagreement"] for row in report["errors"])
    assert len(report["slices"]) == 3


def test_shared_source_pairs_stay_in_the_same_bootstrap_component(pairs):
    pairs["left_id"] = "shared"
    report = analyze_scored_results(pairs, rounds=10)
    assert report["group_count"] == 1
    assert all(row["ci_lower"] is None and row["uncertainty_status"].startswith("N/A")
               for row in report["metrics"])


def test_reference_cannot_change_candidate_universe_or_ground_truth(pairs):
    with pytest.raises(ValueError, match="identical candidates"):
        analyze_scored_results(pairs, reference_rows=pairs.iloc[:-1])
    reference = pairs.copy()
    reference.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="ground truth"):
        analyze_scored_results(pairs, reference_rows=reference)
    pairs.loc[0, "similarity_score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analyze_scored_results(pairs)


def test_previously_exported_paired_errors_can_be_analyzed_again(pairs):
    first = analyze_scored_results(pairs, reference_rows=pairs, rounds=5)
    second = analyze_scored_results(pd.DataFrame(first["errors"]), reference_rows=pairs, rounds=5)
    assert first["metrics"] == second["metrics"]
    assert all(row["score_delta"] == 0 for row in second["errors"])


def test_single_class_has_no_fabricated_auroc(pairs):
    pairs["label"] = 1
    report = analyze_scored_results(pairs, rounds=10)
    assert "auroc" not in {row["metric"] for row in report["metrics"]}


def test_repeated_pair_observations_keep_their_original_weight(pairs):
    repeated = pd.concat([pairs, pairs.iloc[:1]], ignore_index=True)
    report = analyze_scored_results(repeated, reference_rows=repeated, rounds=10)
    assert report["rows"] == 7
    assert len(report["errors"]) == 7
    assert any(row["pair_occurrence"] == 1 for row in report["errors"])


def test_weighted_bootstrap_matches_public_metrics_with_ties_and_repeated_rows():
    rng = np.random.default_rng(7)
    for _ in range(25):
        frame = pd.DataFrame({"label": rng.integers(0, 2, 30),
                              "similarity_score": rng.integers(0, 5, 30) / 4})
        rows = rng.integers(0, len(frame), len(frame))
        weights = np.bincount(rows, minlength=len(frame))
        expected = _metrics(frame.iloc[rows], "pair", .5, 10, "similarity_score")
        actual = _weighted_pair_metrics(_pair_bootstrap_context(frame, .5, "similarity_score"), weights)
        assert actual == pytest.approx(expected)


def test_retrieval_uncertainty_resamples_queries_and_includes_missed_top_k():
    frame = pd.DataFrame([
        {"query_id": "q1", "document_id": "a", "relevance": 0, "similarity_score": .9},
        {"query_id": "q1", "document_id": "b", "relevance": 1, "similarity_score": .8},
        {"query_id": "q2", "document_id": "a", "relevance": 1, "similarity_score": .7},
        {"query_id": "q2", "document_id": "b", "relevance": 0, "similarity_score": .1},
    ])
    report = analyze_scored_results(frame, reference_rows=frame.iloc[::-1], rounds=30, k=1)
    metrics = {row["metric"]: row for row in report["metrics"]}
    assert metrics["mean_average_precision"]["score"] == .75
    assert report["group_count"] == 2
    assert len(report["per_query"]) == 2
    assert metrics["mean_average_precision"]["delta_ci_upper"] == 0
    assert "relevant_below_k" in {row["outcome"] for row in report["errors"]}
    repeated = analyze_scored_results(pd.DataFrame(report["errors"]), reference_rows=frame,
                                      rounds=30, k=1)
    assert repeated["metrics"] == report["metrics"]


def test_exports_escape_html_and_do_not_contain_source_code(pairs, tmp_path):
    pairs["text"] = "PRIVATE SOURCE"
    pairs["assignment"] = "<script>alert(1)</script>"
    report = analyze_scored_results(pairs, rounds=5)
    artifacts = write_result_analysis(report, tmp_path)
    html = artifacts["html"].read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "PRIVATE SOURCE" not in artifacts["json"].read_text()
    assert "PRIVATE SOURCE" not in artifacts["errors"].read_text()
    with pytest.raises(ValueError, match="filename stem"):
        write_result_analysis(report, tmp_path, basename="../escape")


def test_missing_optional_metadata_exports_as_json_null(pairs, tmp_path):
    pairs["language"] = ["python", None, "python", None, "python", "python"]
    report = analyze_scored_results(pairs, rounds=5)
    payload = json.loads(write_result_analysis(report, tmp_path)["json"].read_text())
    assert sum(row["language"] is None for row in payload["errors"]) == 2


def test_custom_score_column_survives_exports_and_reanalysis(pairs, tmp_path):
    pairs = pairs.rename(columns={"similarity_score": "detector_score"})
    report = analyze_scored_results(pairs, score_column="detector_score", rounds=5)
    artifacts = write_result_analysis(report, tmp_path)
    payload = json.loads(artifacts["json"].read_text())
    assert payload["score_column"] == "detector_score"
    assert [row["detector_score"] for row in payload["errors"]] == pairs["detector_score"].tolist()
    exported = pd.read_csv(artifacts["errors"])
    repeated = analyze_scored_results(exported, score_column=payload["score_column"], rounds=5)
    assert repeated["metrics"] == report["metrics"]


def test_cli_offline_analysis(pairs, tmp_path):
    scores = tmp_path / "scores.csv"
    pairs.to_csv(scores, index=False)
    target = tmp_path / "analysis"
    result = CliRunner().invoke(main, ["analyze-results", str(scores), "--rounds", "5",
                                     "--output-dir", str(target)])
    assert result.exit_code == 0, result.output
    report = json.loads((target / "result_analysis.json").read_text())
    assert report["rows"] == 6
    result = CliRunner().invoke(main, ["analyze-results", str(scores), "--threshold", "nan"])
    assert result.exit_code != 0


def test_dataset_enrichment_adds_language_and_length_without_exporting_source(pairs, tmp_path):
    from matheel.datasets import write_pair_dataset

    files = [{"file_id": key, "suffix": ".py", "text": "print('private')\n" * (1000 if key == "b1" else 1)}
             for key in set(pairs["left_id"]) | set(pairs["right_id"])]
    root = tmp_path / "dataset"
    write_pair_dataset(root, files=pd.DataFrame(files), pairs=pairs.drop(columns=["similarity_score"]),
                       metadata={"task_type": "plagiarism", "languages": ["python"]})
    report = analyze_scored_results(pairs, dataset_root=root, rounds=5)
    assert {row["language"] for row in report["errors"]} == {"python"}
    assert {row["length_bucket"] for row in report["errors"]} == {"short (<=1000 chars)", "long (>10000 chars)"}
    assert "private" not in write_result_analysis(report, tmp_path / "analysis")["json"].read_text()
