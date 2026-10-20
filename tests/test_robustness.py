import ast
from dataclasses import asdict
import json

import pytest

from matheel.robustness import (
    controlled_robustness_cases, python_robustness_case, run_robustness_experiment,
)
from matheel.similarity import calculate_similarity


def test_controlled_cases_are_repeatable_and_have_explicit_validation():
    first, second = controlled_robustness_cases(23), controlled_robustness_cases(23)
    assert [asdict(case) for case in first] == [asdict(case) for case in second]
    assert {case.transformation for case in first} == {
        "identity", "formatting", "comments", "identifier_renaming", "partial_copy", "shared_starter_code",
    }
    assert all(case.provenance["validation"] for case in first)
    assert all(len(case.provenance["source_sha256"]["candidate"]) == 64 for case in first)
    renamed = next(case for case in first if case.transformation == "identifier_renaming")
    other = next(case for case in controlled_robustness_cases(24)
                 if case.transformation == "identifier_renaming")
    assert renamed.candidate != other.candidate
    assert renamed.case_id != other.case_id


@pytest.mark.parametrize("transformation", ["formatting", "comments"])
def test_ast_checked_transforms_preserve_string_literals_without_executing_source(transformation):
    source = 'message = "# keep this string"\nraise RuntimeError("must not execute")\n'
    case = python_robustness_case(source, transformation)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(case.candidate))
    assert case.provenance["validation"]["kind"] == "python_ast_equal"
    assert "line numbers can differ" in case.provenance["validation"]["scope"]


def test_renaming_cannot_claim_semantic_safety_without_named_caller_assertions():
    source = "def value():\n    count = 2\n    return count\n"
    with pytest.raises(ValueError, match="validator"):
        python_robustness_case(source, "identifier_renaming", identifier_map={"count": "total"})
    with pytest.raises(ValueError, match="did not pass"):
        python_robustness_case(source, "identifier_renaming", identifier_map={"count": "total"},
                               validator=lambda _before, _after: False, validation_name="behavior check")


def test_renaming_binding_collision_is_rejected_by_explicit_fixture_validation():
    source = "def value():\n    left = 2\n    right = 3\n    return left + right\n"

    def assert_value(before, after):
        values = []
        for code in (before, after):
            namespace = {}
            exec(code, namespace)
            values.append(namespace["value"]())
        return values == [5, 5]

    with pytest.raises(ValueError, match="did not pass"):
        python_robustness_case(source, "identifier_renaming", identifier_map={"left": "right"},
                               validator=assert_value, validation_name="value() == 5")


def test_partial_copy_and_starter_are_content_diagnostics_without_labels():
    cases = {case.transformation: case for case in controlled_robustness_cases()}
    partial, starter = cases["partial_copy"], cases["shared_starter_code"]
    assert 0 < partial.provenance["parameters"]["copied_source_fraction"] < 1
    assert partial.baseline_candidate in partial.candidate
    assert partial.reference == partial.baseline_reference
    assert starter.reference.endswith(starter.baseline_reference)
    assert starter.candidate.endswith(starter.baseline_candidate)
    assert "label" not in asdict(partial) and "label" not in asdict(starter)


def test_existing_basic_preprocessing_is_invariant_to_added_comments():
    comment_case = next(case for case in controlled_robustness_cases() if case.transformation == "comments")
    report, _ = run_robustness_experiment([
        {"run_name": "raw", "feature_weights": {"levenshtein": 1}},
        {"run_name": "basic", "feature_weights": {"levenshtein": 1},
         "preprocess_mode": "basic", "code_language": "python"},
    ], cases=[comment_case])
    raw, basic = report["scores"]
    assert raw["score_delta"] < 0
    assert basic["baseline_score"] == basic["transformed_score"] == 1
    assert basic["score_delta"] == 0


def test_changed_case_provenance_cannot_be_silently_replayed():
    case = controlled_robustness_cases()[0]
    case.provenance["source_sha256"]["candidate"] = "0" * 64
    with pytest.raises(ValueError, match="source hashes"):
        run_robustness_experiment([{"feature_weights": {"levenshtein": 1}}], cases=[case])


def test_robustness_scores_match_existing_scorer_and_export_full_provenance(tmp_path):
    run = {"run_name": "lexical", "feature_weights": {"levenshtein": 1}}
    cases = controlled_robustness_cases(9)
    report, artifacts = run_robustness_experiment([run], cases=cases, output_dir=tmp_path)
    by_id = {case.case_id: case for case in cases}
    assert report["diagnostic_only"]
    for row in report["scores"]:
        case = by_id[row["case_id"]]
        expected = calculate_similarity(case.reference, case.candidate, feature_weights={"levenshtein": 1})
        assert row["transformed_score"] == expected
        assert row["score_delta"] == row["transformed_score"] - row["baseline_score"]
    identity = next(row for row in report["scores"] if row["transformation"] == "identity")
    assert identity["baseline_score"] == identity["transformed_score"] == 1
    assert identity["score_delta"] == 0
    saved = json.loads((tmp_path / "robustness.json").read_text())
    assert saved["cases"] == [asdict(case) for case in cases]
    assert saved["runs"][0]["options"]["feature_weights"] == {"levenshtein": 1.0}
    assert artifacts["scores_csv"].endswith("robustness_scores.csv")


def test_failed_robustness_method_is_retained_as_null_not_perfect_or_zero_score():
    report, _ = run_robustness_experiment([
        {"run_name": "invalid", "feature_weights": {"levenshtein": 1},
         "preprocess_mode": "not_a_preprocessor"},
    ])
    assert all(row["status"] == "error" and row["error"] for row in report["scores"])
    assert all(row["baseline_score"] is None and row["transformed_score"] is None
               and row["score_delta"] is None for row in report["scores"])
    json.dumps(report, allow_nan=False)
