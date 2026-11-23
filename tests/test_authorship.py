import csv
import json
from collections import defaultdict

import pytest

from matheel.authorship import (
    benchmark_authorship,
    fit_authorship_classifier,
    load_authorship_dataset,
    prepare_aigcodeset,
    validate_authorship_report,
    write_authorship_dataset,
    write_authorship_report,
)


def _row(problem, generator, status, code):
    return {
        "problem_id": f"p{problem:03d}",
        "submission_id": f"s{problem:03d}",
        "LLM": generator,
        "status_in_folder": status,
        "code": code,
        "label": int(generator != "Human"),
    }


def _write_tables(tmp_path, human, generated):
    paths = [tmp_path / name for name in ("human.csv", "generated.csv")]
    for path, rows in zip(paths, (human, generated), strict=True):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths


def _tables(tmp_path):
    human, generated = [], []
    for problem in range(12):
        human.append(_row(problem, "Human", "Accepted", f"original_{problem} = {problem}"))
        human.append(_row(problem, "Human", "Runtime", f"runtime_{problem} = {problem}"))
        generated.append(_row(problem, "Model", "Generate", f"generated_{problem} = {problem}"))
        generated.append(_row(problem, "Model", "Runtime", f"repair_{problem} = {problem}"))
    human.append(_row(0, "Human", "Wrong", "known_conflict = 31"))
    generated.append(_row(0, "Model", "Wrong", "known_conflict = 31"))
    human.append(dict(human[0]))
    return _write_tables(tmp_path, human, generated)


def test_authorship_source_audit_and_snapshot(tmp_path):
    human, generated = _tables(tmp_path)
    dataset = prepare_aigcodeset(human, generated)
    assert dataset.audit["source_verified"] is False
    assert dataset.audit["exact_origin_conflict_texts"] == 1
    assert dataset.audit["exclusion_counts"] == {
        "duplicate_within_problem_and_cohort": 1, "exact_origin_conflict": 2,
    }
    split_by_problem = defaultdict(set)
    for row in dataset.samples:
        split_by_problem[row["problem_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_problem.values())
    restored = load_authorship_dataset(write_authorship_dataset(dataset, tmp_path / "private.json"))
    assert restored == dataset
    with pytest.raises(ValueError, match="source size mismatch"):
        prepare_aigcodeset(human, generated, verify_source=True)


def test_near_duplicate_source_problems_remain_in_same_split(tmp_path):
    human, generated = [], []
    for problem in range(8):
        if problem < 2:
            code = f"def f_{problem}(x, y, z):\n    return x * y + y * z + x * z\n"
        else:
            code = f"plain_{problem} = {problem}"
        human.append(_row(problem, "Human", "Accepted", code))
        generated.append(_row(problem, "Model", "Generate", f"generated_{problem} = {problem}"))
    paths = _write_tables(tmp_path, human, generated)
    dataset = prepare_aigcodeset(*paths)
    assert dataset.audit["near_duplicate_audit"]["cross_problem_near_duplicate_pairs"] == 1
    assert dataset.audit["retained_independent_group_count"] == 7
    rows = [row for row in dataset.samples if row["problem_id"] in {"p000", "p001"}]
    assert {row["group_id"] for row in rows} == {"p000"}
    assert len({row["split"] for row in rows}) == 1


def test_benchmark_calibrates_on_validation_and_replays_source_free_artifacts(tmp_path):
    dataset = prepare_aigcodeset(*_tables(tmp_path))
    prior = fit_authorship_classifier(dataset, model="class_prior")
    assert prior.score("standalone source") == 0.5
    report = benchmark_authorship(dataset, models="class_prior")
    result = report["results"][0]
    assert result["test"]["auroc"] == 0.5
    assert result["test"]["average_precision"] == 0.5
    assert result["test"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["validation"]["false_positive_rate"] == 0
    assert result["test"]["false_positive_rate"] == 0
    assert result["repair_slices"]["runtime_repair"]["count"] == 4
    assert result["repair_slices"]["wrong_answer_repair"]["count"] == 0
    assert {row["split"] for row in report["predictions"]} == {"validation", "test"}
    artifacts = write_authorship_report(report, tmp_path / "benchmark")
    assert validate_authorship_report(artifacts["results"])["results"] == report["results"]
    assert "generated_" not in artifacts["predictions"].read_text()
    tampered = json.loads(artifacts["results"].read_text())
    tampered["results"][0]["test"]["auroc"] = 1.0
    artifacts["results"].write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="disagrees with its prediction artifact"):
        validate_authorship_report(artifacts["results"])


def test_char_tfidf_fits_and_scores_a_single_snippet(tmp_path):
    pytest.importorskip("sklearn")
    dataset = prepare_aigcodeset(*_tables(tmp_path))
    classifier = fit_authorship_classifier(dataset)
    assert classifier.score("generated_52 = 52") > classifier.score("original_53 = 53")
    report = benchmark_authorship(dataset)
    assert {result["model"] for result in report["results"]} == {"class_prior", "char_tfidf_logistic"}
    assert report["results"][1]["validation"]["false_positive_rate"] <= 0.05
