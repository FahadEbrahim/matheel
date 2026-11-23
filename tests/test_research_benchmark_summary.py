import hashlib
import json

import pytest

from scripts.build_research_benchmark_summary import DETECTOR_REPORT_PATH, OUTPUT_PATH, build_summary, main


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_research_index_has_portable_sources_and_projects_only_measured_metrics(tmp_path):
    scope = "benchmarks/research-080/similarity/poj104-test/results.json"
    _json(tmp_path / scope, {
        "benchmark_schema_version": 1,
        "dataset": {"adapter": "poj104_retrieval", "split": "test", "official_split_dimensions": True},
        "task_family": "retrieval",
        "results": [{"name": "character_tfidf_cosine", "metrics": {
            "query_count": 10, "mean_average_precision_at_r": 0.75, "mean_reciprocal_rank": 0.9,
        }}],
    })
    _json(tmp_path / DETECTOR_REPORT_PATH, {"schema_version": 1, "cases": [
        {"id": "nicad-java-smoke", "detector": "NiCad", "scope": "Synthetic Java",
         "status": "verified", "metrics": {"eligible": 2, "matched": 1, "recall": 0.5}},
        {"id": "unavailable", "status": "missing_corpus"},
    ]})
    summary = build_summary(tmp_path)
    assert summary["schema_version"] == 1
    assert summary["release"] == "0.8.0"
    assert summary["sources"] == [
        {
            "path": name,
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            "rows": 1,
        }
        for name in (DETECTOR_REPORT_PATH.as_posix(), scope)
    ]
    clone, retrieval = summary["rows"]
    assert (clone["metric"], clone["value"], clone["evaluated"], clone["matched"]) == ("Clone recall", 0.5, 2, 1)
    assert (retrieval["dataset"], retrieval["metric"], retrieval["value"], retrieval["evaluated"]) == (
        "POJ-104", "MAP@R", 0.75, 10,
    )
    assert str(tmp_path) not in json.dumps(summary)
    main(["--root", str(tmp_path)])
    assert json.loads((tmp_path / OUTPUT_PATH).read_text()) == summary
    main(["--root", str(tmp_path), "--check"])
    path = tmp_path / scope
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SystemExit) as failure:
        main(["--root", str(tmp_path), "--check"])
    assert failure.value.code == 1
