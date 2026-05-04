import pandas as pd

import matheel
from matheel.reports import benchmark_report_html, write_benchmark_report


def test_report_helpers_are_exported_from_package_root():
    assert matheel.benchmark_report_html is benchmark_report_html
    assert matheel.write_benchmark_report is write_benchmark_report


def test_benchmark_report_html_escapes_content_and_sanitizes_links(tmp_path):
    report = _synthetic_report()
    output = benchmark_report_html(
        report,
        title="<unsafe>",
        artifact_links={"dataset_map": tmp_path / "nested" / "map.html"},
    )

    assert "<unsafe>" not in output
    assert "&lt;unsafe&gt;" in output
    assert "<dataset>" not in output
    assert "&lt;dataset&gt;" in output
    assert "<algorithm>" not in output
    assert str(tmp_path) not in output
    assert "map.html" in output
    assert "Aggregate Ranking" in output
    assert "Dataset Cards" in output
    assert "Algorithm Cards" in output


def test_write_benchmark_report_writes_static_html(tmp_path):
    output_path = tmp_path / "reports" / "leaderboard.html"

    written = write_benchmark_report(_synthetic_report(), output_path)

    assert written == output_path
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "Per-Dataset Ranking" in text


def _synthetic_report():
    return {
        "metadata": {"name": "<benchmark>", "seed": 7},
        "manifest": {"name": "<benchmark>"},
        "cards": {
            "datasets": [
                {
                    "card_type": "dataset",
                    "name": "<dataset>",
                    "task_family": "pair",
                    "dataset_kind": "pair_classification",
                    "license": "synthetic",
                    "counts": {"pairs": 2},
                    "fingerprint": {"sha256": "a" * 64},
                    "source": {"identifier": "pairs"},
                }
            ],
            "algorithms": [
                {
                    "card_type": "algorithm",
                    "name": "<algorithm>",
                    "algorithm_kind": "custom",
                    "algorithm_path_name": "algo.py",
                    "fingerprint": {"sha256": "b" * 64},
                }
            ],
        },
        "aggregate": pd.DataFrame(
            [
                {
                    "task_family": "pair",
                    "algorithm_name": "<algorithm>",
                    "metric": "f1",
                    "mean_score": 1.0,
                    "rank": 1,
                }
            ]
        ),
        "per_dataset": pd.DataFrame(
            [
                {
                    "task_family": "pair",
                    "dataset_name": "<dataset>",
                    "algorithm_name": "<algorithm>",
                    "metric": "f1",
                    "score": 1.0,
                    "rank": 1,
                }
            ]
        ),
    }
