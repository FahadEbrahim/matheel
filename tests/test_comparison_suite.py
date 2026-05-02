import json

import pandas as pd
import pytest

from matheel.comparison_suite import (
    load_run_configs,
    normalize_run_config,
    parse_run_configs,
    run_comparison_suite,
    slugify_run_name,
)


def test_load_run_configs_accepts_runs_wrapper(tmp_path):
    config_path = tmp_path / "runs.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_name": "baseline", "model": "demo-model", "num": 5},
                ]
            }
        ),
        encoding="utf-8",
    )

    runs = load_run_configs(config_path)

    assert runs[0]["run_name"] == "baseline"
    assert runs[0]["options"]["model_name"] == "demo-model"
    assert runs[0]["options"]["number_results"] == 5


def test_load_run_configs_supports_feature_weights_dict(tmp_path):
    config_path = tmp_path / "runs.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "run_name": "baseline",
                    "feature_weights": {"semantic": 0.1, "levenshtein": 0.2, "jaro_winkler": 0.7},
                },
            ]
        ),
        encoding="utf-8",
    )

    runs = load_run_configs(config_path)

    assert runs[0]["options"]["feature_weights"] == {
        "semantic": 0.1,
        "levenshtein": 0.2,
        "jaro_winkler": 0.7,
    }


def test_parse_run_configs_accepts_json_text():
    runs = parse_run_configs(
        '[{"run_name":"baseline","model":"demo-model","feature_weights":"semantic=0.2,code_metric=0.8"}]'
    )

    assert runs[0]["run_name"] == "baseline"
    assert runs[0]["options"]["model_name"] == "demo-model"
    assert runs[0]["options"]["feature_weights"] == {"semantic": 0.2, "code_metric": 0.8}


def test_normalize_run_config_does_not_leak_nested_run_name():
    run = normalize_run_config(
        {
            "run_name": "outer",
            "options": {
                "run_name": "inner",
                "feature_weights": {"levenshtein": 1.0},
            },
        }
    )

    assert run["run_name"] == "outer"
    assert "run_name" not in run["options"]


def test_parse_run_configs_rejects_legacy_weight_keys():
    with pytest.raises(ValueError, match="Legacy weight keys"):
        parse_run_configs(
            '[{"run_name":"legacy","options":{"ws":0.7,"wl":0.3}}]'
        )


def test_slugify_run_name_removes_path_separators():
    assert slugify_run_name("../baseline/strong") == "baseline_strong"
    assert slugify_run_name("...") == "run"


def test_run_comparison_suite_writes_summary_and_details(tmp_path, monkeypatch):
    def fake_get_sim_list(zipped_file, **kwargs):
        if kwargs["model_name"] == "strong-model":
            return pd.DataFrame(
                [
                    {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 0.9876543},
                    {"file_name_1": "a.py", "file_name_2": "c.py", "similarity_score": 0.1234567},
                ]
            )
        return pd.DataFrame(
            [
                {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 0.4000001},
            ]
        )

    monkeypatch.setattr("matheel.comparison_suite.get_sim_list", fake_get_sim_list)
    times = iter([10.0, 11.23456, 20.0, 20.5])
    monkeypatch.setattr("matheel.comparison_suite.perf_counter", lambda: next(times))

    summary_path = tmp_path / "summary.csv"
    details_dir = tmp_path / "details"
    summary, result_frames = run_comparison_suite(
        "codes.zip",
        [
            {"run_name": "strong", "model_name": "strong-model", "number_results": 10},
            {"run_name": "weak", "model_name": "weak-model", "number_results": 10},
        ],
        summary_out=summary_path,
        details_dir=details_dir,
    )

    assert list(summary["run_name"]) == ["strong", "weak"]
    assert summary_path.exists()
    assert (details_dir / "strong.csv").exists()
    assert (details_dir / "weak.csv").exists()
    assert set(result_frames.keys()) == {"strong", "weak"}
    assert summary.loc[0, "mean_score"] == 0.5556
    assert summary.loc[0, "top_score"] == 0.9877
    assert summary.loc[0, "elapsed_seconds"] == 1.2346
    assert summary.loc[0, "feature_set"] == "levenshtein,semantic"
    assert summary.loc[0, "vector_backend"] == "auto"
    assert summary.loc[0, "code_metric"] == "none"
    assert summary.loc[0, "chunking_method"] == "none"
    assert result_frames["strong"].attrs["elapsed_seconds"] == 1.2346
    assert result_frames["strong"].iloc[0]["similarity_score"] == 0.9877

    summary_text = summary_path.read_text(encoding="utf-8")
    strong_details_text = (details_dir / "strong.csv").read_text(encoding="utf-8")
    assert "elapsed_seconds" in summary_text
    assert "feature_set" in summary_text
    assert "1.2346" in summary_text
    assert "0.9876543" not in summary_text
    assert "0.9877" in summary_text
    assert "0.9876543" not in strong_details_text
    assert "0.9877" in strong_details_text


def test_run_comparison_suite_accepts_already_normalized_configs(monkeypatch):
    captured_options = []

    def fake_get_sim_list(zipped_file, **kwargs):
        captured_options.append(kwargs)
        return pd.DataFrame(
            [
                {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 1.0},
            ]
        )

    monkeypatch.setattr("matheel.comparison_suite.get_sim_list", fake_get_sim_list)
    normalized = normalize_run_config(
        {
            "run_name": "outer",
            "options": {
                "run_name": "inner",
                "feature_weights": {"levenshtein": 1.0},
            },
        }
    )

    summary, _ = run_comparison_suite("codes.zip", [normalized])

    assert list(summary["run_name"]) == ["outer"]
    assert captured_options
    assert "run_name" not in captured_options[0]


def test_run_comparison_suite_reports_progress_events(monkeypatch):
    def fake_get_sim_list(zipped_file, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback({"stage": "compare_pairs", "current": 1, "total": 1})
        return pd.DataFrame(
            [
                {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 1.0},
            ]
        )

    monkeypatch.setattr("matheel.comparison_suite.get_sim_list", fake_get_sim_list)
    events = []

    run_comparison_suite(
        "codes.zip",
        [{"run_name": "baseline", "feature_weights": {"levenshtein": 1.0}}],
        progress_callback=events.append,
    )

    suite_events = [event for event in events if event["stage"] == "suite_runs"]
    pair_events = [event for event in events if event["stage"] == "compare_pairs"]
    assert suite_events[-1]["current"] == 1
    assert suite_events[-1]["total"] == 1
    assert pair_events[-1]["run_name"] == "baseline"
