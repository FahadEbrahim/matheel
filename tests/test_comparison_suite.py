import json

import pandas as pd

from matheel.comparison_suite import load_run_configs, parse_run_configs, run_comparison_suite


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


def test_load_run_configs_maps_lowercase_weight_aliases(tmp_path):
    config_path = tmp_path / "runs.json"
    config_path.write_text(
        json.dumps(
            [
                {"run_name": "baseline", "ws": 0.1, "wl": 0.2, "wj": 0.7},
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
    runs = parse_run_configs('[{"run_name":"baseline","model":"demo-model","ws":0.2}]')

    assert runs[0]["run_name"] == "baseline"
    assert runs[0]["options"]["model_name"] == "demo-model"
    assert runs[0]["options"]["feature_weights"] == {"semantic": 0.2}


def test_run_comparison_suite_writes_summary_and_details(tmp_path, monkeypatch):
    def fake_get_sim_list(zipped_file, **kwargs):
        if kwargs["model_name"] == "strong-model":
            return pd.DataFrame(
                [
                    {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 0.95},
                    {"file_name_1": "a.py", "file_name_2": "c.py", "similarity_score": 0.60},
                ]
            )
        return pd.DataFrame(
            [
                {"file_name_1": "a.py", "file_name_2": "b.py", "similarity_score": 0.40},
            ]
        )

    monkeypatch.setattr("matheel.comparison_suite.get_sim_list", fake_get_sim_list)

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
