from click.testing import CliRunner
import pandas as pd

from matheel.cli import main


def test_compare_command_accepts_new_options(tmp_path, monkeypatch):
    archive_path = tmp_path / "codes.zip"
    archive_path.write_bytes(b"placeholder")

    captured = {}

    def fake_get_sim_list(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            [
                {
                    "file_name_1": "a.py",
                    "file_name_2": "b.py",
                    "similarity_score": 1.0,
                }
            ]
        )

    monkeypatch.setattr("matheel.cli.get_sim_list", fake_get_sim_list)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(archive_path),
            "--feature-weight",
            "semantic=0.5",
            "--feature-weight",
            "code_metric=0.5",
            "--preprocess-mode",
            "basic",
            "--chunking-method",
            "code",
            "--chunk-size",
            "50",
            "--chunk-overlap",
            "10",
            "--max-chunks",
            "4",
            "--chunk-language",
            "python",
            "--chunker-option",
            "include_line_numbers=true",
            "--chunk-aggregation",
            "max",
            "--code-metric",
            "codebleu",
            "--code-metric-weight",
            "0.4",
            "--code-language",
            "python",
            "--codebleu-component-weights",
            "0.4,0.3,0.2,0.1",
            "--crystalbleu-max-order",
            "3",
            "--crystalbleu-trivial-ngram-count",
            "40",
            "--vector-backend",
            "auto",
            "--static-vector-dim",
            "512",
            "--device",
            "cpu",
            "--no-static-vector-lowercase",
            "--no-multivector-bidirectional",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["feature_weights"] == ("semantic=0.5", "code_metric=0.5")
    assert captured["kwargs"]["preprocess_mode"] == "basic"
    assert captured["kwargs"]["chunking_method"] == "code"
    assert captured["kwargs"]["chunk_size"] == 50
    assert captured["kwargs"]["chunk_overlap"] == 10
    assert captured["kwargs"]["max_chunks"] == 4
    assert captured["kwargs"]["chunk_language"] == "python"
    assert captured["kwargs"]["chunker_options"] == ("include_line_numbers=true",)
    assert captured["kwargs"]["chunk_aggregation"] == "max"
    assert captured["kwargs"]["code_metric"] == "codebleu"
    assert captured["kwargs"]["code_metric_weight"] == 0.4
    assert captured["kwargs"]["code_language"] == "python"
    assert captured["kwargs"]["codebleu_component_weights"] == "0.4,0.3,0.2,0.1"
    assert captured["kwargs"]["crystalbleu_max_order"] == 3
    assert captured["kwargs"]["crystalbleu_trivial_ngram_count"] == 40
    assert captured["kwargs"]["vector_backend"] == "auto"
    assert captured["kwargs"]["static_vector_dim"] == 512
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["static_vector_lowercase"] is False
    assert captured["kwargs"]["multivector_bidirectional"] is False


def test_compare_command_accepts_directory_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")

    captured = {}

    def fake_get_sim_list(*args, **kwargs):
        captured["args"] = args
        return pd.DataFrame(
            [
                {
                    "file_name_1": "a.py",
                    "file_name_2": "b.py",
                    "similarity_score": 1.0,
                }
            ]
        )

    monkeypatch.setattr("matheel.cli.get_sim_list", fake_get_sim_list)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(source_dir)])

    assert result.exit_code == 0
    assert captured["args"][0] == str(source_dir)


def test_compare_suite_command_runs_config_file(tmp_path, monkeypatch):
    archive_path = tmp_path / "codes.zip"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "runs.json"
    config_path.write_text("[]", encoding="utf-8")

    captured = {}

    def fake_load_run_configs(config_file):
        captured["config_file"] = config_file
        return [{"run_name": "baseline", "options": {}}]

    def fake_run_comparison_suite(zipfile, run_configs, summary_out, details_dir, output_format):
        captured["zipfile"] = zipfile
        captured["run_configs"] = run_configs
        captured["summary_out"] = summary_out
        captured["details_dir"] = details_dir
        captured["output_format"] = output_format
        return (
            pd.DataFrame(
                [
                    {
                        "run_name": "baseline",
                        "pair_count": 1,
                        "mean_score": 0.9,
                        "median_score": 0.9,
                        "max_score": 0.9,
                        "min_score": 0.9,
                        "std_score": 0.0,
                        "top_file_1": "a.py",
                        "top_file_2": "b.py",
                        "top_score": 0.9,
                        "vector_backend": "transformer",
                        "code_metric": "none",
                        "chunking_method": "none",
                    }
                ]
            ),
            {"baseline": pd.DataFrame()},
        )

    monkeypatch.setattr("matheel.cli.load_run_configs", fake_load_run_configs)
    monkeypatch.setattr("matheel.cli.run_comparison_suite", fake_run_comparison_suite)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare-suite",
            str(archive_path),
            str(config_path),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--details-dir",
            str(tmp_path / "details"),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert captured["config_file"] == str(config_path)
    assert captured["zipfile"] == str(archive_path)
    assert captured["summary_out"].endswith("summary.json")
    assert captured["details_dir"].endswith("details")
    assert captured["output_format"] == "json"
