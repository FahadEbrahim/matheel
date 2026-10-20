import json

from click.testing import CliRunner

from matheel.cli import main


def test_default_robustness_cli_writes_usable_scores_and_provenance(tmp_path):
    result = CliRunner().invoke(main, ["experiments", "robustness", "--seed", "12",
                                       "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["scored_rows"] == 18 and summary["error_rows"] == 0
    report = json.loads((tmp_path / "robustness.json").read_text())
    assert report["cases"][0]["provenance"]["seed"] == 12
    assert (tmp_path / "robustness_scores.csv").is_file()


def test_source_robustness_cli_only_parses_supplied_code_without_executing_it(tmp_path):
    source = tmp_path / "untrusted.py"
    source.write_text("raise RuntimeError('source must not be executed')\n")
    output = tmp_path / "diagnostics"
    result = CliRunner().invoke(main, ["experiments", "robustness", "--source", str(source),
                                       "--output-dir", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads((output / "robustness.json").read_text())
    assert {case["transformation"] for case in report["cases"]} == {"formatting", "comments"}
    assert len(report["scores"]) == 6


def test_preflight_cli_is_offline_without_explicit_model_inspection(tmp_path, monkeypatch):
    from matheel import similarity

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preflight must not load a model unless explicitly requested")

    monkeypatch.setattr(similarity, "load_backend_model", forbidden)
    source, options, output = (tmp_path / name for name in ("source.py", "options.json", "out.json"))
    source.write_text("print('écho')\n")
    options.write_text(json.dumps({"vector_backend": "sentence_transformers", "max_token_length": 2}))
    result = CliRunner().invoke(main, ["experiments", "preflight", str(source),
                                       "--options-file", str(options), "--output", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["unknown_truncation_chunks"] == 1
    assert report["chunks"][0]["input_tokens"] is None


def test_long_file_cli_uses_declared_existing_scorer_configuration(tmp_path):
    left, right, config = (tmp_path / name for name in ("left.py", "right.py", "runs.json"))
    left.write_text("left = 1\n")
    right.write_text("right = 9\n")
    config.write_text(json.dumps({"runs": [
        {"run_name": "raw", "feature_weights": {"semantic": 1}, "vector_backend": "static_hash"},
    ]}))
    result = CliRunner().invoke(main, ["experiments", "long-files", str(left), str(right),
                                       "--config", str(config), "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["scored_rows"] == 1


def test_scoring_failure_returns_nonzero_but_keeps_null_error_rows(tmp_path):
    config = tmp_path / "runs.json"
    config.write_text(json.dumps({"runs": [
        {"run_name": "failed", "feature_weights": {"levenshtein": 1}, "preprocess_mode": "invalid"},
    ]}))
    result = CliRunner().invoke(main, ["experiments", "robustness", "--config", str(config),
                                       "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 1
    report = json.loads((tmp_path / "out" / "robustness.json").read_text())
    assert all(row["status"] == "error" and row["score_delta"] is None for row in report["scores"])


def test_explicit_empty_run_config_is_not_silently_replaced_by_defaults(tmp_path):
    config = tmp_path / "empty.json"
    config.write_text("[]")
    result = CliRunner().invoke(main, ["experiments", "robustness", "--config", str(config),
                                       "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "At least one declared scoring run" in result.output
