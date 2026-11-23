import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from examples.evaluation.research_benchmarks_demo import run_research_benchmarks
from matheel.cli import main


@pytest.fixture
def sample_benchmarks(tmp_path):
    return run_research_benchmarks(tmp_path / "sample")


def test_clone_cli_reports_types_and_bigcloneeval_export(sample_benchmarks, tmp_path):
    runner = CliRunner()
    findings = sample_benchmarks / "findings.csv"
    references = sample_benchmarks / "references.csv"
    report = tmp_path / "report.json"
    result = runner.invoke(main, ["evaluate-clones", str(references), str(findings), "--output", str(report)])
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["overall"] == {"matched": 1, "eligible": 2, "recall": 0.5}
    assert next(row for row in payload["strata"] if row["clone_type"] == "type1")["recall"] == 1
    assert set(payload["input_sha256"]) == {"references", "findings"}

    rejected = runner.invoke(main, ["export-bigcloneeval", str(findings), str(findings)])
    assert rejected.exit_code == 1
    assert "distinct paths" in rejected.output

    exported = tmp_path / "bigcloneeval.csv"
    source_map = tmp_path / "source_map.json"
    source_map.write_text(json.dumps({
        "src/Original.java": "selected/Original.java", "src/Second.java": "selected/Second.java",
    }))
    result = runner.invoke(main, [
        "export-bigcloneeval", str(findings), str(exported), "--source-map", str(source_map),
    ])
    assert result.exit_code == 0, result.output
    assert len(exported.read_text().strip().split(",")) == 8


def test_authorship_cli_prepares_source_free_replay(sample_benchmarks, tmp_path):
    runner = CliRunner()
    normalized = tmp_path / "snapshot.json"
    output = tmp_path / "origin"
    prepare = runner.invoke(main, [
        "authorship", "prepare-aigcodeset", str(sample_benchmarks / "origin_0.csv"),
        str(sample_benchmarks / "origin_1.csv"), str(normalized),
    ])
    assert prepare.exit_code == 0, prepare.output
    run = runner.invoke(main, ["authorship", "benchmark", str(normalized), str(output), "--model", "class_prior"])
    assert run.exit_code == 0, run.output
    report = output / "results.json"
    assert len(list(output.glob("predictions-*.csv"))) == 1
    check = runner.invoke(main, ["authorship", "verify", str(report)])
    assert check.exit_code == 0, check.output


def test_gradio_clone_panel_exports_source_free_evidence(sample_benchmarks):
    pytest.importorskip("gradio")
    from gradio_app.research_benchmarks import evaluate_clones_gradio

    summary, rows, output = evaluate_clones_gradio(
        str(sample_benchmarks / "references.csv"), str(sample_benchmarks / "findings.csv"), 0.70,
    )
    assert "1 / 2" in summary
    assert set(rows["clone_type"]) >= {"type1", "weak_type3_type4"}
    assert json.loads(Path(output).read_text())["protocol"]["coverage_threshold"] == 0.7
