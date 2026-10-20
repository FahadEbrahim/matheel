import json

from scripts.check_benchmark_regressions import DEFAULT_FIXTURE, run_regression


def test_offline_benchmark_score_regression():
    report = run_regression()
    assert report["passed"], report["differences"]
    assert all(run["elapsed_seconds"] >= 0 for run in report["runs"])


def test_regression_detects_changed_scores(tmp_path):
    fixture = json.loads(DEFAULT_FIXTURE.read_text())
    fixture["methods"][0]["expected_scores"][0] = .125
    target = tmp_path / "altered.json"
    target.write_text(json.dumps(fixture))
    report = run_regression(target)
    assert not report["passed"]
    assert any("Levenshtein pair 0" in value for value in report["differences"])
