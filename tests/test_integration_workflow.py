from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("test_files", [
    "tests/test_real_models.py tests/test_experiment_real_models.py",
    "tests/test_external_detectors_integration.py",
])
def test_real_integration_workflow_overrides_default_marker_exclusion(test_files):
    workflow = (ROOT / ".github" / "workflows" / "integration-tests.yml").read_text(
        encoding="utf-8"
    )
    command = f"python -m pytest -o addopts='' -m integration {test_files}"

    assert workflow.count(command) == 2
    assert f"{command} --collect-only" in workflow


def test_real_integrations_do_not_run_on_pull_requests():
    workflow = (ROOT / ".github/workflows/integration-tests.yml").read_text(encoding="utf-8")
    assert "  pull_request:" not in workflow
    assert "  schedule:" in workflow
    assert "  workflow_dispatch:" in workflow
    assert "  models:" in workflow
    assert "  detectors:" in workflow
    assert "needs:" not in workflow


@pytest.mark.parametrize("testcase", [
    '<testcase name="optional"><skipped message="Backend unavailable" /></testcase>',
    '<testcase name="broken"><failure message="Scoring failed" /></testcase>',
    '<testcase name="broken"><error message="No runtime" /></testcase>',
    "",
])
def test_integration_report_rejects_silent_skips_failures_and_missing_tests(tmp_path, testcase):
    check_report = runpy.run_path(str(ROOT / "scripts/ci/check_integration_report.py"))["check_report"]
    report = tmp_path / "results.xml"
    report.write_text(f"<testsuites><testsuite>{testcase}</testsuite></testsuites>", encoding="utf-8")
    with pytest.raises(ValueError):
        check_report(report, expected_tests=1, allowed_skips=0)


def test_integration_report_only_allows_the_declared_skip_reason(tmp_path):
    check_report = runpy.run_path(str(ROOT / "scripts/ci/check_integration_report.py"))["check_report"]
    report = tmp_path / "results.xml"
    report.write_text(
        '<testsuites><testsuite><testcase name="directional">'
        '<skipped message="Only JPlag supports directionality" /></testcase>'
        '<testcase name="coverage" /></testsuite></testsuites>', encoding="utf-8",
    )
    check_report(report, 2, 1, "Only JPlag supports directionality")
    with pytest.raises(ValueError, match="unexpected skip reason"):
        check_report(report, 2, 1, "Missing runtime")
