from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_model_workflow_overrides_default_marker_exclusion():
    workflow = (ROOT / ".github" / "workflows" / "integration-tests.yml").read_text(
        encoding="utf-8"
    )
    command = "python -m pytest -o addopts='' -m integration tests/test_real_models.py"

    assert workflow.count(command) == 2
    assert f"{command} --collect-only" in workflow
