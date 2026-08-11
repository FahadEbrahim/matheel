import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFTST_PATH = ROOT / "tests" / "browser" / "conftest.py"
CONFTST_SPEC = importlib.util.spec_from_file_location("matheel_browser_conftest", CONFTST_PATH)
assert CONFTST_SPEC is not None and CONFTST_SPEC.loader is not None
BROWSER_CONFTEST = importlib.util.module_from_spec(CONFTST_SPEC)
CONFTST_SPEC.loader.exec_module(BROWSER_CONFTEST)


class SetupFailingPage:
    def route(self, *_args):
        return None

    def route_web_socket(self, *_args):
        return None

    def on(self, *_args):
        return None

    def set_default_timeout(self, _timeout):
        return None

    def goto(self, _url, **_kwargs):
        raise RuntimeError("synthetic page setup failure")

    def screenshot(self, path, **_kwargs):
        Path(path).write_bytes(b"synthetic screenshot")


def test_gradio_browser_workflow_overrides_default_marker_exclusion():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    command = "python -m pytest -o addopts='' -m browser tests/browser"
    browser_job = workflow.split("  gradio-browser:", 1)[1].split("  package-smoke:", 1)[0]

    assert workflow.count(command) == 2
    assert f"{command} --collect-only" in workflow
    assert "MATHEEL_BROWSER_ARTIFACTS_DIR: test-results/browser" in browser_job
    assert "--output test-results/browser" in browser_job


def test_gradio_page_persists_diagnostics_when_setup_fails(tmp_path):
    page = SetupFailingPage()
    request = SimpleNamespace(node=SimpleNamespace(nodeid="browser::setup-failure"))
    fixture = BROWSER_CONFTEST.gradio_page.__wrapped__(
        page,
        request,
        "http://127.0.0.1:9999/",
        tmp_path,
    )

    with pytest.raises(RuntimeError, match="synthetic page setup failure"):
        next(fixture)

    logs = list(tmp_path.glob("*-browser.log"))
    assert len(logs) == 1
    assert "fixture-error: RuntimeError: synthetic page setup failure" in logs[0].read_text(
        encoding="utf-8"
    )
