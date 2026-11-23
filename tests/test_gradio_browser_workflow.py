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
    def __init__(self):
        self.init_scripts = []
        self.context = SimpleNamespace(
            pages=[self], add_init_script=self.add_init_script, on=self.on,
        )

    def add_init_script(self, script):
        self.init_scripts.append(script)

    def evaluate(self, _script):
        return {"joins": [], "events": []}

    def is_closed(self):
        return False

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
    assert any("__matheelQueueEvents" in script for script in page.init_scripts)
    assert len(list(tmp_path.glob("*-queue.json"))) == 1


QUEUE_CLOSE_ABORT = (
    "console.error: AbortError: signal is aborted without reason\n"
    "    at ye (http://127.0.0.1:9999/assets/index-DputZZxm.js:4:1346)\n"
    "    at Zt.a.onmessage (http://127.0.0.1:9999/assets/index-DputZZxm.js:4:841)\n"
    "    at http://127.0.0.1:9999/assets/index-DputZZxm.js:4:2505"
)


def _successful_closed_queue():
    return {
        "joins": [{"event_id": "finished-job"}],
        "events": [
            {"msg": "process_completed", "event_id": "finished-job", "success": True},
            {"msg": "close_stream"},
        ],
    }


def test_browser_diagnostics_accept_a_clean_queue_close():
    assert BROWSER_CONFTEST._browser_diagnostic_errors([], {object(): _successful_closed_queue()}) == []


@pytest.mark.parametrize("error", [
    QUEUE_CLOSE_ABORT,
    "console.error: AbortError: failed unrelated request",
    "console.error: TypeError: broken application output",
    "pageerror: browser callback failed",
])
def test_browser_diagnostics_reject_errors_even_after_a_successful_queue(error):
    traces = {object(): _successful_closed_queue()}
    assert BROWSER_CONFTEST._browser_diagnostic_errors([error], traces) == [error]


def test_browser_fixture_retains_console_error_diagnostics_after_queue_completion(tmp_path):
    class CompletedQueuePage(SetupFailingPage):
        def __init__(self):
            self.handlers = {}
            super().__init__()

        def on(self, event, callback):
            self.handlers[event] = callback

        def goto(self, _url, **_kwargs):
            self.handlers["console"](SimpleNamespace(
                type="error", text=QUEUE_CLOSE_ABORT.removeprefix("console.error: "),
            ))

        def get_by_role(self, *_args, **_kwargs):
            return SimpleNamespace(wait_for=lambda **_kwargs: None)

        def wait_for_function(self, *_args, **_kwargs):
            return None

        def evaluate(self, _script):
            return _successful_closed_queue()

    page = CompletedQueuePage()
    request = SimpleNamespace(node=SimpleNamespace(nodeid="browser::normal-close"))
    fixture = BROWSER_CONFTEST.gradio_page.__wrapped__(
        page, request, "http://127.0.0.1:9999/", tmp_path,
    )

    assert next(fixture) is page
    with pytest.raises(AssertionError, match="Browser emitted unexpected errors"):
        next(fixture)

    logs = list(tmp_path.glob("*-browser.log"))
    assert len(logs) == 1
    assert QUEUE_CLOSE_ABORT in logs[0].read_text()
    assert len(list(tmp_path.glob("*-queue.json"))) == 1


def test_browser_diagnostics_keep_failed_or_incomplete_jobs_fatal():
    failed_trace = {
        "joins": [{"event_id": "failed"}, {"event_id": "unfinished"}],
        "events": [
            {"msg": "process_completed", "event_id": "failed", "success": False},
            {"msg": "process_completed", "event_id": "failed", "success": True},
        ],
    }
    fatal = BROWSER_CONFTEST._browser_diagnostic_errors(
        [], {object(): _successful_closed_queue(), object(): failed_trace},
    )

    assert any(item.startswith("queue-error:") and '"event_id": "failed"' in item for item in fatal)
    assert "queue-incomplete: unfinished" in fatal
