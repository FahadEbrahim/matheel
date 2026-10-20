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


def test_browser_diagnostics_log_only_proven_normal_queue_close_cleanup():
    current_page = object()
    fatal, informational = BROWSER_CONFTEST._classify_browser_diagnostics(
        [QUEUE_CLOSE_ABORT], [(current_page, QUEUE_CLOSE_ABORT)],
        {current_page: _successful_closed_queue()},
    )

    assert fatal == []
    assert len(informational) == 1
    assert "expected-cleanup: Gradio 5.50 normal queue close" in informational[0]
    assert QUEUE_CLOSE_ABORT in informational[0]


def test_browser_fixture_retains_expected_cleanup_diagnostics_on_success(tmp_path):
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
    with pytest.raises(StopIteration):
        next(fixture)

    logs = list(tmp_path.glob("*-browser.log"))
    assert len(logs) == 1
    assert "expected-cleanup: Gradio 5.50 normal queue close" in logs[0].read_text()
    assert len(list(tmp_path.glob("*-queue.json"))) == 1


@pytest.mark.parametrize("case", [
    "other_abort", "changed_client", "no_server_close", "no_admissions",
    "incomplete_job", "failed_job", "duplicate_completion", "other_page_close",
    "ordinary_console_error",
])
def test_browser_diagnostics_keep_unproven_cleanup_and_other_console_errors_fatal(case):
    current_page = object()
    trace = _successful_closed_queue()
    traces = {current_page: trace}
    event = QUEUE_CLOSE_ABORT
    if case == "other_abort":
        event = "console.error: AbortError: failed unrelated request"
    elif case == "changed_client":
        event = event.replace("index-DputZZxm.js", "index-another-client.js")
    elif case == "no_server_close":
        trace["events"].pop()
    elif case == "no_admissions":
        trace["joins"].clear()
    elif case == "incomplete_job":
        trace["joins"].append({"event_id": "unfinished-job"})
    elif case == "failed_job":
        trace["events"][0]["success"] = False
    elif case == "duplicate_completion":
        trace["events"].insert(0, {
            "msg": "process_completed", "event_id": "finished-job", "success": False,
        })
    elif case == "other_page_close":
        traces = {object(): trace}
    elif case == "ordinary_console_error":
        event = "console.error: TypeError: broken application output"

    fatal, informational = BROWSER_CONFTEST._classify_browser_diagnostics(
        [event], [(current_page, event)], traces,
    )

    assert event in fatal
    assert informational == []


def test_browser_diagnostics_keep_page_errors_and_failed_or_incomplete_jobs_fatal():
    first_page, second_page = object(), object()
    page_error = QUEUE_CLOSE_ABORT.replace("console.error:", "pageerror:")
    failed_trace = {
        "joins": [{"event_id": "failed"}, {"event_id": "unfinished"}],
        "events": [{"msg": "process_completed", "event_id": "failed", "success": False}],
    }
    fatal, informational = BROWSER_CONFTEST._classify_browser_diagnostics(
        [QUEUE_CLOSE_ABORT, page_error], [(first_page, QUEUE_CLOSE_ABORT)],
        {first_page: _successful_closed_queue(), second_page: failed_trace},
    )

    assert page_error in fatal
    assert any(item.startswith("queue-error:") and '"event_id": "failed"' in item for item in fatal)
    assert "queue-incomplete: unfinished" in fatal
    assert len(informational) == 1
