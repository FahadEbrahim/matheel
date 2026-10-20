import json
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXED_ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def _write_deterministic_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return path


@pytest.fixture(scope="session")
def collection_zip(tmp_path_factory):
    return _write_deterministic_zip(
        tmp_path_factory.mktemp("browser-fixtures") / "collection.zip",
        {
            "alpha.py": "def total(values):\n    return sum(values)\n",
            "beta.py": "def total(values):\n    return sum(values)\n",
            "gamma.py": "def product(values):\n    result = 1\n    for value in values:\n        result *= value\n    return result\n",
        },
    )


@pytest.fixture(scope="session")
def normalized_dataset_zip(tmp_path_factory):
    prefix = "browser_pair_dataset"
    return _write_deterministic_zip(
        tmp_path_factory.mktemp("browser-datasets") / "browser_pair_dataset.zip",
        {
            f"{prefix}/metadata.json": json.dumps(
                {
                    "dataset_kind": "pair_classification",
                    "name": "browser_pair_dataset",
                    "task_type": "plagiarism",
                },
                sort_keys=True,
            ),
            f"{prefix}/files.csv": (
                "file_id,file_path\n"
                "a,files/a.py\n"
                "b,files/b.py\n"
                "c,files/c.py\n"
                "d,files/d.py\n"
            ),
            f"{prefix}/pairs.csv": (
                "left_id,right_id,label\n"
                "a,b,1\n"
                "c,d,1\n"
                "a,c,0\n"
                "b,d,0\n"
            ),
            f"{prefix}/files/a.py": "print('same')\n",
            f"{prefix}/files/b.py": "print('same')\n",
            f"{prefix}/files/c.py": "print('left')\n",
            f"{prefix}/files/d.py": "print('right')\n",
        },
    )


@pytest.fixture(scope="session")
def invalid_dataset_zip(tmp_path_factory):
    return _write_deterministic_zip(
        tmp_path_factory.mktemp("browser-invalid") / "not_a_dataset.zip",
        {"README.txt": "This is intentionally not a normalized Matheel dataset.\n"},
    )


@pytest.fixture(scope="session")
def browser_artifacts_dir(tmp_path_factory):
    configured = os.environ.get("MATHEEL_BROWSER_ARTIFACTS_DIR")
    root = Path(configured).resolve() if configured else tmp_path_factory.mktemp("browser-artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unused_local_port():
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return port_socket.getsockname()[1]


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture(scope="session")
def official_benchmark_json(tmp_path_factory):
    import pandas as pd

    from matheel.benchmark_protocol import freeze_benchmark_protocol, run_benchmark_protocol
    from matheel.datasets import write_pair_dataset, write_retrieval_dataset

    root = tmp_path_factory.mktemp("browser-official-benchmark")
    files = pd.DataFrame([
        {"file_id": "a", "text": "print(1)", "suffix": ".py"},
        {"file_id": "b", "text": "print(1)", "suffix": ".py"},
        {"file_id": "c", "text": "unrelated long program", "suffix": ".py"},
    ])
    write_pair_dataset(root / "pairs", files=files, pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1},
        {"left_id": "a", "right_id": "c", "label": 0},
    ]), metadata={"task_type": "plagiarism"})
    write_retrieval_dataset(
        root / "retrieval", files=files,
        queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame([
            {"document_id": "positive", "file_id": "b"},
            {"document_id": "negative", "file_id": "c"},
        ]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "positive", "relevance": 1}]),
        metadata={"task_type": "plagiarism"},
    )
    broken = root / "unsupported.py"
    broken.write_text('def score_pair(code_a, code_b):\n    raise RuntimeError("unsupported browser fixture")\n')
    protocol = freeze_benchmark_protocol({
        "domain": "plagiarism", "pair_metrics": ["f1", "auroc"],
        "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
        "datasets": [
            {"name": name, "task": task, "path": str(root / name),
             "source_url": "https://example.org/versioned-data", "source_revision": "a" * 40,
             "split": "test"}
            for name, task in (("pairs", "pair"), ("retrieval", "retrieval"))
        ],
    })
    _, artifacts = run_benchmark_protocol(
        protocol,
        [{"name": "Levenshtein", "feature_weights": {"levenshtein": 1}},
         {"name": "Unsupported", "algorithm_path": str(broken)}],
        output_dir=root / "published", basename="matheel_plagiarism_v1",
    )
    return artifacts["json"]


@pytest.fixture(scope="session")
def gradio_server(browser_artifacts_dir, official_benchmark_json):
    port = _unused_local_port()
    url = f"http://127.0.0.1:{port}/"
    log_path = browser_artifacts_dir / "gradio-server.log"
    environment = os.environ.copy()
    environment.update(
        {
            "GRADIO_ANALYTICS_ENABLED": "False",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                os.fspath(Path(__file__).with_name("serve_gradio.py")),
                "--port",
                str(port),
                "--official-benchmark",
                os.fspath(official_benchmark_json),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urlopen(f"{url}config", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, URLError):
                time.sleep(0.2)
        else:
            _stop_process(process)
            raise RuntimeError(f"Gradio server did not become ready; see {log_path}")

        if process.poll() is not None:
            log_handle.flush()
            details = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Gradio server exited during startup:\n{details[-4000:]}")

        try:
            yield url
        finally:
            _stop_process(process)


@pytest.fixture(scope="session")
def gradio_config(gradio_server):
    with urlopen(f"{gradio_server}config", timeout=10) as response:
        return json.load(response)


@pytest.fixture
def gradio_page(page, request, gradio_server, browser_artifacts_dir):
    events = []
    unexpected_external_traffic = []
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    offline_stubs = {
        "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js": (
            "application/javascript",
            "",
        ),
        "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap": (
            "text/css",
            "",
        ),
    }

    def record_external(kind, method, url):
        event = f"external-{kind}: {method} {url}"
        unexpected_external_traffic.append(event)
        events.append(event)

    def route_request(route):
        request_url = route.request.url
        parsed = urlsplit(request_url)
        if request_url in offline_stubs:
            content_type, body = offline_stubs[request_url]
            events.append(f"external-stubbed: {route.request.method} {request_url}")
            route.fulfill(status=200, content_type=content_type, body=body)
            return
        if parsed.hostname not in local_hosts:
            record_external("request", route.request.method, request_url)
            route.abort("blockedbyclient")
            return
        # Gradio 5.50 can mount its progress EventSource before assigning the
        # upload ID. The file POST is independent, so complete only that broken
        # local progress stream and keep console errors meaningful.
        if (
            parsed.path.endswith("/gradio_api/upload_progress")
            and parse_qs(parsed.query).get("upload_id") == ["undefined"]
        ):
            route.fulfill(
                status=200,
                content_type="text/event-stream",
                body='data: {"msg": "done"}\n\n',
            )
            return
        route.continue_()

    def route_web_socket(web_socket):
        parsed = urlsplit(web_socket.url)
        if parsed.hostname not in local_hosts:
            record_external("websocket", "CONNECT", web_socket.url)
            web_socket.close(code=1008, reason="External browser traffic is disabled")
            return
        web_socket.connect_to_server()

    page.route("**/*", route_request)
    page.route_web_socket("**/*", route_web_socket)
    # Observe actual queue completion rather than treating a POST response or
    # a transient frontend pending flag as evidence that outputs were applied.
    # Cloned response bodies leave Gradio's own streams and errors untouched.
    page.add_init_script("""(() => {
        window.__matheelQueueJoins = [];
        window.__matheelQueueEvents = [];
        window.__matheelGradioTargets = [];
        const originalListen = EventTarget.prototype.addEventListener;
        EventTarget.prototype.addEventListener = function(type) {
            if (type === "gradio") window.__matheelGradioTargets.push(this);
            return originalListen.apply(this, arguments);
        };
        const originalFetch = window.fetch;
        window.fetch = async function(input, init) {
            const response = await originalFetch.apply(this, arguments);
            const url = typeof input === "string" ? input : input.url || String(input);
            if (url.includes("/gradio_api/queue/join")) {
                const request = JSON.parse(init?.body || "{}");
                response.clone().json().then(result => {
                    window.__matheelQueueJoins.push({...request, ...result});
                }).catch(() => {});
            } else if (url.includes("/gradio_api/queue/data") && response.body) {
                (async () => {
                    const reader = response.clone().body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = "";
                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, {stream: true});
                        const lines = buffer.split("\\n");
                        buffer = lines.pop();
                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                window.__matheelQueueEvents.push(JSON.parse(line.slice(6)));
                            }
                        }
                    }
                })().catch(() => {});
            }
            return response;
        };
    })();""")
    page.on("console", lambda message: events.append(f"console.{message.type}: {message.text}"))
    page.on("pageerror", lambda error: events.append(f"pageerror: {error}"))
    page.on(
        "response",
        lambda response: events.append(f"http.{response.status}: {response.url}")
        if response.status >= 400
        else None,
    )
    page.on(
        "requestfailed",
        lambda failed_request: events.append(
            f"requestfailed: {failed_request.method} {failed_request.url} {failed_request.failure}"
        ),
    )
    page.set_default_timeout(15_000)
    setup_succeeded = False
    try:
        page.goto(gradio_server, wait_until="domcontentloaded")
        page.get_by_role("heading", name="Matheel", exact=True).wait_for(state="visible")
        setup_succeeded = True
        yield page
    except Exception as exc:
        events.append(f"fixture-error: {type(exc).__name__}: {exc}")
        raise
    finally:
        reports = [
            getattr(request.node, f"rep_{phase}", None)
            for phase in ("setup", "call", "teardown")
        ]
        report_failed = any(report is not None and report.failed for report in reports)
        fatal_events = [
            event
            for event in events
            if event.startswith("console.error:") or event.startswith("pageerror:")
        ]
        if setup_succeeded:
            queue_failures = page.evaluate("""() => (window.__matheelQueueEvents || []).filter(
                event => event.msg === "process_completed" && event.success === false
            )""")
            fatal_events.extend(f"queue-error: {json.dumps(event)}" for event in queue_failures)
        should_persist = (
            not setup_succeeded
            or report_failed
            or fatal_events
            or unexpected_external_traffic
        )
        if should_persist:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid).strip("_")
            try:
                queue_trace = page.evaluate("({joins: window.__matheelQueueJoins, events: window.__matheelQueueEvents})")
                (browser_artifacts_dir / f"{safe_name}-queue.json").write_text(
                    json.dumps(queue_trace, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                events.append(f"queue-trace-error: {exc}")
            try:
                page.screenshot(path=browser_artifacts_dir / f"{safe_name}.png", full_page=True)
            except Exception as exc:
                events.append(f"screenshot-error: {exc}")
            (browser_artifacts_dir / f"{safe_name}-browser.log").write_text(
                "\n".join(events) + "\n",
                encoding="utf-8",
            )
        if setup_succeeded:
            assert not unexpected_external_traffic, (
                "Unexpected external browser traffic:\n"
                + "\n".join(unexpected_external_traffic)
            )
            assert not fatal_events, "Browser emitted unexpected errors:\n" + "\n".join(fatal_events)
