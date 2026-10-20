import csv
import io
import json
import re
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest


pytestmark = pytest.mark.browser


def _component_roots(page, config, *, label=None, component_type=None):
    roots = []
    for component in config["components"]:
        props = component.get("props") or {}
        if label is not None and props.get("label") != label:
            continue
        if component_type is not None and component.get("type") != component_type:
            continue
        roots.append(page.locator(f"#component-{component['id']}"))
    return roots


def _visible_component(page, config, **criteria):
    visible = []

    def find_one_visible_component():
        visible[:] = [
            root for root in _component_roots(page, config, **criteria) if root.is_visible()
        ]
        return len(visible) == 1

    _wait_until(find_one_visible_component)
    assert len(visible) == 1, f"Expected one visible Gradio component for {criteria}, found {len(visible)}"
    return visible[0]


def _action_output(page, config, button_name, output_index=0):
    button_ids = {
        component["id"]
        for component in config["components"]
        if component.get("type") == "button"
        and (component.get("props") or {}).get("value") == button_name
    }
    assert len(button_ids) == 1, f"Expected one configured button named {button_name!r}"
    dependencies = [
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id in button_ids and trigger == "click"
            for target_id, trigger in dependency.get("targets", [])
        )
    ]
    assert len(dependencies) == 1, f"Expected one click dependency for {button_name!r}"
    output_id = dependencies[0]["outputs"][output_index]
    root = page.locator(f"#component-{output_id}")
    _wait_until(root.is_visible)
    return root


def _wait_until(predicate, *, timeout=30, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        time.sleep(interval)
    raise AssertionError("Timed out waiting for the browser workflow state")


def _wait_for_text(root, text, *, timeout=30):
    root.get_by_text(re.compile(re.escape(text), re.IGNORECASE), exact=False).first.wait_for(
        state="visible",
        timeout=timeout * 1000,
    )


def _select_dropdown(page, config, label, value):
    root = _visible_component(page, config, label=label, component_type="dropdown")
    listbox = root.get_by_role("listbox", name=label, exact=True)
    listbox.click()
    page.get_by_role("option", name=value, exact=True).last.click()
    _wait_until(lambda: listbox.input_value() == value)


def _select_dropdown_with_keyboard(page, config, label, value):
    root = _visible_component(page, config, label=label, component_type="dropdown")
    listbox = root.get_by_role("listbox", name=label, exact=True)
    listbox.click()
    listbox.fill(value)
    listbox.press("Enter")
    _wait_until(lambda: listbox.input_value() == value)


def _upload(page, config, label, path):
    root = _visible_component(page, config, label=label, component_type="file")
    file_input = root.locator("input[type=file]")
    if not file_input.count():
        root.get_by_role("button", name="Clear", exact=True).click()
        file_input.wait_for(state="attached")
    file_input.set_input_files(Path(path))
    root.get_by_label(Path(path).name, exact=True).wait_for(state="visible", timeout=30_000)
    return root


def _assert_zip_members(path, expected):
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
    assert set(expected) <= members


def _csv_records_from_zip(path, member):
    with zipfile.ZipFile(path) as archive:
        content = archive.read(member).decode("utf-8")
    return list(csv.DictReader(io.StringIO(content)))


def _download_file(page, component, destination):
    _wait_until(lambda: component.locator("a[href]").count() > 0)
    download_link = component.locator("a[download]")
    if not download_link.count():
        download_link = component.locator("a[href]")
    with page.expect_download(timeout=30_000) as download_info:
        download_link.first.click()
    download_info.value.save_as(destination)
    assert destination.is_file()
    return destination


def _open_tab(page, name):
    tab = page.get_by_role("tab", name=name, exact=True)
    tab.click()
    _wait_until(lambda: tab.get_attribute("aria-selected") == "true")


@contextmanager
def _hold_dataset_events(page, config, *, trigger):
    components = {component["id"]: component for component in config["components"]}
    validate_id = next(
        component_id
        for component_id, component in components.items()
        if component.get("type") == "button"
        and component["props"].get("value") == "Validate Dataset"
    )
    dependency = next(
        dependency
        for dependency in config["dependencies"]
        if [validate_id, "click"] in dependency["targets"]
    )
    trigger_ids = {
        target_id for target_id, event in dependency["targets"] if event == trigger
    }
    held = []

    def hold(route):
        data = route.request.post_data_json
        if data.get("fn_index") == dependency["id"] and data.get("trigger_id") in trigger_ids:
            held.append(route)
        else:
            route.continue_()

    pattern = "**/gradio_api/queue/join*"
    page.route(pattern, hold)

    def wait_for_pending_validation(filename):
        def is_latest_validation(data):
            return (
                data.get("fn_index") == dependency["id"]
                and data.get("trigger_id") == validate_id
                and data.get("data", [None])[0]
                and data["data"][0].get("orig_name") == filename
                and data["data"][1] == "Pair Classification"
            )

        def latest_request_is_held_or_admitted():
            admitted = page.evaluate("window.__matheelQueueJoins")
            return any(
                is_latest_validation(data)
                for data in [*(route.request.post_data_json for route in held), *admitted]
            )

        # Observe the actual latest-upload request. Gradio's pending_request and
        # final_event are transient implementation flags and can be replaced by
        # delayed change events even though the user clicked Validate last.
        _wait_until(latest_request_is_held_or_admitted)

    try:
        yield held, wait_for_pending_validation
    finally:
        page.unroute(pattern, hold)
        for route in held:
            route.continue_()
            # Admission acknowledgment gives the intentionally withheld events
            # an explicit server ordering before the next request is released.
            response = route.request.response()
            assert response is not None and response.ok


@pytest.mark.parametrize("hold_trigger", ["change", "click"])
def test_dataset_validation_preserves_latest_intent_while_an_event_is_pending(
    gradio_page, gradio_config, invalid_dataset_zip, normalized_dataset_zip, hold_trigger
):
    page = gradio_page
    _open_tab(page, "Datasets")
    validate = page.get_by_role("button", name="Validate Dataset", exact=True)
    if hold_trigger == "click":
        _upload(page, gradio_config, "Normalized Dataset ZIP", invalid_dataset_zip)
    with _hold_dataset_events(page, gradio_config, trigger=hold_trigger) as (held, wait_pending):
        if hold_trigger == "click":
            validate.click()
        else:
            _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
        # Pump Playwright's event loop while observing the intercepted request.
        _wait_until(lambda: page.evaluate("true") and bool(held))
        if hold_trigger == "click":
            _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
            task = _visible_component(page, gradio_config, label="Task", component_type="radio")
            task.get_by_role("radio", name="Retrieval", exact=True).check()
            task.get_by_role("radio", name="Pair Classification", exact=True).check()
        validate.click()
        wait_pending(normalized_dataset_zip.name)

    _wait_for_text(_action_output(page, gradio_config, "Validate Dataset"), "browser_pair_dataset")
    _wait_until(
        lambda: page.get_by_role("button", name="Run Dataset Evaluation", exact=True).is_enabled()
    )


def test_late_upload_change_cannot_erase_completed_validation(
    gradio_page, gradio_config, normalized_dataset_zip
):
    page = gradio_page
    _open_tab(page, "Datasets")
    _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
    page.get_by_role("button", name="Validate Dataset", exact=True).click()
    summary = _action_output(page, gradio_config, "Validate Dataset")
    _wait_for_text(summary, "browser_pair_dataset")
    run = page.get_by_role("button", name="Run Dataset Evaluation", exact=True)
    _wait_until(run.is_enabled)
    file_id = next(
        component["id"] for component in gradio_config["components"]
        if component.get("type") == "file"
        and component["props"].get("label") == "Normalized Dataset ZIP"
    )
    before = page.evaluate("window.__matheelQueueJoins.length")
    # Replay the real Gradio component event after validation, making the stale
    # reset ordering from #281 deterministic without changing the file value.
    page.evaluate(
        """id => window.__matheelGradioTargets.at(-1).dispatchEvent(new CustomEvent("gradio", {
            detail: {id, event: "change", data: null}, bubbles: true
        }))""", file_id,
    )
    page.wait_for_function(
        "count => window.__matheelQueueJoins.length > count", arg=before,
    )
    event_id = page.evaluate("window.__matheelQueueJoins.at(-1).event_id")
    page.wait_for_function(
        "id => window.__matheelQueueEvents.some(event => event.event_id === id && event.msg === 'process_completed')",
        arg=event_id,
    )
    assert page.evaluate(
        "id => window.__matheelQueueEvents.find(event => event.event_id === id && event.msg === 'process_completed').success",
        event_id,
    ) is True
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert "browser_pair_dataset" in summary.inner_text()
    assert run.is_enabled()


def test_pairwise_lexical_browser_workflow(gradio_page, gradio_config):
    page = gradio_page
    _select_dropdown(page, gradio_config, "Metric Preset", "Lexical Only")
    _wait_until(
        lambda: not any(
            root.is_visible()
            for root in _component_roots(
                page,
                gradio_config,
                label="Embedding Model",
            )
        )
    )

    code = "def total(values):\n    return sum(values)\n"
    _visible_component(page, gradio_config, label="Code A", component_type="textbox").get_by_role(
        "textbox"
    ).fill(code)
    _visible_component(page, gradio_config, label="Code B", component_type="textbox").get_by_role(
        "textbox"
    ).fill(code)
    page.get_by_role("button", name="Run Pair", exact=True).click()

    summary = _action_output(page, gradio_config, "Run Pair")
    _wait_for_text(summary, "1.0000")


def test_pairwise_lcs_browser_workflow(gradio_page, gradio_config):
    page = gradio_page
    _select_dropdown(page, gradio_config, "Metric Preset", "Longest Common Subsequence (LCS)")
    table = _visible_component(
        page, gradio_config, label="Additional lexical weights", component_type="dataframe"
    )
    _wait_for_text(table, "Longest Common Subsequence (LCS)")
    _visible_component(page, gradio_config, label="Code A", component_type="textbox").get_by_role(
        "textbox"
    ).fill("abcdef")
    _visible_component(page, gradio_config, label="Code B", component_type="textbox").get_by_role(
        "textbox"
    ).fill("ace")
    page.get_by_role("button", name="Run Pair", exact=True).click()

    _wait_for_text(_action_output(page, gradio_config, "Run Pair"), "0.5000")


def test_mobile_workflow_overflow_menu_accessibility(gradio_page):
    page = gradio_page
    page.set_viewport_size({"width": 390, "height": 844})
    trigger = page.get_by_role("button", name="More workflow tabs", exact=True)
    trigger.wait_for(state="visible")

    assert trigger.get_attribute("aria-haspopup") == "menu"
    assert trigger.get_attribute("aria-expanded") == "false"
    assert page.get_by_role("menu").count() == 0

    trigger.click()
    _wait_until(
        lambda: trigger.get_attribute("aria-expanded") == "true"
        and page.get_by_role("menu").count() == 1
    )

    trigger.click()
    _wait_until(
        lambda: trigger.get_attribute("aria-expanded") == "false"
        and page.get_by_role("menu").count() == 0
    )


def test_collection_zip_upload_and_lexical_scoring(
    gradio_page,
    gradio_config,
    collection_zip,
):
    page = gradio_page
    _open_tab(page, "Collection")
    _upload(page, gradio_config, "Code ZIP", collection_zip)
    _select_dropdown(page, gradio_config, "Metric Preset", "Lexical Only")
    _wait_until(
        lambda: not any(
            root.is_visible()
            for root in _component_roots(
                page,
                gradio_config,
                label="Embedding Model",
            )
        )
    )
    page.get_by_role("button", name="Run Collection", exact=True).click()

    ranked_pairs = _visible_component(
        page,
        gradio_config,
        label="Ranked Pairs",
        component_type="dataframe",
    )
    _wait_for_text(ranked_pairs, "alpha.py")
    _wait_for_text(ranked_pairs, "beta.py")
    _wait_for_text(_action_output(page, gradio_config, "Run Collection"), "1.000")


def test_demo_leaderboard_displays_plagiarism_category_independently_of_task(
    gradio_page, gradio_config
):
    page = gradio_page
    _open_tab(page, "Reports")
    _open_tab(page, "Ready-made Leaderboard")
    page.get_by_text("Demo leaderboard", exact=False).first.wait_for(state="visible")
    _select_dropdown(page, gradio_config, "Dataset Category", "Plagiarism")
    ranking = _visible_component(
        page, gradio_config, label="Interactive Aggregate Ranking", component_type="dataframe"
    )
    _wait_for_text(ranking, "plagiarism")
    _select_dropdown(page, gradio_config, "Task", "retrieval")
    # The task callback rebuilds filter choices and both tables together.
    # Wait for that transaction before interacting with its replacement control.
    _wait_for_text(ranking, "retrieval")
    _select_dropdown(page, gradio_config, "Dataset Category", "Plagiarism")
    _wait_for_text(ranking, "retrieval")
    _wait_for_text(ranking, "plagiarism")
    page.get_by_role("button", name="Reset Leaderboard Filters", exact=True).click()
    _wait_for_text(ranking, "pair")


def test_official_benchmark_preserves_failed_methods_and_task_specific_rankings(
    gradio_page, gradio_config, tmp_path
):
    page = gradio_page
    _open_tab(page, "Reports")
    _open_tab(page, "Official Benchmark")
    summary = _action_output(page, gradio_config, "Refresh Official Results")
    _wait_for_text(summary, "Validated published results")
    _wait_for_text(summary, "Matheel-Plagiarism-v1")
    ranking = _visible_component(
        page, gradio_config, label="Official Aggregate Ranking", component_type="dataframe"
    )
    _wait_for_text(ranking, "Unsupported")
    _wait_for_text(ranking, "Levenshtein")
    _select_dropdown(page, gradio_config, "Official Evaluation Task", "Retrieval")
    _wait_for_text(ranking, "mean_average_precision")
    _select_dropdown(page, gradio_config, "Official Ranking Metric", "ndcg_at_k")
    _wait_for_text(ranking, "ndcg_at_k")
    method = next(
        label for component in gradio_config["components"]
        if component.get("props", {}).get("label") == "Official Method Configuration"
        for label, _ in component["props"]["choices"] if label.startswith("Unsupported")
    )
    _select_dropdown(page, gradio_config, "Official Method Configuration", method)
    _wait_until(lambda: ranking.get_by_text("Levenshtein", exact=True).count() == 0)
    _wait_for_text(ranking, "Unsupported")
    artifact = _visible_component(
        page, gradio_config, label="Official Benchmark Result JSON", component_type="file"
    )
    path = _download_file(page, artifact, tmp_path / "official-benchmark.json")
    payload = json.loads(path.read_text())
    failed = [row for row in payload["aggregate"] if row["algorithm_name"] == "Unsupported"]
    assert failed and all(row["rank"] is None and row["eligible"] is False for row in failed)
    assert all(row["mean_score"] is None for row in failed)
    assert payload["metadata"]["benchmark_protocol"]["protocol_id"]


def test_dataset_exports_and_leaderboard_artifact_inspection(
    gradio_page,
    gradio_config,
    invalid_dataset_zip,
    normalized_dataset_zip,
    tmp_path,
):
    page = gradio_page
    _open_tab(page, "Datasets")

    _upload(page, gradio_config, "Normalized Dataset ZIP", invalid_dataset_zip)
    page.get_by_role("button", name="Validate Dataset", exact=True).click()
    error_text = "Could not find a normalized Matheel dataset"
    page.get_by_text(re.compile(error_text, re.IGNORECASE), exact=False).last.wait_for(
        state="visible",
        timeout=30_000,
    )

    _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
    page.get_by_role("button", name="Validate Dataset", exact=True).click()
    validation_summary = _action_output(page, gradio_config, "Validate Dataset")
    _wait_for_text(validation_summary, "browser_pair_dataset")
    validation_artifacts = _visible_component(
        page,
        gradio_config,
        label="Validation Artifacts",
        component_type="file",
    )
    validation_zip = _download_file(
        page,
        validation_artifacts,
        tmp_path / "dataset_validation_artifacts.zip",
    )
    _assert_zip_members(
        validation_zip,
        {"dataset_validation_report.json", "dataset_validation_report.html"},
    )

    page.get_by_role("button", name="Run Dataset Evaluation", exact=True).click()
    evaluation_summary = _action_output(page, gradio_config, "Run Dataset Evaluation")
    _wait_for_text(evaluation_summary, "browser_pair_dataset")
    _wait_for_text(evaluation_summary, "Lexical Only")
    metrics = _visible_component(
        page,
        gradio_config,
        label="Metrics",
        component_type="dataframe",
    )
    _wait_for_text(metrics, "F1")
    scored_rows = _visible_component(
        page,
        gradio_config,
        label="Scored Rows",
        component_type="dataframe",
    )
    scored_rows.get_by_text("a", exact=True).first.wait_for(state="visible", timeout=30_000)
    scored_rows.get_by_text("b", exact=True).first.wait_for(state="visible", timeout=30_000)
    evaluation_artifacts = _visible_component(
        page,
        gradio_config,
        label="Leaderboard Artifacts",
        component_type="file",
    )
    evaluation_zip = _download_file(
        page,
        evaluation_artifacts,
        tmp_path / "leaderboard_artifacts.zip",
    )
    _assert_zip_members(
        evaluation_zip,
        {"leaderboard_manifest.json", "pair_metrics.json", "pair_scored_rows.csv"},
    )
    scored_records = _csv_records_from_zip(evaluation_zip, "pair_scored_rows.csv")
    assert len(scored_records) == 4
    assert {
        (record["left_id"], record["right_id"], record["label"])
        for record in scored_records
    } == {
        ("a", "b", "1"),
        ("c", "d", "1"),
        ("a", "c", "0"),
        ("b", "d", "0"),
    }

    _open_tab(page, "Reports")
    _open_tab(page, "Inspect Artifacts")
    _upload(page, gradio_config, "Leaderboard JSON or ZIP", evaluation_zip)
    page.get_by_role("button", name="Inspect Leaderboard", exact=True).click()
    inspector_summary = _action_output(page, gradio_config, "Inspect Leaderboard")
    _wait_for_text(inspector_summary, "browser_pair_dataset Dataset Evaluation")
    aggregate = _visible_component(
        page,
        gradio_config,
        label="Aggregate Ranking",
        component_type="dataframe",
    )
    _wait_for_text(aggregate, "Lexical Only")
    per_dataset = _visible_component(
        page,
        gradio_config,
        label="Per-Dataset Ranking",
        component_type="dataframe",
    )
    _wait_for_text(per_dataset, "browser_pair_dataset")
    report_artifacts = _visible_component(
        page,
        gradio_config,
        label="Leaderboard Report Artifacts",
        component_type="file",
    )
    report_zip = _download_file(
        page,
        report_artifacts,
        tmp_path / "leaderboard_report_artifacts.zip",
    )
    _assert_zip_members(
        report_zip,
        {
            "leaderboard_report.json",
            "leaderboard_report.html",
            "leaderboard_report_aggregate.csv",
            "leaderboard_report_per_dataset.csv",
        },
    )


@contextmanager
def _hold_dataset_callback(barriers, stage):
    paths = {name: barriers / f"{stage}.{name}" for name in ("hold", "started", "release", "finished")}
    for path in paths.values():
        path.unlink(missing_ok=True)
    paths["hold"].touch()
    try:
        yield paths
    finally:
        paths["release"].touch()
        if paths["started"].exists():
            _wait_until(paths["finished"].exists)
        for path in paths.values():
            path.unlink(missing_ok=True)


def _button_dependency(config, button_name):
    button_id = next(component["id"] for component in config["components"]
                     if component.get("type") == "button" and component["props"].get("value") == button_name)
    return next(dependency for dependency in config["dependencies"] if [button_id, "click"] in dependency["targets"])


def _wait_for_dataset_action_completion(page, config, action, previous_joins):
    dependency = _button_dependency(config, action)
    chain_ids = {dependency["id"], *(item["id"] for item in config["dependencies"] if item.get("trigger_after") == dependency["id"])}
    page.wait_for_function("""({ids, start}) => {
        const joins = window.__matheelQueueJoins.slice(start).filter(item => ids.includes(item.fn_index));
        return joins.length === ids.length && joins.every(join => window.__matheelQueueEvents.some(
            event => event.event_id === join.event_id && event.msg === 'process_completed' && event.success
        ));
    }""", arg={"ids": list(chain_ids), "start": previous_joins}, timeout=30_000)


def _dispatch_component_event_and_wait(page, config, component_id, event):
    dependency = next(item for item in config["dependencies"] if [component_id, event] in item["targets"])
    before = page.evaluate("window.__matheelQueueJoins.length")
    page.evaluate("""({id, event}) => window.__matheelGradioTargets.at(-1).dispatchEvent(new CustomEvent("gradio", {
        detail: {id, event, data: null}, bubbles: true
    }))""", {"id": component_id, "event": event})
    page.wait_for_function("""({index, start}) => window.__matheelQueueJoins.slice(start).some(
        join => join.fn_index === index && window.__matheelQueueEvents.some(
            event => event.event_id === join.event_id && event.msg === 'process_completed' && event.success
        )
    )""", arg={"index": dependency["id"], "start": before})
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


@pytest.mark.parametrize("reset", ["upload", "task", "metric"])
def test_completed_old_evaluation_cannot_restore_results_after_dataset_reset(
    gradio_page, gradio_config, normalized_dataset_zip, invalid_dataset_zip,
    dataset_callback_barriers, reset,
):
    page = gradio_page
    _open_tab(page, "Datasets")
    _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
    _select_dropdown(page, gradio_config, "Metric Preset", "Lexical Only")
    page.get_by_role("button", name="Validate Dataset", exact=True).click()
    _wait_for_text(_action_output(page, gradio_config, "Validate Dataset"), "browser_pair_dataset")
    run = page.get_by_role("button", name="Run Dataset Evaluation", exact=True)
    _wait_until(run.is_enabled)
    before = page.evaluate("window.__matheelQueueJoins.length")
    with _hold_dataset_callback(dataset_callback_barriers, "evaluation") as held:
        run.click()
        _wait_until(lambda: page.evaluate("true") and held["started"].exists())
        if reset == "upload":
            _upload(page, gradio_config, "Normalized Dataset ZIP", invalid_dataset_zip)
            _wait_until(lambda: run.is_disabled())
        elif reset == "task":
            task = _visible_component(page, gradio_config, label="Task", component_type="radio")
            task.get_by_role("radio", name="Retrieval", exact=True).check()
            _wait_until(lambda: run.is_disabled())
            # Gradio can emit input when the previously hidden retrieval
            # slider mounts. Force that event after reset instead of relying
            # on platform-specific mounting timing observed in CI.
            slider_id = next(component["id"] for component in gradio_config["components"]
                             if component.get("props", {}).get("label") == "Retrieval k")
            _dispatch_component_event_and_wait(page, gradio_config, slider_id, "input")
        else:
            _select_dropdown(page, gradio_config, "Metric Preset", "Jaro-Winkler")
        status_id = _button_dependency(gradio_config, "Run Dataset Evaluation")["outputs"][7]
        status = page.locator(f"#component-{status_id}")
        if reset == "metric":
            _wait_for_text(status, "Settings changed")
        else:
            _wait_for_text(status, "Upload a dataset")
        held["release"].touch()
        _wait_for_dataset_action_completion(page, gradio_config, "Run Dataset Evaluation", before)
    summary_id = _button_dependency(gradio_config, "Run Dataset Evaluation")["outputs"][0]
    assert not page.locator(f"#component-{summary_id}").is_visible()
    assert not page.locator("#dataset-next-actions").is_visible()
    assert not page.get_by_role("button", name="Tune Pair Threshold", exact=True).is_visible()
    artifacts = _visible_component(page, gradio_config, label="Leaderboard Artifacts", component_type="file")
    assert artifacts.locator("a[download]").count() == 0
    if reset == "upload":
        page.get_by_label(invalid_dataset_zip.name, exact=True).wait_for(state="visible")
        assert run.is_disabled()
    elif reset == "task":
        assert page.get_by_role("radio", name="Retrieval", exact=True).is_checked()
        assert run.is_disabled()
    else:
        selected = _visible_component(page, gradio_config, label="Metric Preset", component_type="dropdown")
        assert selected.get_by_role("listbox", name="Metric Preset", exact=True).input_value() == "Jaro-Winkler"
        assert run.is_enabled()


def test_old_queued_evaluation_cannot_start_after_settings_reset(
    gradio_page, gradio_server, gradio_config, normalized_dataset_zip, dataset_callback_barriers,
):
    first = gradio_page
    second = first.context.new_page()
    second.goto(gradio_server, wait_until="domcontentloaded")
    second.get_by_role("heading", name="Matheel", exact=True).wait_for(state="visible")
    for current in (first, second):
        _open_tab(current, "Datasets")
        _upload(current, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
        _select_dropdown(current, gradio_config, "Metric Preset", "Lexical Only")
        current.get_by_role("button", name="Validate Dataset", exact=True).click()
        _wait_for_text(_action_output(current, gradio_config, "Validate Dataset"), "browser_pair_dataset")
        _wait_until(current.get_by_role("button", name="Run Dataset Evaluation", exact=True).is_enabled)

    dependency = _button_dependency(gradio_config, "Run Dataset Evaluation")
    first_before = first.evaluate("window.__matheelQueueJoins.length")
    second_before = second.evaluate("window.__matheelQueueJoins.length")
    with _hold_dataset_callback(dataset_callback_barriers, "evaluation") as held:
        first.get_by_role("button", name="Run Dataset Evaluation", exact=True).click()
        _wait_until(lambda: first.evaluate("true") and held["started"].exists())
        second.get_by_role("button", name="Run Dataset Evaluation", exact=True).click()
        second.wait_for_function("""({index, start}) => window.__matheelQueueJoins.slice(start)
            .some(join => join.fn_index === index && join.event_id)
        """, arg={"index": dependency["id"], "start": second_before})
        queued = second.evaluate("""({index, start}) => window.__matheelQueueJoins.slice(start)
            .find(join => join.fn_index === index)
        """, {"index": dependency["id"], "start": second_before})
        # Another session occupies this callback's sole worker. The old settings
        # were admitted, but have not started when the reset completes.
        assert not second.evaluate("""id => window.__matheelQueueEvents.some(
            event => event.event_id === id && event.msg === 'process_starts')
        """, queued["event_id"])
        _select_dropdown(second, gradio_config, "Metric Preset", "Jaro-Winkler")
        _wait_for_text(second.locator(f"#component-{dependency['outputs'][7]}"), "Settings changed")
        held["release"].touch()
        _wait_for_dataset_action_completion(first, gradio_config, "Run Dataset Evaluation", first_before)
        _wait_for_dataset_action_completion(second, gradio_config, "Run Dataset Evaluation", second_before)

    assert not second.locator(f"#component-{dependency['outputs'][0]}").is_visible()
    assert not second.locator("#dataset-next-actions").is_visible()
    assert not second.get_by_role("button", name="Tune Pair Threshold", exact=True).is_visible()
    artifacts = _visible_component(second, gradio_config, label="Leaderboard Artifacts", component_type="file")
    assert artifacts.locator("a[download]").count() == 0
    _wait_for_text(_action_output(first, gradio_config, "Run Dataset Evaluation"), "Lexical Only")
    second.get_by_role("button", name="Run Dataset Evaluation", exact=True).click()
    _wait_for_text(_action_output(second, gradio_config, "Run Dataset Evaluation"), "Jaro-Winkler")


@pytest.mark.parametrize("stage", ["threshold", "explanation"])
def test_completed_old_dataset_stage_cannot_restore_results_after_option_reset(
    gradio_page, gradio_config, normalized_dataset_zip, dataset_callback_barriers, stage,
):
    page = gradio_page
    _open_tab(page, "Datasets")
    _upload(page, gradio_config, "Normalized Dataset ZIP", normalized_dataset_zip)
    _select_dropdown(page, gradio_config, "Metric Preset", "Lexical Only")
    page.get_by_role("button", name="Validate Dataset", exact=True).click()
    _wait_for_text(_action_output(page, gradio_config, "Validate Dataset"), "browser_pair_dataset")
    run = page.get_by_role("button", name="Run Dataset Evaluation", exact=True)
    _wait_until(run.is_enabled)
    run.click()
    summary = _action_output(page, gradio_config, "Run Dataset Evaluation")
    _wait_for_text(summary, "browser_pair_dataset")
    if stage == "threshold":
        action, artifact_label = "Tune Pair Threshold", "Threshold Tuning Artifacts"
    else:
        page.get_by_text("Scored Pair Explanation", exact=True).click()
        action, artifact_label = "Explain Scored Pair", "Scored Pair Explanation Artifacts"
    before = page.evaluate("window.__matheelQueueJoins.length")
    with _hold_dataset_callback(dataset_callback_barriers, stage) as held:
        page.get_by_role("button", name=action, exact=True).click()
        _wait_until(lambda: page.evaluate("true") and held["started"].exists())
        if stage == "threshold":
            _select_dropdown_with_keyboard(page, gradio_config, "Threshold Tuning Metric", "precision")
        else:
            _select_dropdown_with_keyboard(page, gradio_config, "Segment Mode", "token")
        status_id = _button_dependency(gradio_config, "Run Dataset Evaluation")["outputs"][7]
        _wait_for_text(page.locator(f"#component-{status_id}"), "options changed")
        held["release"].touch()
        _wait_for_dataset_action_completion(page, gradio_config, action, before)
    stage_summary_id = _button_dependency(gradio_config, action)["outputs"][0]
    assert not page.locator(f"#component-{stage_summary_id}").is_visible()
    artifacts = _visible_component(page, gradio_config, label=artifact_label, component_type="file")
    assert artifacts.locator("a[download]").count() == 0
    assert summary.is_visible()
    _wait_for_text(summary, "browser_pair_dataset")
