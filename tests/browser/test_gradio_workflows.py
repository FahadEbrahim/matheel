import csv
import io
import re
import time
import zipfile
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
