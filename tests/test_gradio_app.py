import importlib.util
import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pytest

from matheel.datasets import available_dataset_presets, write_pair_dataset, write_retrieval_dataset
from matheel.lexical import available_lexical_metrics, lexical_metric_metadata


pytest.importorskip("gradio")
pytest.importorskip("gradio_huggingfacehub_search")


APP_PATH = Path(__file__).resolve().parents[1] / "gradio_app" / "app.py"
SPEC = importlib.util.spec_from_file_location("matheel_gradio_app", APP_PATH)
gradio_app = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gradio_app)


EXPECTED_GRADIO_TABS = (
    "Compare",
    "Collection",
    "Suites",
    "Datasets",
    "Explain",
    "Dataset Map",
    "Pair Explanation",
    "Reports",
    "Official Benchmark",
    "Ready-made Leaderboard",
    "Build Leaderboard",
    "Inspect Artifacts",
)
EXPECTED_PRIMARY_ACTIONS = {
    "Run Pair": "calculate_similarity_gradio",
    "Run Collection": "get_sim_list_gradio",
    "Run Suite": "run_suite_gradio",
    "Validate Dataset": False,
    "Run Dataset Evaluation": False,
    "Generate Map": "generate_dataset_map_gradio",
    "Generate Explanation": "generate_pair_explanation_gradio",
    "Run Custom Leaderboard": "run_ready_leaderboard_gradio",
    "Inspect Leaderboard": "inspect_leaderboard_artifacts_gradio",
}


def test_gradio_shell_copy_is_guided_and_html_safe():
    header = gradio_app.app_header_html()
    intro = gradio_app.workflow_intro_html("<kicker>", "<title>", "<description>", "<outcome>")

    assert "Recommended workflow" in header
    assert "Start with two snippets" in header
    assert "<kicker>" not in intro
    assert "&lt;kicker&gt;" in intro
    assert "&lt;title&gt;" in intro
    assert "&lt;description&gt;" in intro
    assert "&lt;outcome&gt;" in intro


def test_gradio_ui_keeps_core_workflow_tabs_and_primary_actions_wired():
    config = gradio_app.demo.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    tab_labels = tuple(
        component["props"]["label"]
        for component in components.values()
        if component["type"] == "tabitem"
    )

    assert tab_labels == EXPECTED_GRADIO_TABS

    buttons = {
        component["props"].get("value"): component_id
        for component_id, component in components.items()
        if component["type"] == "button"
    }
    for button_text, api_name in EXPECTED_PRIMARY_ACTIONS.items():
        button_id = buttons[button_text]
        click_dependencies = [
            dependency
            for dependency in config["dependencies"]
            if (button_id, "click") in dependency["targets"]
        ]

        assert len(click_dependencies) == 1
        assert click_dependencies[0]["backend_fn"] is True
        assert click_dependencies[0]["api_name"] == api_name


def test_dataset_public_api_contracts_preserve_legacy_shape_and_order():
    config = gradio_app.demo.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    contracts = {
        "validate_dataset_gradio": {
            "inputs": ["Normalized Dataset ZIP", "Task"],
            "outputs": ["html", "dataframe", "html", "file"],
        },
        "evaluate_dataset_gradio_with_state": {
            "inputs": [
                "Normalized Dataset ZIP",
                "Task",
                "Metric Preset",
                "Embedding Model",
                "Vector Backend",
                "Runtime Device",
                "Preprocessing Mode",
                "Code Language",
                "Lexical Tokenizer",
                "Pair Threshold",
                "Retrieval k",
                "K-fold Resampling Folds (0 = off)",
                "Resampling Seed",
            ],
            "outputs": ["html", "dataframe", "dataframe", "dataframe", "dataframe", "file"],
        },
    }

    for api_name, expected in contracts.items():
        dependencies = [
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == api_name
        ]
        assert len(dependencies) == 1
        dependency = dependencies[0]
        assert dependency["show_api"] is True
        assert [
            components[component_id]["props"].get("label")
            for component_id in dependency["inputs"]
        ] == expected["inputs"]
        assert [components[component_id]["type"] for component_id in dependency["outputs"]] == (
            expected["outputs"]
        )

    signature = inspect.signature(gradio_app.evaluate_dataset_gradio_with_state)
    assert list(signature.parameters) == [
        "dataset_file",
        "task_label",
        "metric_preset",
        "model_name",
        "vector_backend",
        "runtime_device",
        "preprocess_mode",
        "code_language",
        "lexical_tokenizer",
        "threshold",
        "retrieval_k",
        "resampling_folds",
        "resampling_seed",
        "progress",
    ]


def test_dataset_public_api_direct_empty_calls_keep_output_types_without_validation():
    validation_outputs = gradio_app.validate_dataset_gradio(
        None,
        "Pair Classification",
        progress=None,
    )
    evaluation_outputs = gradio_app.evaluate_dataset_gradio_with_state(
        None,
        "Pair Classification",
        "Lexical Only",
        gradio_app.DEFAULT_MODEL,
        "auto",
        "auto",
        "none",
        "python",
        "raw",
        0.5,
        10,
        0,
        7,
        progress=None,
    )

    assert len(validation_outputs) == 4
    assert isinstance(validation_outputs[0], str)
    assert isinstance(validation_outputs[1], pd.DataFrame)
    assert isinstance(validation_outputs[2], str)
    assert validation_outputs[3] is None
    assert len(evaluation_outputs) == 6
    assert isinstance(evaluation_outputs[0], str)
    assert all(isinstance(output, pd.DataFrame) for output in evaluation_outputs[1:5])
    assert evaluation_outputs[5] is None


def test_gradio_overflow_tabs_have_live_accessibility_metadata():
    app_js = gradio_app.demo.get_config_file()["js"]

    assert 'querySelectorAll(".overflow-menu")' in app_js
    assert '"aria-label", "More workflow tabs"' in app_js
    assert '"aria-haspopup", "menu"' in app_js
    assert '"aria-expanded"' in app_js
    assert 'dropdown.classList.contains("hide")' in app_js
    assert "MutationObserver" in app_js
    assert 'dropdown.setAttribute("role", "menu")' in app_js
    assert 'item.setAttribute("role", "menuitem")' in app_js


def test_gradio_config_exposes_guided_dataset_stages_and_correct_tsed_order():
    config = gradio_app.demo.get_config_file()
    components = config["components"]
    labels = [component["props"].get("label") for component in components]

    assert "Insert, Delete, Rename Costs" not in labels
    assert labels.count("Delete, Insert, Rename Costs") == 3
    assert any(
        component["props"].get("elem_id") == "dataset-artifact-workspace"
        and component["props"].get("label") == "Artifact Workspace"
        for component in components
    )
    assert gradio_app.dataset_run.interactive is False
    assert gradio_app.dataset_threshold_tune.visible is False
    assert gradio_app.dataset_pair_explanation_actions.visible is False
    assert gradio_app.dataset_next_actions.visible is False
    setup_ids = {
        gradio_app.dataset_file._id,
        gradio_app.dataset_task._id,
        gradio_app.dataset_validate._id,
        gradio_app.dataset_run._id,
    }
    component_order = [component["id"] for component in components]
    assert max(
        component_order.index(component_id) for component_id in setup_ids
    ) < component_order.index(gradio_app.dataset_validation_summary._id)
    assert all(
        component.visible is False
        for component in (
            gradio_app.dataset_summary,
            gradio_app.dataset_metrics,
            gradio_app.dataset_scores,
            gradio_app.dataset_threshold_summary,
            gradio_app.dataset_pair_explain_summary,
        )
    )


def test_dataset_upstream_events_clear_stale_dependent_state():
    config = gradio_app.demo.get_config_file()
    state_id = gradio_app.dataset_scored_state._id
    validation_state_id = gradio_app.dataset_validation_state._id

    reset_dependencies = [
        dependency
        for dependency in config["dependencies"]
        if "reset_dataset" in str(dependency.get("api_name"))
    ]
    assert reset_dependencies
    assert all(dependency["show_api"] is False for dependency in reset_dependencies)

    for component in (
        gradio_app.dataset_metric_preset,
        gradio_app.dataset_model,
        gradio_app.dataset_vector_backend,
        gradio_app.dataset_runtime_device,
        gradio_app.dataset_preprocess_mode,
        gradio_app.dataset_code_language,
        gradio_app.dataset_lexical_tokenizer,
        gradio_app.dataset_threshold,
        gradio_app.dataset_k,
        gradio_app.dataset_resampling_folds,
        gradio_app.dataset_resampling_seed,
    ):
        assert any(
            (component._id, "input") in dependency["targets"] and state_id in dependency["outputs"]
            for dependency in config["dependencies"]
        )

    for component in (gradio_app.dataset_file, gradio_app.dataset_task):
        assert any(
            (component._id, "change") in dependency["targets"]
            and state_id in dependency["outputs"]
            and validation_state_id in dependency["outputs"]
            for dependency in config["dependencies"]
        )

    assert any(
        (gradio_app.dataset_threshold_optimize._id, "input") in dependency["targets"]
        and gradio_app.dataset_threshold_artifacts._id in dependency["outputs"]
        for dependency in config["dependencies"]
    )
    for component in (
        gradio_app.dataset_pair_explain_row,
        gradio_app.dataset_pair_explain_segment_mode,
        gradio_app.dataset_pair_explain_high_threshold,
        gradio_app.dataset_pair_explain_medium_threshold,
        gradio_app.dataset_pair_explain_low_threshold,
        gradio_app.dataset_pair_explain_chunk_size,
    ):
        assert any(
            (component._id, "input") in dependency["targets"]
            and gradio_app.dataset_pair_explain_artifacts._id in dependency["outputs"]
            for dependency in config["dependencies"]
        )


def test_dataset_validation_and_upload_resets_share_one_serial_listener():
    config = gradio_app.demo.get_config_file()
    targets = {
        (gradio_app.dataset_file._id, "change"),
        (gradio_app.dataset_task._id, "change"),
        (gradio_app.dataset_validate._id, "click"),
    }
    dependencies = [
        dependency
        for dependency in config["dependencies"]
        if targets.intersection(map(tuple, dependency["targets"]))
        and gradio_app.dataset_validation_summary._id in dependency["outputs"]
    ]

    assert len(dependencies) == 1
    assert set(map(tuple, dependencies[0]["targets"])) == targets
    assert dependencies[0]["trigger_mode"] == "multiple"
    assert gradio_app.demo.fns[dependencies[0]["id"]].concurrency_limit == 1


def test_dataset_upload_event_dispatch_keeps_validation_and_reset_outputs_aligned(monkeypatch):
    monkeypatch.setattr(
        gradio_app,
        "validate_dataset_journey_gradio",
        lambda *args, **kwargs: (
            "validated summary",
            "issues",
            "report",
            "archive",
            "identity",
            "enabled",
            "status",
        ),
    )
    validated = gradio_app.update_dataset_upload_journey(
        "dataset.zip",
        "Pair Classification",
        gradio_app.gr.EventData(gradio_app.dataset_validate, None),
        progress=None,
    )
    assert validated[gradio_app.dataset_validation_summary] == "validated summary"
    assert validated[gradio_app.dataset_validation_state] == "identity"
    assert validated[gradio_app.dataset_run] == "enabled"
    assert len(validated) == 27
    assert validated[gradio_app.dataset_scored_state] is None
    assert validated[gradio_app.dataset_summary]["visible"] is False
    assert validated[gradio_app.dataset_latest_revision] == 0

    for target in (gradio_app.dataset_file, gradio_app.dataset_task):
        reset = gradio_app.update_dataset_upload_journey(
            None,
            "Pair Classification",
            gradio_app.gr.EventData(target, None),
            progress=None,
        )
        assert "No dataset checked" in reset[gradio_app.dataset_validation_summary]
        assert reset[gradio_app.dataset_validation_state] is None
        assert reset[gradio_app.dataset_run]["interactive"] is False
        assert reset[gradio_app.dataset_scored_state] is None
        assert reset[gradio_app.dataset_summary]["visible"] is False


def test_late_reset_keeps_validation_for_the_same_upload_and_task(monkeypatch):
    monkeypatch.setattr(gradio_app, "dataset_upload_identity", lambda *args: "current-identity")
    event = gradio_app.gr.EventData(gradio_app.dataset_file, None)
    assert gradio_app.update_dataset_upload_journey(
        "dataset.zip", "Pair Classification", event,
        validation_state="current-identity", progress=None,
    ) == {gradio_app.dataset_latest_revision: 0}
    reset = gradio_app.update_dataset_upload_journey(
        "dataset.zip", "Pair Classification", event,
        validation_state="older-identity", progress=None,
    )
    assert reset[gradio_app.dataset_validation_state] is None
    assert reset[gradio_app.dataset_run]["interactive"] is False


@pytest.mark.parametrize("target_name", ["dataset_file", "dataset_task", "dataset_validate"])
def test_older_client_revision_cannot_replace_a_newer_dataset_event(target_name, monkeypatch):
    def stale_validation(*args, **kwargs):
        raise AssertionError("A superseded validation must not execute")

    monkeypatch.setattr(gradio_app, "validate_dataset_journey_gradio", stale_validation)
    event = gradio_app.gr.EventData(getattr(gradio_app, target_name), None)
    assert gradio_app.update_dataset_upload_journey(
        None, "Pair Classification", event,
        validation_state={"sha256": "new upload", "task": "pair"},
        event_revision=1, latest_revision=2, progress=None,
    ) == gradio_app.gr.skip()


def test_dataset_validation_input_error_returns_data_and_clears_stale_results(monkeypatch):
    def fail_validation(*args, **kwargs):
        raise gradio_app.gr.Error("Invalid <dataset>")

    monkeypatch.setattr(gradio_app, "validate_dataset_journey_gradio", fail_validation)
    outputs = gradio_app.update_dataset_upload_journey(
        "dataset.zip",
        "Pair Classification",
        gradio_app.gr.EventData(gradio_app.dataset_validate, None),
        progress=None,
    )
    assert "Invalid &lt;dataset&gt;" in outputs[gradio_app.dataset_validation_summary]
    assert "Invalid &lt;dataset&gt;" in outputs[gradio_app.dataset_journey_status]
    assert outputs[gradio_app.dataset_validation_state] is None
    assert outputs[gradio_app.dataset_run]["interactive"] is False
    assert outputs[gradio_app.dataset_scored_state] is None
    assert outputs[gradio_app.dataset_summary]["visible"] is False


def test_dataset_next_actions_share_inputs_and_select_destination_tabs():
    explain_outputs = gradio_app.handoff_dataset_to_explain(
        "dataset.zip",
        "Pair Classification",
    )
    reports_outputs = gradio_app.handoff_dataset_to_reports("leaderboard_artifacts.zip")

    assert explain_outputs[:2] == ("dataset.zip", "Pair Classification")
    assert explain_outputs[2]["selected"] == "explain"
    assert explain_outputs[3]["selected"] == "dataset-map"
    assert reports_outputs[0] == "leaderboard_artifacts.zip"
    assert reports_outputs[1]["selected"] == "reports"
    assert reports_outputs[2]["selected"] == "inspect-artifacts"

    config = gradio_app.demo.get_config_file()
    expected = {
        gradio_app.dataset_continue_explain._id: [
            gradio_app.map_dataset_file._id,
            gradio_app.map_task._id,
            gradio_app.workflow_tabs._id,
            gradio_app.explain_tabs._id,
        ],
        gradio_app.dataset_continue_reports._id: [
            gradio_app.leaderboard_file._id,
            gradio_app.workflow_tabs._id,
            gradio_app.reports_tabs._id,
        ],
    }
    for button_id, output_ids in expected.items():
        dependencies = [
            dependency
            for dependency in config["dependencies"]
            if (button_id, "click") in dependency["targets"]
        ]
        assert len(dependencies) == 1
        assert dependencies[0]["api_name"] is False
        assert dependencies[0]["outputs"] == output_ids


def test_gradio_app_launches_and_serves_root_and_config():
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        server_port = port_socket.getsockname()[1]

    _, local_url, _ = gradio_app.demo.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        prevent_thread_lock=True,
        quiet=True,
        show_error=True,
    )
    try:
        with urlopen(local_url, timeout=10) as response:
            root_html = response.read().decode("utf-8")
            assert response.status == 200
        with urlopen(f"{local_url}config", timeout=10) as response:
            live_config = json.load(response)
            assert response.status == 200
    finally:
        gradio_app.demo.close()

    assert "<gradio-app" in root_html
    assert "Matheel Framework" in root_html
    assert live_config["title"] == "Matheel Framework"
    assert len(live_config["components"]) == len(gradio_app.demo.get_config_file()["components"])


def test_gradio_temp_workspace_cleanup_removes_only_stale_known_prefixes(tmp_path):
    old_workspace = tmp_path / "matheel-suite-old"
    fresh_workspace = tmp_path / "matheel-suite-fresh"
    unrelated = tmp_path / "other-old"
    old_workspace.mkdir()
    fresh_workspace.mkdir()
    unrelated.mkdir()
    os.utime(old_workspace, (1, 1))
    os.utime(unrelated, (1, 1))

    cleaned = gradio_app.cleanup_stale_temp_workspaces(
        temp_root=tmp_path,
        ttl_seconds=60,
        prefixes=("matheel-suite-",),
    )

    assert cleaned == 1
    assert not old_workspace.exists()
    assert fresh_workspace.exists()
    assert unrelated.exists()


def test_gradio_temp_workspace_creation_runs_cleanup(tmp_path, monkeypatch):
    calls = []

    def fake_cleanup(temp_root=None):
        calls.append(temp_root)
        return 0

    monkeypatch.setattr(gradio_app, "cleanup_stale_temp_workspaces", fake_cleanup)

    workspace = gradio_app.make_temp_workspace("matheel-suite-", temp_root=tmp_path)

    assert calls == [tmp_path]
    assert workspace.exists()
    assert workspace.parent == tmp_path


def _build_suite_row(**overrides):
    kwargs = {
        "rows": gradio_app.empty_suite_rows(),
        "run_name": "",
        "selected_features": ["Levenshtein"],
        "model_name": gradio_app.DEFAULT_MODEL,
        "vector_backend": "auto",
        "similarity_function": "cosine",
        "pooling_method": "mean",
        "max_token_length": 256,
        "runtime_device": "auto",
        "semantic_weight": 0.0,
        "levenshtein_weight": 1.0,
        "jaro_winkler_weight": 0.0,
        "winnowing_weight": 0.0,
        "gst_weight": 0.0,
        "levenshtein_weights": "1,1,1",
        "jaro_winkler_prefix_weight": 0.1,
        "winnowing_kgram": 5,
        "winnowing_window": 4,
        "gst_min_match_length": 5,
        "lexical_tokenizer": "raw",
        "code_metric": "codebleu",
        "code_metric_weight": 0.0,
        "code_language": "java",
        "codebleu_component_weights": "0.25,0.25,0.25,0.25",
        "crystalbleu_max_order": 4,
        "crystalbleu_trivial_ngram_count": 50,
        "ruby_mode": gradio_app.DEFAULT_RUBY_MODE,
        "ruby_graph_timeout_seconds": gradio_app.DEFAULT_RUBY_GRAPH_TIMEOUT,
        "tsed_costs": gradio_app.DEFAULT_TSED_COSTS,
        "codebertscore_model": gradio_app.DEFAULT_CODEBERTSCORE_MODEL,
        "codebertscore_max_length": gradio_app.DEFAULT_CODEBERTSCORE_MAX_LENGTH,
        "selected_preparation": [],
        "preprocess_mode": "none",
        "chunking_method": "none",
        "chunk_size": 120,
        "chunk_overlap": 0,
        "max_chunks": 0,
        "chunk_aggregation": "mean",
        "chunk_language": "text",
        "chunker_options": "",
        "threshold": 0.0,
        "number_results": 50,
    }
    kwargs.update(overrides)
    return gradio_app.build_suite_run_row_data(**kwargs)


def test_suite_threshold_preserves_zero():
    row = _build_suite_row(threshold=0.0)

    assert row["threshold"] == 0.0

    configs = gradio_app.suite_rows_to_configs([row])

    assert configs[0]["options"]["threshold"] == 0.0


def test_suite_preserves_device_and_valid_zero_metric_controls():
    row = _build_suite_row(
        selected_features=["Jaro-Winkler", "Code Metric"],
        levenshtein_weight=0.0,
        jaro_winkler_weight=0.5,
        code_metric="crystalbleu",
        code_metric_weight=0.5,
        runtime_device="cpu",
        jaro_winkler_prefix_weight=0.0,
        crystalbleu_trivial_ngram_count=0,
    )

    config = gradio_app.suite_rows_to_configs([row])[0]["options"]

    assert row["runtime_device"] == "cpu"
    assert config["device"] == "cpu"
    assert config["jaro_winkler_prefix_weight"] == 0.0
    assert config["crystalbleu_trivial_ngram_count"] == 0


def test_suite_rejects_zero_ruby_timeout_instead_of_defaulting_it():
    with pytest.raises(gradio_app.gr.Error, match="at least 0.1 seconds"):
        _build_suite_row(
            selected_features=["Code Metric"],
            levenshtein_weight=0.0,
            code_metric="ruby",
            code_metric_weight=1.0,
            ruby_graph_timeout_seconds=0.0,
        )


def test_empty_active_metric_selection_is_rejected():
    with pytest.raises(gradio_app.gr.Error, match="Select at least one active metric"):
        _build_suite_row(selected_features=[], levenshtein_weight=0.0)


def test_tsed_costs_map_delete_insert_rename_in_display_order():
    kwargs = gradio_app.resolve_metric_kwargs(
        "tsed",
        gradio_app.DEFAULT_RUBY_MODE,
        gradio_app.DEFAULT_RUBY_GRAPH_TIMEOUT,
        "1,2,3",
        gradio_app.DEFAULT_CODEBERTSCORE_MODEL,
        gradio_app.DEFAULT_CODEBERTSCORE_MAX_LENGTH,
    )

    assert kwargs == {
        "tsed_delete_cost": 1.0,
        "tsed_insert_cost": 2.0,
        "tsed_rename_cost": 3.0,
    }


def test_shared_scoring_bundles_keep_the_same_handler_order():
    bundles = (
        gradio_app.pair_controls,
        gradio_app.collection_controls,
        gradio_app.suite_controls,
    )

    assert all(isinstance(bundle, gradio_app.ScoringControls) for bundle in bundles)
    labels = [
        [getattr(component, "label", None) for component in bundle.handler_inputs()]
        for bundle in bundles
    ]
    assert labels[0] == labels[1] == labels[2]
    assert labels[0][5:8] == ["Max Tokens per Input", "Runtime Device", "Embedding Weight"]


def test_suite_rows_preserve_lexical_tokenizer():
    row = _build_suite_row(
        selected_features=["Winnowing"],
        levenshtein_weight=0.0,
        winnowing_weight=1.0,
        lexical_tokenizer="parser",
    )

    configs = gradio_app.suite_rows_to_configs([row])

    assert row["lexical_tokenizer"] == "parser"
    assert configs[0]["options"]["lexical_tokenizer"] == "parser"


def test_default_suite_run_name_uses_algorithm_names():
    rows = gradio_app.empty_suite_rows()

    assert (
        gradio_app.default_suite_run_name(
            rows,
            ["Levenshtein"],
            [],
            gradio_app.DEFAULT_MODEL,
            "auto",
            "cosine",
            "mean",
            "codebleu",
            "none",
            "none",
        )
        == "levenshtein"
    )
    assert (
        gradio_app.default_suite_run_name(
            rows,
            ["Code Metric"],
            [],
            gradio_app.DEFAULT_MODEL,
            "auto",
            "cosine",
            "mean",
            "codebleu",
            "none",
            "none",
        )
        == "codebleu"
    )
    assert (
        gradio_app.default_suite_run_name(
            rows,
            ["Embedding", "Levenshtein"],
            [],
            gradio_app.DEFAULT_MODEL,
            "auto",
            "cosine",
            "mean",
            "codebleu",
            "none",
            "none",
        )
        == "embedding_levenshtein"
    )


def test_build_feature_weights_ignores_code_metric_weight_when_metric_is_inactive():
    weights = gradio_app.build_feature_weights(
        True,
        1.0,
        False,
        0.0,
        False,
        0.0,
        False,
        0.0,
        False,
        0.0,
        "none",
        1.0,
    )

    assert weights == {"semantic": 1.0}


def test_gradio_exposes_every_lexical_metric_and_preserves_relative_weights():
    assert {
        lexical_metric_metadata(name)["label"] for name in available_lexical_metrics()
    }.issubset(gradio_app.FEATURE_UI_CHOICES)
    label = lexical_metric_metadata("lcs")["label"]

    weights = gradio_app.selected_feature_weights(
        ["Levenshtein", label], 0.0, 3.0, 0.0, 0.0, 0.0, "none", 0.0,
        pd.DataFrame([[label, 1.0]], columns=["Metric", "Weight"]),
    )

    assert weights == {"levenshtein": 0.75, "lcs": 0.25}


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), "not a number"])
def test_additional_lexical_weights_reject_invalid_values(weight):
    with pytest.raises(gradio_app.gr.Error, match="finite non-negative"):
        gradio_app.parse_additional_lexical_weights({"lcs": weight})


def test_weight_normalization_handles_overflow_and_rejects_all_zero():
    assert gradio_app.normalize_feature_weights_map(
        {"lcs": 1e308, "levenshtein": 1e308}
    ) == {"lcs": 0.5, "levenshtein": 0.5}
    with pytest.raises(gradio_app.gr.Error, match="positive weight"):
        gradio_app.normalize_feature_weights_map({"lcs": 0.0})
    with pytest.raises(gradio_app.gr.Error, match="non-negative"):
        gradio_app.normalize_feature_weights_map({"lcs": -1.0})


def test_additional_lexical_weight_table_retains_selected_weights_and_defaults_new_rows():
    label = lexical_metric_metadata("lcs")["label"]
    ratio_label = lexical_metric_metadata("rapidfuzz_ratio")["label"]
    frame = gradio_app.additional_lexical_weights_frame(
        [label, ratio_label], {"lcs": 2.5, "osa": 7.0}
    )

    assert gradio_app.parse_additional_lexical_weights(frame) == {"lcs": 2.5, "rapidfuzz_ratio": 1.0}
    assert gradio_app.additional_lexical_weights_frame(["Levenshtein"]).empty
    assert gradio_app.update_additional_lexical_weights([], frame)["visible"] is False


def test_suite_roundtrips_additional_weights_and_loads_old_positional_rows():
    label = lexical_metric_metadata("lcs")["label"]
    row = _build_suite_row(
        selected_features=["Levenshtein", label],
        levenshtein_weight=3.0,
        lexical_weights={"lcs": 1.0},
    )

    assert json.loads(row["lexical_weights"]) == {"lcs": 0.25}
    options = gradio_app.suite_rows_to_configs([row])[0]["options"]
    assert options["feature_weights"] == {"levenshtein": 0.75, "lcs": 0.25}
    old_row = _build_suite_row()
    old_values = [[old_row[column] for column in gradio_app.SUITE_COLUMNS[:-1]]]
    assert gradio_app.suite_rows_to_configs(old_values)[0]["options"]["feature_weights"] == {
        "levenshtein": 1.0
    }


def _scoring_handler_kwargs():
    parameters = list(inspect.signature(gradio_app.calculate_similarity_gradio).parameters)[2:]
    return {
        name: component.value
        for name, component in zip(parameters, gradio_app.pair_controls.handler_inputs())
    }


def test_pair_api_runs_lcs_with_legacy_inputs_and_appended_weights():
    options = _scoring_handler_kwargs()
    options["selected_features"] = [lexical_metric_metadata("lcs")["label"]]

    summary = gradio_app.calculate_similarity_gradio(
        "abcdef", "ace", **options, progress=None, lexical_weights={"lcs": 2.0}
    )

    assert "0.5000" in summary
    assert list(inspect.signature(gradio_app.calculate_similarity_gradio).parameters)[-2:] == [
        "progress", "lexical_weights"
    ]


def test_collection_api_runs_new_metric_with_appended_weights(tmp_path):
    archive_path = tmp_path / "codes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "abcdef")
        archive.writestr("b.py", "ace")
    options = _scoring_handler_kwargs()
    options["selected_features"] = [lexical_metric_metadata("lcs")["label"]]

    _, results = gradio_app.get_sim_list_gradio(
        archive_path, **options, threshold=0.0, number_results=10,
        progress=None, lexical_weights={"lcs": 1.0},
    )

    assert results.iloc[0]["similarity_score"] == 0.5


def test_pair_collection_and_saved_suite_share_effective_scoring_options(monkeypatch):
    captured = {}

    def pair(*args, **kwargs):
        captured["pair"] = kwargs
        return 0.5

    def collection(*args, **kwargs):
        captured["collection"] = kwargs
        return pd.DataFrame(columns=["file_name_1", "file_name_2", "similarity_score"])

    monkeypatch.setattr(gradio_app, "calculate_similarity", pair)
    monkeypatch.setattr(gradio_app, "get_sim_list", collection)
    controls = _scoring_handler_kwargs()
    controls.update(
        selected_features=["Levenshtein", "Code Metric", lexical_metric_metadata("lcs")["label"]],
        levenshtein_weight=2.0, code_metric="tsed", code_metric_weight=3.0,
        levenshtein_weights="1,2,3", tsed_costs="0,2,3", runtime_device="cpu",
        lexical_weights={"lcs": 5.0},
    )
    gradio_app.calculate_similarity_gradio("left", "right", **controls, progress=None)
    gradio_app.get_sim_list_gradio("collection.zip", **controls, threshold=0.2, number_results=10, progress=None)
    row = _build_suite_row(**controls, threshold=0.2, number_results=10)
    saved = gradio_app.suite_rows_to_configs([row])[0]["options"]

    pair_options = {key: value for key, value in captured["pair"].items() if key != "progress_callback"}
    collection_options = {
        key: value for key, value in captured["collection"].items()
        if key not in ("progress_callback", "threshold", "number_results")
    }
    saved_options = {key: value for key, value in saved.items() if key not in ("threshold", "number_results")}
    assert pair_options == collection_options == saved_options
    assert pair_options["feature_weights"] == {"levenshtein": 0.2, "code_metric": 0.3, "lcs": 0.5}
    assert pair_options["tsed_delete_cost"] == 0


def test_suite_api_saves_and_executes_new_metrics(tmp_path):
    options = _scoring_handler_kwargs()
    options["selected_features"] = [lexical_metric_metadata("lcs")["label"]]
    saved_rows, _, _, _ = gradio_app.append_suite_run_gradio(
        gradio_app.empty_suite_rows(), "lcs_test", **options,
        threshold=0.0, number_results=10, lexical_weights={"lcs": 4.0},
    )
    assert gradio_app.suite_rows_to_configs(saved_rows)[0]["options"]["feature_weights"] == {"lcs": 1.0}
    archive_path = tmp_path / "codes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "abcdef")
        archive.writestr("b.py", "ace")
    result = gradio_app.run_suite_gradio(
        archive_path, gradio_app.empty_suite_rows(), "csv", "lcs_test", **options,
        threshold=0.0, number_results=10, progress=None, lexical_weights={"lcs": 4.0},
    )

    assert result[3].iloc[0]["similarity_score"] == 0.5
    exported_configs = json.loads(Path(result[7]).read_text(encoding="utf-8"))
    assert exported_configs[0]["options"]["feature_weights"] == {"lcs": 1.0}


def test_metric_presets_retain_new_weights_for_dataset_scoring():
    preset_name = lexical_metric_metadata("lcs")["label"]
    preset = gradio_app.metric_preset_options(preset_name)
    assert preset["lexical_weights"] == {"lcs": 1.0}
    assert preset["features"] == [preset_name]
    updates = gradio_app.apply_metric_preset_gradio(preset_name)
    assert gradio_app.parse_additional_lexical_weights(updates[-1]["value"]) == {"lcs": 1.0}
    assert gradio_app.dataset_similarity_options(
        preset_name, gradio_app.DEFAULT_MODEL, "auto", "cpu", "none", "python"
    )["feature_weights"] == {"lcs": 1.0}


def test_scoring_actions_append_weight_table_without_changing_existing_control_order():
    config = gradio_app.demo.get_config_file()
    for api_name in (
        "calculate_similarity_gradio", "get_sim_list_gradio", "append_suite_run_gradio", "run_suite_gradio"
    ):
        dependency = next(item for item in config["dependencies"] if item["api_name"] == api_name)
        table_id = dependency["inputs"][-1]
        component = next(item for item in config["components"] if item["id"] == table_id)
        assert component["type"] == "dataframe"
        assert component["props"]["label"] == "Additional lexical weights"
        assert component["props"]["static_columns"] == [0]


def test_flat_space_checkout_imports_extracted_helpers(tmp_path):
    for name in ("app.py", "html_utils.py", "presentation.py", "theme.py", "control_types.py", "official_benchmark.py"):
        shutil.copyfile(APP_PATH.parent / name, tmp_path / name)
    (tmp_path / "assets").mkdir()
    shutil.copyfile(
        APP_PATH.parent / "assets" / "ready_leaderboard.json",
        tmp_path / "assets" / "ready_leaderboard.json",
    )
    script = """
import importlib.abc
import runpy
import sys
class FlatSpace(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gradio_app" or fullname.startswith("gradio_app."):
            raise ModuleNotFoundError("Flat Space has no gradio_app package", name=fullname)
sys.meta_path.insert(0, FlatSpace())
app = runpy.run_path("app.py", run_name="matheel_space_smoke")
assert app["ScoringControls"].__module__ == "control_types"
assert app["summary_panel_html"].__module__ == "presentation"
assert app["demo"].get_config_file()["mode"] == "blocks"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"},
        text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_status_html_helpers_escape_dynamic_values():
    profile_html = gradio_app.profile_status_html('<script>alert("x")</script>')

    assert "<script>" not in profile_html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in profile_html

    error_html = gradio_app.model_status_html(error_message="<img src=x>")

    assert "<img" not in error_html
    assert "&lt;img src=x&gt;" in error_html

    status_html = gradio_app.model_status_html(
        {
            "resolved_vector_backend": "<b>semantic</b>",
            "detected_max_token_length": 512,
            "configured_max_token_length": 256,
            "runtime_device": "<svg>",
        }
    )

    assert "<b>semantic</b>" not in status_html
    assert "&lt;b&gt;semantic&lt;/b&gt;" in status_html
    assert "&lt;svg&gt;" in status_html


def test_suite_html_helpers_escape_run_names():
    rows = pd.DataFrame([{"run_name": '<img src=x onerror="alert(1)">'}])

    overview_html = gradio_app.suite_runs_overview_html(rows)

    assert "<img" not in overview_html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in overview_html

    summary = pd.DataFrame(
        [
            {
                "run_name": "<b>best</b>",
                "mean_score": 0.8,
                "max_score": 0.9,
                "elapsed_seconds": 1.2345,
                "feature_set": "<i>semantic</i>",
            }
        ]
    )

    summary_html = gradio_app.suite_summary_html(summary)

    assert "<b>best</b>" not in summary_html
    assert "&lt;b&gt;best&lt;/b&gt;" in summary_html
    assert "1.234s" in summary_html
    assert "&lt;i&gt;semantic&lt;/i&gt;" in summary_html


def test_suite_summary_uses_all_statistics_from_the_selected_winner():
    summary = pd.DataFrame(
        [
            {
                "run_name": "selected-winner",
                "mean_score": 0.71,
                "max_score": 0.82,
                "elapsed_seconds": 0.432,
                "feature_set": "semantic",
            },
            {
                "run_name": "different-maxima",
                "mean_score": 0.99,
                "max_score": 1.0,
                "elapsed_seconds": 2.0,
                "feature_set": "lexical",
            },
        ]
    )

    summary_html = gradio_app.suite_summary_html(summary)

    assert "0.710" in summary_html
    assert "0.820" in summary_html
    assert "0.990" not in summary_html
    assert "1.000" not in summary_html


def test_results_summary_html_escapes_uploaded_file_names():
    results = pd.DataFrame(
        [
            {
                "file_name_1": "<b>a.py</b>",
                "file_name_2": "<script>x</script>",
                "similarity_score": 0.9,
            }
        ]
    )
    results.attrs["elapsed_seconds"] = 0.4321
    results.attrs["feature_set"] = "<i>levenshtein</i>"

    summary_html = gradio_app.results_summary_html(results, "auto", "none", "none", "cpu")

    assert "<b>a.py</b>" not in summary_html
    assert "<script>" not in summary_html
    assert "&lt;b&gt;a.py&lt;/b&gt;" in summary_html
    assert "&lt;script&gt;x&lt;/script&gt;" in summary_html
    assert "0.432s" in summary_html
    assert "&lt;i&gt;levenshtein&lt;/i&gt;" in summary_html


def test_score_card_html_displays_elapsed_time():
    score_html = gradio_app.score_card_html(0.75, elapsed_seconds=2.3456)

    assert "0.7500" in score_html
    assert "Interpretation" in score_html
    assert "High" in score_html
    assert "2.346s" in score_html
    assert "Pairwise Result" in score_html


def test_metric_presets_resolve_common_workflows():
    lexical = gradio_app.metric_preset_options("Lexical Only")
    code_aware = gradio_app.metric_preset_options("Code-Aware")
    jaro = gradio_app.metric_preset_options("Jaro-Winkler")

    assert lexical["features"] == ["Levenshtein", "Winnowing", "GST"]
    assert lexical["semantic_weight"] == 0.0
    assert code_aware["features"] == ["Embedding", "Levenshtein", "Code Metric"]
    assert code_aware["code_metric"] == "codebleu"
    assert code_aware["code_metric_weight"] == 0.25
    assert jaro["features"] == ["Jaro-Winkler"]
    assert "Winnowing" in gradio_app.READY_LEADERBOARD_ALGORITHM_CHOICES
    assert "CodeBLEU" in gradio_app.READY_LEADERBOARD_ALGORITHM_CHOICES


def test_empty_result_panels_use_readable_states():
    pair_html = gradio_app.empty_pair_summary_html()
    collection_html = gradio_app.empty_summary_html()
    suite_html = gradio_app.empty_suite_summary_html()

    assert "No comparison run" in pair_html
    assert "No collection run" in collection_html
    assert "No runs executed" in suite_html
    assert "matheel-empty" in pair_html
    assert "matheel-empty" in collection_html
    assert "matheel-empty" in suite_html


def test_pair_comparison_requires_both_snippets():
    with pytest.raises(gradio_app.gr.Error, match="Paste both snippets"):
        gradio_app.calculate_similarity_gradio(
            "",
            "print(1)",
            gradio_app.DEFAULT_FEATURE_SELECTION,
            gradio_app.DEFAULT_MODEL,
            "auto",
            "cosine",
            "mean",
            256,
            "auto",
            0.7,
            0.3,
            0.0,
            0.0,
            0.0,
            "1,1,1",
            0.1,
            5,
            4,
            5,
            "raw",
            "codebleu",
            0.0,
            "python",
            "0.25,0.25,0.25,0.25",
            4,
            50,
            gradio_app.DEFAULT_RUBY_MODE,
            gradio_app.DEFAULT_RUBY_GRAPH_TIMEOUT,
            gradio_app.DEFAULT_TSED_COSTS,
            gradio_app.DEFAULT_CODEBERTSCORE_MODEL,
            gradio_app.DEFAULT_CODEBERTSCORE_MAX_LENGTH,
            [],
            "none",
            "none",
            120,
            0,
            0,
            "mean",
            "text",
            "",
        )


def test_run_suite_export_sanitizes_detail_zip_filenames(monkeypatch):
    row = _build_suite_row(run_name="../baseline/strong")
    summary = pd.DataFrame(
        [
            {
                "run_name": "../baseline/strong",
                "pair_count": 1,
                "mean_score": 1.0,
                "median_score": 1.0,
                "max_score": 1.0,
                "min_score": 1.0,
                "std_score": 0.0,
                "top_file_1": "a.py",
                "top_file_2": "b.py",
                "top_score": 1.0,
                "vector_backend": "auto",
                "code_metric": "none",
                "chunking_method": "none",
            }
        ]
    )
    details = pd.DataFrame(
        [
            {
                "file_name_1": "a.py",
                "file_name_2": "b.py",
                "similarity_score": 1.0,
            }
        ]
    )

    def fake_run_comparison_suite(
        zipped_file,
        run_configs,
        summary_out=None,
        details_dir=None,
        output_format="csv",
        progress_callback=None,
    ):
        _ = zipped_file, summary_out, output_format, progress_callback
        assert details_dir is None
        assert run_configs[0]["run_name"] == "../baseline/strong"
        assert run_configs[0]["options"]["device"] == "auto"
        return summary, {"../baseline/strong": details}

    monkeypatch.setattr(gradio_app, "run_comparison_suite", fake_run_comparison_suite)

    outputs = gradio_app.run_suite_gradio(
        "codes.zip",
        [row],
        "csv",
        "",
        ["Levenshtein"],
        gradio_app.DEFAULT_MODEL,
        "auto",
        "cosine",
        "mean",
        256,
        "auto",
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        "1,1,1",
        0.1,
        5,
        4,
        5,
        "raw",
        "codebleu",
        0.0,
        "java",
        "0.25,0.25,0.25,0.25",
        4,
        50,
        gradio_app.DEFAULT_RUBY_MODE,
        gradio_app.DEFAULT_RUBY_GRAPH_TIMEOUT,
        gradio_app.DEFAULT_TSED_COSTS,
        gradio_app.DEFAULT_CODEBERTSCORE_MODEL,
        gradio_app.DEFAULT_CODEBERTSCORE_MAX_LENGTH,
        [],
        "none",
        "none",
        120,
        0,
        0,
        "mean",
        "text",
        "",
        0.0,
        50,
    )

    details_zip_path = outputs[6]
    with zipfile.ZipFile(details_zip_path) as archive:
        assert archive.namelist() == ["baseline_strong.csv"]
    run_configs = json.loads(Path(outputs[7]).read_text(encoding="utf-8"))
    assert run_configs[0]["options"]["device"] == "auto"


def test_suite_detail_filenames_are_unique_after_slugification():
    filenames = gradio_app.unique_suite_detail_filenames(
        ["baseline/strong", "baseline_strong", "BASELINE_STRONG", "../"],
    )

    assert filenames == [
        "baseline_strong.csv",
        "baseline_strong_2.csv",
        "BASELINE_STRONG_3.csv",
        "run.csv",
    ]
    assert len({name.casefold() for name in filenames}) == len(filenames)


def test_suite_detail_filenames_bound_long_unicode_names_and_collision_suffixes(tmp_path):
    long_name = "metric_" * 100
    unicode_name = "指標🚀metric_" * 100
    run_names = ["short", long_name, long_name, unicode_name, unicode_name + "other"]

    filenames = gradio_app.unique_suite_detail_filenames(run_names)

    assert filenames[0] == "short.csv"
    assert filenames == gradio_app.unique_suite_detail_filenames(run_names)
    assert len({name.casefold() for name in filenames}) == len(run_names)
    assert all(len(name.encode("utf-8")) <= 204 for name in filenames)
    for filename in filenames:
        (tmp_path / filename).write_text("score\n1.0\n", encoding="utf-8")


def test_suite_exports_all_string_metrics_with_full_generated_display_name(tmp_path):
    options = _scoring_handler_kwargs()
    options["selected_features"] = [
        lexical_metric_metadata(name)["label"] for name in available_lexical_metrics()
    ]
    options["levenshtein_weight"] = 1.0
    options["jaro_winkler_weight"] = 1.0
    archive_path = tmp_path / "codes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "abcdef")
        archive.writestr("b.py", "ace")

    outputs = gradio_app.run_suite_gradio(
        archive_path, gradio_app.empty_suite_rows(), "csv", "", **options,
        threshold=0.0, number_results=10, progress=None,
        lexical_weights={name: 1.0 for name in gradio_app.ADDITIONAL_LEXICAL_METRICS},
    )

    config = json.loads(Path(outputs[7]).read_text(encoding="utf-8"))[0]
    assert len(config["run_name"]) > 255
    assert outputs[1].iloc[0]["run_name"] == config["run_name"]
    assert set(config["options"]["feature_weights"]) == set(available_lexical_metrics())
    with zipfile.ZipFile(outputs[6]) as archive:
        assert len(archive.namelist()) == 1
        filename = archive.namelist()[0]
        assert len(filename.encode("utf-8")) <= 204
        assert len(pd.read_csv(archive.open(filename))) == 1


def _zip_directory(source_root, zip_path):
    source_root = Path(source_root)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=Path(source_root.name) / path.relative_to(source_root))
    return zip_path


def test_dataset_pair_evaluation_exports_leaderboard_artifacts(tmp_path):
    dataset_root = tmp_path / "pair_dataset"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
                {"file_id": "c", "text": "print(2)", "suffix": ".py"},
                {"file_id": "d", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "c", "right_id": "d", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
                {"left_id": "b", "right_id": "d", "label": 0},
            ]
        ),
        metadata={"name": "tiny_pairs"},
    )
    archive_path = _zip_directory(dataset_root, tmp_path / "pair_dataset.zip")

    outputs = gradio_app.evaluate_dataset_gradio_with_state(
        archive_path,
        "Pair Classification",
        "Lexical Only",
        gradio_app.DEFAULT_MODEL,
        "auto",
        "auto",
        "none",
        "python",
        "raw",
        0.5,
        10,
        2,
        7,
        progress=None,
    )

    (
        summary_html,
        metrics_frame,
        scored_frame,
        resample_metrics,
        resample_summary,
        artifacts_path,
    ) = outputs
    assert "Dataset Evaluation" in summary_html
    assert "tiny_pairs" in summary_html
    assert "F1" in metrics_frame["Metric"].tolist()
    assert "Interpretation" in scored_frame.columns
    assert not resample_metrics.empty
    assert not resample_summary.empty

    with zipfile.ZipFile(artifacts_path) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "leaderboard_manifest.json" in names
        assert "leaderboard_report.json" in names
        assert "pair_scored_rows.csv" in names
        assert "pair_metrics.json" in names
        assert "pair_resampling_summary.csv" in names
        assert "pair_threshold_tuning_threshold_sweep.csv" in names
        manifest = json.loads(archive.read("leaderboard_manifest.json").decode("utf-8"))
        report = json.loads(archive.read("leaderboard_report.json").decode("utf-8"))
    assert manifest["workflow"] == "gradio_dataset_evaluation"
    assert manifest["dataset_kind"] == "pair_classification"
    assert report["metadata"]["name"] == "tiny_pairs Dataset Evaluation"
    assert {row["algorithm_name"] for row in report["aggregate"]} == {"Lexical Only"}

    inspection = gradio_app.inspect_leaderboard_artifacts_gradio(artifacts_path)
    assert "tiny_pairs Dataset Evaluation" in inspection[0]
    assert set(inspection[1]["algorithm_name"]) == {"Lexical Only"}
    assert set(inspection[2]["dataset_name"]) == {"tiny_pairs"}


def test_dataset_display_rounding_does_not_change_interpretation_or_state(monkeypatch, tmp_path):
    raw_score = 0.84996
    display = gradio_app.dataset_scores_display_frame(
        pd.DataFrame(
            [
                {
                    "left_id": "a",
                    "right_id": "b",
                    "label": 1,
                    "similarity_score": raw_score,
                }
            ]
        ),
        "Pair Classification",
    )

    assert display.loc[0, "Similarity Score"] == 0.85
    assert display.loc[0, "Interpretation"].startswith("High:")
    assert display.attrs["raw_scored_records"][0]["similarity_score"] == raw_score

    monkeypatch.setattr(
        gradio_app,
        "evaluate_dataset_gradio_with_state",
        lambda *args, **kwargs: (
            "summary",
            pd.DataFrame(),
            display,
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        ),
    )
    monkeypatch.setattr(gradio_app, "_dataset_root_from_upload", lambda *args: tmp_path)

    outputs = gradio_app.evaluate_dataset_gradio_for_journey(
        "dataset.zip", "Pair Classification"
    )

    assert outputs[-1]["scored"][0]["similarity_score"] == raw_score


def test_gradio_dataset_validation_exports_report(tmp_path):
    dataset_root = tmp_path / "pair_dataset"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
        metadata={"name": "validation_pairs"},
    )
    archive_path = _zip_directory(dataset_root, tmp_path / "pair_dataset.zip")

    summary_html, issues_frame, report_html, artifacts_path = gradio_app.validate_dataset_gradio(
        archive_path,
        "Pair Classification",
        progress=None,
    )

    assert "Dataset Validation" in summary_html
    assert "validation_pairs" in summary_html
    assert "single_class_labels" in issues_frame["Code"].tolist()
    assert "Matheel Dataset Validation" in report_html
    with zipfile.ZipFile(artifacts_path) as archive:
        assert "dataset_validation_report.json" in archive.namelist()

    journey_outputs = gradio_app.validate_dataset_journey_gradio(
        archive_path,
        "Pair Classification",
        progress=None,
    )
    assert journey_outputs[4] == gradio_app.dataset_upload_identity(
        archive_path,
        "Pair Classification",
    )
    assert journey_outputs[5]["interactive"] is True
    assert "run the dataset evaluation" in journey_outputs[6]


def test_dataset_evaluation_requires_matching_successful_validation(monkeypatch, tmp_path):
    dataset_file = tmp_path / "dataset.zip"
    dataset_file.write_bytes(b"dataset")
    validation_state = gradio_app.dataset_upload_identity(dataset_file, "Pair Classification")

    with pytest.raises(gradio_app.gr.Error, match="Validate this dataset and task"):
        gradio_app.evaluate_dataset_journey_gradio(
            validation_state,
            dataset_file,
            "Retrieval",
            progress=None,
        )

    evaluated_state = {"task": "pair", "dataset_root": str(tmp_path), "scored": []}
    monkeypatch.setattr(
        gradio_app,
        "evaluate_dataset_gradio_for_journey",
        lambda *args, **kwargs: (
            "summary",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "artifacts.zip",
            evaluated_state,
        ),
    )

    outputs = gradio_app.evaluate_dataset_journey_gradio(
        validation_state,
        dataset_file,
        "Pair Classification",
        progress=None,
    )

    assert all(update["visible"] is True for update in outputs[:5])
    assert outputs[6] == evaluated_state
    assert "Tune the pair threshold" in outputs[7]
    assert all(update["visible"] is True for update in outputs[8:])


def test_dataset_result_reset_hides_and_clears_all_dependent_stages():
    outputs = gradio_app.reset_dataset_results()

    assert outputs[0] is None
    assert all(update["visible"] is False for update in outputs[1:6])
    assert outputs[6] is None
    assert all(outputs[index]["visible"] is False for index in (7, 8, 9, 11, 12, 13))
    assert outputs[10] is None
    assert outputs[14] is None
    assert all(update["visible"] is False for update in outputs[16:])


def test_gradio_threshold_tuning_uses_scored_pair_state(tmp_path):
    _ = tmp_path
    state = {
        "task": "pair",
        "scored": [
            {"left_id": "a", "right_id": "b", "similarity_score": 0.9, "label": 1},
            {"left_id": "c", "right_id": "d", "similarity_score": 0.2, "label": 0},
        ],
    }

    summary_html, sweep_frame, report_html, artifacts_path = gradio_app.threshold_tuning_gradio(
        state,
        "f1",
        progress=None,
    )

    assert "Threshold Tuning" in summary_html
    assert not sweep_frame.empty
    assert "Threshold Tuning" in report_html
    with zipfile.ZipFile(artifacts_path) as archive:
        assert "threshold_tuning_threshold_sweep.csv" in archive.namelist()

    journey_outputs = gradio_app.threshold_tuning_journey_gradio(state, "f1", progress=None)
    assert all(update["visible"] is True for update in journey_outputs[:3])
    assert "Threshold tuning complete" in journey_outputs[4]


def test_gradio_scored_pair_explanation_uses_dataset_state(tmp_path):
    dataset_root = tmp_path / "pair_dataset"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "same\nleft", "suffix": ".py"},
                {"file_id": "b", "text": "same\nright", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
        metadata={"name": "scored_pairs"},
    )
    state = {
        "task": "pair",
        "dataset_root": str(dataset_root),
        "scored": [
            {"left_id": "a", "right_id": "b", "similarity_score": 0.9, "label": 1},
        ],
    }

    summary_html, matches_frame, report_html, artifacts_path = (
        gradio_app.explain_scored_pair_gradio(
            state,
            0,
            "line",
            0.85,
            0.6,
            0.3,
            5,
            progress=None,
        )
    )

    assert "Pair Explanation" in summary_html
    assert not matches_frame.empty
    assert "a vs b" in report_html
    with zipfile.ZipFile(artifacts_path) as archive:
        assert "scored_pair_explanation.html" in archive.namelist()

    journey_outputs = gradio_app.explain_scored_pair_journey_gradio(
        state,
        0,
        "line",
        0.85,
        0.6,
        0.3,
        5,
        progress=None,
    )
    assert all(update["visible"] is True for update in journey_outputs[:3])
    assert "Pair explanation complete" in journey_outputs[4]


def test_dataset_pair_resampling_reports_invalid_label_folds():
    scored = pd.DataFrame(
        {
            "left_id": ["a", "b"],
            "right_id": ["c", "d"],
            "label": [0, 1],
            "similarity_score": [0.1, 0.9],
        }
    )

    with pytest.raises(gradio_app.gr.Error, match="smallest label count"):
        gradio_app._resample_pair_scores(scored, folds=2, seed=7, threshold=0.5)


def test_dataset_retrieval_evaluation_exports_leaderboard_artifacts(tmp_path):
    dataset_root = tmp_path / "retrieval_dataset"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "q1_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "q2_file", "text": "print(2)", "suffix": ".py"},
                {"file_id": "d1_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d2_file", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame(
            [
                {"query_id": "q1", "file_id": "q1_file"},
                {"query_id": "q2", "file_id": "q2_file"},
            ]
        ),
        corpus=pd.DataFrame(
            [
                {"document_id": "d1", "file_id": "d1_file"},
                {"document_id": "d2", "file_id": "d2_file"},
            ]
        ),
        qrels=pd.DataFrame(
            [
                {"query_id": "q1", "document_id": "d1", "relevance": 1},
                {"query_id": "q2", "document_id": "d2", "relevance": 1},
            ]
        ),
        metadata={"name": "tiny_retrieval"},
    )
    archive_path = _zip_directory(dataset_root, tmp_path / "retrieval_dataset.zip")

    outputs = gradio_app.evaluate_dataset_gradio(
        archive_path,
        "Retrieval",
        "Lexical Only",
        gradio_app.DEFAULT_MODEL,
        "auto",
        "auto",
        "none",
        "python",
        "raw",
        0.5,
        1,
        2,
        7,
        progress=None,
    )

    (
        summary_html,
        metrics_frame,
        scored_frame,
        resample_metrics,
        resample_summary,
        artifacts_path,
    ) = outputs
    assert "tiny_retrieval" in summary_html
    assert "Mean Average Precision" in metrics_frame["Metric"].tolist()
    assert "Interpretation" in scored_frame.columns
    assert not resample_metrics.empty
    assert not resample_summary.empty

    with zipfile.ZipFile(artifacts_path) as archive:
        names = archive.namelist()
        assert "retrieval_scored_rows.csv" in names
        assert "retrieval_metrics.json" in names
        assert "retrieval_resampling_summary.csv" in names
        manifest = json.loads(archive.read("leaderboard_manifest.json").decode("utf-8"))
    assert manifest["dataset_kind"] == "retrieval"


def test_dataset_retrieval_resampling_reports_invalid_query_folds(tmp_path):
    dataset = write_retrieval_dataset(
        tmp_path / "retrieval_dataset",
        files=pd.DataFrame(
            [
                {"file_id": "q1_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d1_file", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "q1_file"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "d1_file"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
    )
    scored = pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["d1"],
            "similarity_score": [1.0],
            "relevance": [1.0],
        }
    )

    with pytest.raises(gradio_app.gr.Error, match="number of queries"):
        gradio_app._resample_retrieval_scores(scored, dataset, folds=2, seed=7, k=1)


def test_gradio_dataset_map_preserves_zero_seed_and_exports_artifacts(tmp_path):
    dataset_root = tmp_path / "map_pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py", "split": "train"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py", "split": "train"},
                {"file_id": "c", "text": "print(2)", "suffix": ".py", "split": "test"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
            ]
        ),
        metadata={"name": "tiny_map_pairs"},
    )
    archive_path = _zip_directory(dataset_root, tmp_path / "map_pairs.zip")

    summary_html, points_frame, map_html, artifacts_path = gradio_app.generate_dataset_map_gradio(
        archive_path,
        "Pair Classification",
        "pca",
        0,
        32,
        "split",
        progress=None,
    )

    assert "Dataset Map" in summary_html
    assert "tiny_map_pairs" in summary_html
    assert ">0</span>" in summary_html
    assert "document_id" in points_frame.columns
    assert "split" in points_frame.columns
    assert len(points_frame) == 3
    assert "train" in map_html
    assert "test" in map_html
    assert "<svg" in map_html

    with zipfile.ZipFile(artifacts_path) as archive:
        assert archive.namelist() == ["dataset_map.csv", "dataset_map.html", "dataset_map.json"]


def test_gradio_pair_explanation_exports_safe_html(tmp_path):
    _ = tmp_path
    summary_html, matches_frame, explanation_html, artifacts_path = (
        gradio_app.generate_pair_explanation_gradio(
            "print('<script>')\nprint(1)",
            "print('<script>')\nprint(2)",
            "line",
            0.85,
            0.6,
            0.3,
            5,
        )
    )

    assert "Pair Explanation" in summary_html
    assert not matches_frame.empty
    assert "<script>" not in explanation_html
    assert "&lt;script&gt;" in explanation_html

    with zipfile.ZipFile(artifacts_path) as archive:
        assert archive.namelist() == ["pair_explanation.html", "pair_explanation.json"]


def test_gradio_leaderboard_inspection_renders_uploaded_zip(tmp_path):
    payload = {
        "schema_version": 1,
        "metadata": {"name": "<tiny leaderboard>", "seed": 7},
        "manifest": {"name": "<tiny leaderboard>"},
        "cards": {
            "datasets": [{"name": "pairs", "card_type": "dataset", "task_family": "pair"}],
            "algorithms": [
                {"name": "<exact>", "card_type": "algorithm", "algorithm_kind": "builtin"}
            ],
        },
        "aggregate": [
            {
                "task_family": "pair",
                "algorithm_name": "<exact>",
                "metric": "f1",
                "mean_score": 1.0,
                "median_score": 1.0,
                "dataset_count": 1,
                "sample_count": 2,
                "rank": 1,
            }
        ],
        "per_dataset": [
            {
                "task_family": "pair",
                "dataset_name": "pairs",
                "algorithm_name": "<exact>",
                "metric": "f1",
                "score": 1.0,
                "sample_count": 2,
                "dataset_source": "local",
                "algorithm_kind": "builtin",
                "rank": 1,
            }
        ],
    }
    payload_path = tmp_path / "leaderboard.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    zip_path = tmp_path / "leaderboard_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(payload_path, arcname="nested/leaderboard.json")

    summary_html, aggregate, per_dataset, report_html, artifacts_path = (
        gradio_app.inspect_leaderboard_artifacts_gradio(zip_path)
    )

    assert "Leaderboard Inspection" in summary_html
    assert "nested" not in summary_html
    assert not aggregate.empty
    assert not per_dataset.empty
    assert "<tiny leaderboard>" not in report_html
    assert "&lt;tiny leaderboard&gt;" in report_html
    assert "<exact>" not in report_html
    assert "&lt;exact&gt;" in report_html

    with zipfile.ZipFile(artifacts_path) as archive:
        assert archive.namelist() == [
            "leaderboard_report.html",
            "leaderboard_report.json",
            "leaderboard_report_aggregate.csv",
            "leaderboard_report_details.html",
            "leaderboard_report_per_dataset.csv",
            "leaderboard_report_reproducibility.json",
        ]


def test_ready_leaderboard_registered_datasets_frame_lists_all_presets():
    frame = gradio_app.ready_leaderboard_registered_datasets_frame()

    assert set(frame["Preset"]) == set(available_dataset_presets())
    assert "Evaluation Metrics" in frame.columns
    assert "Sampling Default" in frame.columns
    assert set(frame["Category"]) == {"Plagiarism"}


def test_leaderboard_filters_keep_plagiarism_and_clone_categories_separate():
    rows = [
        {"task_type": category, "task_family": "pair", "dataset_name": category,
         "algorithm_name": "same method", "metric": "f1", "score": score,
         "sample_count": 10, "dataset_source": "test", "rank": 1}
        for category, score in (("plagiarism", 0.2), ("code_clone", 0.9))
    ]
    report = gradio_app._leaderboard_report_from_payload({
        "metadata": {}, "per_dataset": rows, "aggregate": [],
    })
    aggregate, per_dataset = gradio_app._filter_ready_made_leaderboard_report(
        report, "pair", "f1", gradio_app.READY_LEADERBOARD_ALL_DATASETS,
        gradio_app.READY_LEADERBOARD_ALL_ALGORITHMS, gradio_app.READY_LEADERBOARD_ALL_SOURCES,
    )

    assert len(aggregate) == 2
    assert set(aggregate["mean_score"]) == {0.2, 0.9}
    assert set(aggregate["rank"]) == {1}
    assert len(per_dataset) == 2
    filtered, rows = gradio_app._filter_ready_made_leaderboard_report(
        report, "pair", "f1", gradio_app.READY_LEADERBOARD_ALL_DATASETS,
        gradio_app.READY_LEADERBOARD_ALL_ALGORITHMS, gradio_app.READY_LEADERBOARD_ALL_SOURCES,
        dataset_category="plagiarism",
    )
    assert filtered.iloc[0]["mean_score"] == 0.2
    assert set(rows["task_type"]) == {"plagiarism"}


def test_legacy_demo_leaderboard_is_labeled_plagiarism_and_demo():
    report = gradio_app._leaderboard_report_from_payload(
        gradio_app.load_ready_made_leaderboard_payload()
    )
    assert set(report["per_dataset"]["task_type"]) == {"plagiarism"}
    assert set(gradio_app.ready_made_leaderboard_coverage_frame(report)["Category"]) == {"Plagiarism"}
    assert "Demo" in gradio_app.ready_made_leaderboard_summary_html(report)
    assert "Dataset Category" in {
        component.get("props", {}).get("label")
        for component in gradio_app.demo.get_config_file()["components"]
    }


def test_leaderboard_filter_preserves_benchmark_protocol_identity():
    rows = [
        {"task_type": "plagiarism", "domain": "plagiarism", "task_family": "pair",
         "benchmark_id": "Matheel-Plagiarism-v1", "protocol_id": protocol,
         "dataset_name": "dataset", "algorithm_name": "method", "metric": "f1",
         "score": score, "sample_count": 10, "dataset_source": "test", "rank": 1}
        for protocol, score in (("protocol-a", 0.2), ("protocol-b", 0.9))
    ]
    report = gradio_app._leaderboard_report_from_payload({
        "metadata": {}, "per_dataset": rows, "aggregate": [],
        "benchmark": {"benchmark_id": "Matheel-Plagiarism-v1"},
    })
    aggregate, _ = gradio_app._filter_ready_made_leaderboard_report(
        report, "pair", "f1", gradio_app.READY_LEADERBOARD_ALL_DATASETS,
        gradio_app.READY_LEADERBOARD_ALL_ALGORITHMS, gradio_app.READY_LEADERBOARD_ALL_SOURCES,
    )

    assert report["benchmark"]["benchmark_id"] == "Matheel-Plagiarism-v1"
    assert set(aggregate["protocol_id"]) == {"protocol-a", "protocol-b"}
    assert set(aggregate["mean_score"]) == {0.2, 0.9}
    assert set(aggregate["domain"]) == {"plagiarism"}


def test_official_leaderboard_filter_preserves_coverage_and_unranked_methods():
    rows = [
        {"task_type": "plagiarism", "task_family": "pair", "dataset_name": dataset,
         "dataset_source": "test", "algorithm_name": method, "algorithm_id": identity,
         "metric": "f1", "score": score, "sample_count": 10, "rank": None}
        for method, identity, scores in (
            ("Partial", "method-a", (1.0, None)),
            ("Complete", "method-b", (0.8, 0.8)),
            ("Failed", "method-c", (None, None)),
        )
        for dataset, score in zip(("dataset-a", "dataset-b"), scores)
    ]
    aggregates = [
        {"task_type": "plagiarism", "task_family": "pair", "algorithm_name": method,
         "algorithm_id": identity, "metric": "f1", "mean_score": score,
         "median_score": score, "rank": rank, "eligible": eligible, "coverage": coverage,
         "dataset_count": count, "expected_dataset_count": 2, "sample_count": 20}
        for method, identity, score, rank, eligible, coverage, count in (
            ("Partial", "method-a", None, None, False, 0.5, 1),
            ("Complete", "method-b", 0.8, 1, True, 1.0, 2),
            ("Failed", "method-c", None, None, False, 0.0, 0),
        )
    ]
    report = gradio_app._leaderboard_report_from_payload({
        "metadata": {"strict_coverage": True}, "per_dataset": rows, "aggregate": aggregates,
    })
    aggregate, details = gradio_app._filter_ready_made_leaderboard_report(
        report, "pair", "f1", "dataset-a",
        gradio_app.READY_LEADERBOARD_ALL_ALGORITHMS, gradio_app.READY_LEADERBOARD_ALL_SOURCES,
    )
    ranked = aggregate.set_index("algorithm_id")
    assert pd.isna(ranked.loc["method-a", "mean_score"])
    assert pd.isna(ranked.loc["method-a", "rank"])
    assert pd.isna(ranked.loc["method-c", "rank"])
    assert ranked.loc["method-b", "rank"] == 1
    assert ranked.loc["method-a", "coverage"] == 0.5
    assert ranked.loc["method-a", "expected_dataset_count"] == 2
    assert ranked.loc["method-a", "eligible"] == False  # noqa: E712
    assert set(details["dataset_name"]) == {"dataset-a"}
    assert aggregate["ranking_scope"].str.startswith("Full benchmark").all()


def test_ready_made_leaderboard_records_historical_preset_coverage():
    payload = gradio_app.load_ready_made_leaderboard_payload()
    report = gradio_app._leaderboard_report_from_payload(payload)
    profile = report["metadata"]["benchmark_profile"]
    coverage = gradio_app.ready_made_leaderboard_coverage_frame(report)

    assert set(profile["dataset_presets"]) == {
        "conplag",
        "criminal_minds",
        "ipca",
        "irplag",
        "soco14",
        "student_code_similarity",
    }
    assert set(profile["dataset_presets"]).issubset(set(available_dataset_presets()))
    assert set(profile["algorithm_presets"]).issubset(gradio_app.READY_LEADERBOARD_ALGORITHM_CHOICES)
    assert profile["dataset_task_count"] == 7
    assert len(coverage) == 7
    assert set(report["per_dataset"]["algorithm_name"]) == set(
        profile["algorithm_presets"]
    )
    assert set(report["aggregate"]["task_family"]) == {"pair", "retrieval"}
    assert tuple(report["metadata"]["pair_metrics"]) == (
        "f1",
        "accuracy",
        "precision",
        "recall",
        "auroc",
        "average_precision",
    )
    assert tuple(report["metadata"]["retrieval_metrics"]) == (
        "mean_average_precision",
        "mean_reciprocal_rank",
        "ndcg_at_k",
        "precision_at_k",
        "recall_at_k",
    )
    assert len(report["aggregate"]) == 88
    assert len(report["per_dataset"]) == 320
    assert "static_hash" in gradio_app.ready_made_leaderboard_summary_html(report)


def test_ready_made_leaderboard_uses_task_metrics_and_descending_defaults():
    pair_options = gradio_app.ready_made_leaderboard_filter_options("pair")
    retrieval_options = gradio_app.ready_made_leaderboard_filter_options("retrieval")

    assert pair_options["metric"] == "f1"
    assert pair_options["metrics"] == list(gradio_app.READY_LEADERBOARD_PAIR_METRICS)
    assert retrieval_options["metric"] == "mean_average_precision"
    assert retrieval_options["metrics"] == list(gradio_app.READY_LEADERBOARD_RETRIEVAL_METRICS)

    aggregate, per_dataset = gradio_app.filter_ready_made_leaderboard(
        "pair",
        "f1",
    )

    assert set(aggregate["task_family"]) == {"pair"}
    assert set(aggregate["metric"]) == {"f1"}
    assert aggregate["mean_score"].is_monotonic_decreasing
    assert per_dataset["score"].is_monotonic_decreasing


def test_ready_made_leaderboard_supports_explicit_sort_controls():
    aggregate, per_dataset = gradio_app.filter_ready_made_leaderboard(
        "pair",
        "f1",
        aggregate_sort="algorithm_name",
        aggregate_direction="Ascending",
        per_dataset_sort="dataset_name",
        per_dataset_direction="Descending",
    )

    assert aggregate["algorithm_name"].is_monotonic_increasing
    assert per_dataset["dataset_name"].is_monotonic_decreasing

    config = gradio_app.demo.get_config_file()
    labels = {
        component["props"].get("label")
        for component in config["components"]
        if component["type"] == "dropdown"
    }
    assert {
        "Sort Aggregate By",
        "Aggregate Direction",
        "Sort Per-Dataset By",
        "Per-Dataset Direction",
    }.issubset(labels)


def test_ready_made_leaderboard_filters_dataset_algorithm_and_source():
    payload = gradio_app.load_ready_made_leaderboard_payload()
    report = gradio_app._leaderboard_report_from_payload(payload)
    row = report["per_dataset"].query("task_family == 'retrieval'").iloc[0]

    aggregate, per_dataset = gradio_app.filter_ready_made_leaderboard(
        "retrieval",
        "mean_average_precision",
        row["dataset_name"],
        row["algorithm_name"],
        row["dataset_source"],
    )

    assert set(per_dataset["dataset_name"]) == {row["dataset_name"]}
    assert set(per_dataset["algorithm_name"]) == {row["algorithm_name"]}
    assert set(per_dataset["dataset_source"]) == {row["dataset_source"]}
    assert set(per_dataset["metric"]) == {"mean_average_precision"}
    assert aggregate.iloc[0]["dataset_count"] == 1
    assert aggregate.iloc[0]["algorithm_name"] == row["algorithm_name"]


def test_ready_leaderboard_rejects_empty_algorithm_selection():
    with pytest.raises(gradio_app.gr.Error, match="Select at least one algorithm"):
        gradio_app.ready_leaderboard_algorithm_configs(
            [],
            gradio_app.DEFAULT_MODEL,
            "auto",
            "auto",
            "none",
            "python",
            "raw",
        )


def test_gradio_ready_leaderboard_runs_pair_and_retrieval_uploads(tmp_path):
    pair_root = tmp_path / "ready_pairs"
    write_pair_dataset(
        pair_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
                {"file_id": "c", "text": "print(2)", "suffix": ".py"},
                {"file_id": "d", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "c", "right_id": "d", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
                {"left_id": "b", "right_id": "d", "label": 0},
            ]
        ),
        metadata={"name": "ready_pairs"},
    )
    pair_zip = _zip_directory(pair_root, tmp_path / "ready_pairs.zip")

    retrieval_root = tmp_path / "ready_retrieval"
    write_retrieval_dataset(
        retrieval_root,
        files=pd.DataFrame(
            [
                {"file_id": "q1_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "q2_file", "text": "print(2)", "suffix": ".py"},
                {"file_id": "d1_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d2_file", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame(
            [
                {"query_id": "q1", "file_id": "q1_file"},
                {"query_id": "q2", "file_id": "q2_file"},
            ]
        ),
        corpus=pd.DataFrame(
            [
                {"document_id": "d1", "file_id": "d1_file"},
                {"document_id": "d2", "file_id": "d2_file"},
            ]
        ),
        qrels=pd.DataFrame(
            [
                {"query_id": "q1", "document_id": "d1", "relevance": 1},
                {"query_id": "q2", "document_id": "d2", "relevance": 1},
            ]
        ),
        metadata={"name": "ready_retrieval"},
    )
    retrieval_zip = _zip_directory(retrieval_root, tmp_path / "ready_retrieval.zip")

    summary_html, aggregate, per_dataset, report_html, artifacts_path = (
        gradio_app.run_ready_leaderboard_gradio(
            [pair_zip, retrieval_zip],
            ["Lexical Only"],
            gradio_app.DEFAULT_MODEL,
            "auto",
            "auto",
            "none",
            "python",
            "raw",
            0.5,
            2,
            7,
            progress=None,
        )
    )

    assert "Ready Leaderboard" in summary_html
    assert "ready_pairs" in per_dataset["dataset_name"].tolist()
    assert "ready_retrieval" in per_dataset["dataset_name"].tolist()
    assert set(aggregate["task_family"]) == {"pair", "retrieval"}
    assert "f1" in per_dataset["metric"].tolist()
    assert "mean_average_precision" in per_dataset["metric"].tolist()
    assert "Ranked Algorithms" not in report_html
    assert "Aggregate Ranking" in report_html

    with zipfile.ZipFile(artifacts_path) as archive:
        names = archive.namelist()
        assert "ready_leaderboard.json" in names
        assert "ready_leaderboard_aggregate.csv" in names
        assert "ready_leaderboard_details.html" in names
        assert "ready_leaderboard_per_dataset.csv" in names
        assert "ready_leaderboard_reproducibility.json" in names


@pytest.mark.parametrize("stage,reset_name", [
    ("evaluation", "reset_dataset_upload_journey"),
    ("evaluation", "reset_dataset_results"),
    ("threshold", "reset_dataset_threshold_stage"),
    ("explanation", "reset_dataset_explanation_stage"),
])
def test_held_dataset_work_cannot_commit_after_its_stage_resets(stage, reset_name):
    import threading
    state = {}
    started, release = threading.Event(), threading.Event()
    prepared = []
    def callback(progress=None):
        started.set()
        assert release.wait(5)
        return ("obsolete summary", {"scored": ["obsolete row"]})
    thread = threading.Thread(target=lambda: prepared.append(
        gradio_app.run_current_dataset_journey(state, stage, callback)
    ))
    thread.start()
    try:
        assert started.wait(5)
        getattr(gradio_app, reset_name)(state)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert prepared[0][:-1] == (gradio_app.gr.skip(), gradio_app.gr.skip())
    assert gradio_app.commit_current_dataset_journey(state, prepared[0][-1]) == (
        gradio_app.gr.skip(), gradio_app.gr.skip(),
    )


def test_dataset_result_commit_and_all_resets_share_a_serial_concurrency_group():
    config = gradio_app.demo.get_config_file()
    guarded = [dependency for dependency in config["dependencies"] if (
        dependency["inputs"] and dependency["inputs"][0] == gradio_app.dataset_journey_generation._id
        and (dependency.get("trigger_after") is not None or "reset_dataset" in str(dependency.get("api_name")))
    )]
    assert len(guarded) >= 3 + 11 + 1 + 6
    for dependency in guarded:
        function = gradio_app.demo.fns[dependency["id"]]
        assert function.concurrency_id == "dataset-journey-commit"
        assert function.concurrency_limit == 1
        if dependency.get("trigger_after") is not None:
            assert dependency["trigger_only_on_success"] is True


def test_generation_state_is_mutable_by_reference_and_isolated_per_browser_session():
    from gradio.state_holder import SessionState
    first, second = SessionState(gradio_app.demo), SessionState(gradio_app.demo)
    identifier = gradio_app.dataset_journey_generation._id
    running_reference = first[identifier]
    assert running_reference is first[identifier]
    assert running_reference is not second[identifier]
    gradio_app.reset_dataset_results(first[identifier])
    assert running_reference["evaluation"] == 1
    assert second[identifier] == {}


def test_stage_specific_reset_does_not_invalidate_unrelated_work():
    state = {}
    prepared = gradio_app.run_current_dataset_journey(state, "evaluation", lambda progress=None: ("current",))
    gradio_app.reset_dataset_threshold_stage(state)
    gradio_app.reset_dataset_explanation_stage(state)
    assert gradio_app.commit_current_dataset_journey(state, prepared[-1]) == ("current",)


@pytest.mark.parametrize("stage", ["evaluation", "threshold", "explanation"])
def test_old_request_admission_cannot_start_after_its_queued_stage_resets(stage):
    state = {}
    calls = []
    gradio_app.invalidate_dataset_journey(state, stage, event_revision=7)
    queued = gradio_app.run_current_dataset_journey(
        state, stage, lambda progress=None: calls.append("obsolete") or ("obsolete",),
        admission_revision=6, output_count=1,
    )
    assert calls == []
    assert queued == (gradio_app.gr.skip(), None)
    assert gradio_app.commit_current_dataset_journey(state, queued[-1]) == gradio_app.gr.skip()
    current = gradio_app.run_current_dataset_journey(
        state, stage, lambda progress=None: ("current",), admission_revision=7, output_count=1,
    )
    assert gradio_app.commit_current_dataset_journey(state, current[-1]) == ("current",)


def test_queued_request_admission_remains_scoped_to_its_stage():
    state = {}
    gradio_app.reset_dataset_threshold_stage(state, event_revision=7)
    prepared = gradio_app.run_current_dataset_journey(
        state, "evaluation", lambda progress=None: ("current",), admission_revision=6, output_count=1,
    )
    assert gradio_app.commit_current_dataset_journey(state, prepared[-1]) == ("current",)


def test_rejected_queued_evaluation_keeps_current_dependent_work_valid():
    state = {}
    gradio_app.reset_dataset_results(state, event_revision=7)
    threshold = gradio_app.run_current_dataset_journey(
        state, "threshold", lambda progress=None: ("current threshold",),
        admission_revision=8, output_count=1,
    )
    gradio_app.run_current_dataset_journey(
        state, "evaluation", lambda progress=None: pytest.fail("obsolete evaluation ran"),
        admission_revision=6, output_count=12, invalidate_stages=("threshold", "explanation"),
    )
    assert gradio_app.commit_current_dataset_journey(state, threshold[-1]) == ("current threshold",)


@pytest.mark.parametrize("stage,reset_name,count", [
    ("evaluation", "reset_dataset_results", 20),
    ("evaluation", "reset_dataset_upload_journey", 26),
    ("threshold", "reset_dataset_threshold_stage", 5),
    ("explanation", "reset_dataset_explanation_stage", 5),
])
def test_delayed_older_reset_preserves_newer_work_and_visible_results(stage, reset_name, count):
    state = {}
    reset_outputs = []
    def callback(progress=None):
        reset_outputs.append(getattr(gradio_app, reset_name)(state, event_revision=9))
        return ("newer current result",)
    pending = gradio_app.run_current_dataset_journey(
        state, stage, callback, admission_revision=10, output_count=1,
    )
    assert reset_outputs == [tuple(gradio_app.gr.skip() for _ in range(count))]
    assert state.get(stage, 0) == 0
    assert gradio_app.commit_current_dataset_journey(state, pending[-1]) == ("newer current result",)


def test_new_validation_identity_invalidates_work_even_before_file_change_arrives(monkeypatch):
    generation = {}
    monkeypatch.setattr(gradio_app, "validate_dataset_journey_gradio", lambda *args, **kwargs: (
        "validated", "issues", "report", "artifact", "new identity", "enabled", "status",
    ))
    outputs = gradio_app.update_dataset_upload_journey(
        "new.zip", "Pair Classification", gradio_app.gr.EventData(gradio_app.dataset_validate, None),
        validation_state="old identity", generation_state=generation, progress=None,
    )
    assert generation == {"evaluation": 1, "threshold": 1, "explanation": 1}
    assert outputs[gradio_app.dataset_scored_state] is None
    assert outputs[gradio_app.dataset_summary]["visible"] is False


def test_later_scoring_option_reset_does_not_discard_upload_validation(monkeypatch):
    generation = {}
    gradio_app.reset_dataset_results(generation, event_revision=16)
    monkeypatch.setattr(gradio_app, "validate_dataset_journey_gradio", lambda *args, **kwargs: (
        "validated", "issues", "report", "artifact", "new identity", "enabled", "status",
    ))
    outputs = gradio_app.update_dataset_upload_journey(
        "new.zip", "Pair Classification", gradio_app.gr.EventData(gradio_app.dataset_validate, None),
        validation_state=None, event_revision=15, latest_revision=13,
        generation_state=generation, progress=None,
    )
    assert outputs[gradio_app.dataset_validation_state] == "new identity"
    assert outputs[gradio_app.dataset_run] == "enabled"
    assert outputs[gradio_app.dataset_summary] == gradio_app.gr.skip()
    assert generation["evaluation"] == 1


@pytest.mark.parametrize("validation_state", [None, "validated upload/task"])
def test_scoring_reset_preserves_validation_guidance_until_validation_passes(validation_state):
    generation = {}
    outputs = gradio_app.reset_dataset_results(generation, event_revision=10, validation_state=validation_state)
    assert outputs[0] is None
    assert outputs[1]["visible"] is False
    assert outputs[6] is None
    assert generation["evaluation"] == 1
    if validation_state is None:
        assert outputs[15] == gradio_app.gr.skip()
    else:
        assert "Settings changed" in outputs[15]


def test_late_same_upload_notification_keeps_active_work_generation(monkeypatch):
    generation = {"evaluation": 3, "threshold": 2, "explanation": 1}
    monkeypatch.setattr(gradio_app, "dataset_upload_identity", lambda *args: "current identity")
    outputs = gradio_app.update_dataset_upload_journey(
        "dataset.zip", "Pair Classification", gradio_app.gr.EventData(gradio_app.dataset_file, None),
        validation_state="current identity", generation_state=generation, progress=None,
    )
    assert generation == {"evaluation": 3, "threshold": 2, "explanation": 1}
    assert outputs == {gradio_app.dataset_latest_revision: 0}
