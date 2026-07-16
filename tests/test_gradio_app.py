import importlib.util
import json
import os
import socket
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pytest

from matheel.datasets import available_dataset_presets, write_pair_dataset, write_retrieval_dataset


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
    "Ready-made Leaderboard",
    "Build Leaderboard",
    "Inspect Artifacts",
)
EXPECTED_PRIMARY_ACTIONS = {
    "Run Pair": "calculate_similarity_gradio",
    "Run Collection": "get_sim_list_gradio",
    "Run Suite": "run_suite_gradio",
    "Validate Dataset": "validate_dataset_gradio",
    "Run Dataset Evaluation": "evaluate_dataset_gradio_with_state",
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
        _ = zipped_file, summary_out, details_dir, output_format, progress_callback
        assert run_configs[0]["run_name"] == "../baseline/strong"
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

    outputs = gradio_app.evaluate_dataset_gradio(
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

    summary_html, metrics_frame, scored_frame, resample_metrics, resample_summary, artifacts_path = outputs
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
        assert "pair_scored_rows.csv" in names
        assert "pair_metrics.json" in names
        assert "pair_resampling_summary.csv" in names
        assert "pair_threshold_tuning_threshold_sweep.csv" in names
        manifest = json.loads(archive.read("leaderboard_manifest.json").decode("utf-8"))
    assert manifest["workflow"] == "gradio_dataset_evaluation"
    assert manifest["dataset_kind"] == "pair_classification"


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

    summary_html, matches_frame, report_html, artifacts_path = gradio_app.explain_scored_pair_gradio(
        state,
        0,
        "line",
        0.85,
        0.6,
        0.3,
        5,
        progress=None,
    )

    assert "Pair Explanation" in summary_html
    assert not matches_frame.empty
    assert "a vs b" in report_html
    with zipfile.ZipFile(artifacts_path) as archive:
        assert "scored_pair_explanation.html" in archive.namelist()


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

    summary_html, metrics_frame, scored_frame, resample_metrics, resample_summary, artifacts_path = outputs
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


def test_gradio_dataset_map_exports_visualization_artifacts(tmp_path):
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
        7,
        32,
        "split",
        progress=None,
    )

    assert "Dataset Map" in summary_html
    assert "tiny_map_pairs" in summary_html
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
            "algorithms": [{"name": "<exact>", "card_type": "algorithm", "algorithm_kind": "builtin"}],
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


def test_ready_made_leaderboard_covers_all_presets_and_algorithms():
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
    assert set(profile["algorithm_presets"]) == set(
        gradio_app.READY_LEADERBOARD_ALGORITHM_CHOICES
    )
    assert profile["dataset_task_count"] == 7
    assert len(coverage) == 7
    assert set(report["per_dataset"]["algorithm_name"]) == set(
        gradio_app.READY_LEADERBOARD_ALGORITHM_CHOICES
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
    assert retrieval_options["metrics"] == list(
        gradio_app.READY_LEADERBOARD_RETRIEVAL_METRICS
    )

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

    summary_html, aggregate, per_dataset, report_html, artifacts_path = gradio_app.run_ready_leaderboard_gradio(
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
