import importlib.util
import zipfile
from pathlib import Path

import pandas as pd
import pytest


pytest.importorskip("gradio")
pytest.importorskip("gradio_huggingfacehub_search")


APP_PATH = Path(__file__).resolve().parents[1] / "gradio_app" / "app.py"
SPEC = importlib.util.spec_from_file_location("matheel_gradio_app", APP_PATH)
gradio_app = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gradio_app)


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
            }
        ]
    )

    summary_html = gradio_app.suite_summary_html(summary)

    assert "<b>best</b>" not in summary_html
    assert "&lt;b&gt;best&lt;/b&gt;" in summary_html


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

    summary_html = gradio_app.results_summary_html(results, "auto", "none", "none", "cpu")

    assert "<b>a.py</b>" not in summary_html
    assert "<script>" not in summary_html
    assert "&lt;b&gt;a.py&lt;/b&gt;" in summary_html
    assert "&lt;script&gt;x&lt;/script&gt;" in summary_html


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
    ):
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
