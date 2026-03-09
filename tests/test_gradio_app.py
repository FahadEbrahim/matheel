import importlib.util
from pathlib import Path

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
