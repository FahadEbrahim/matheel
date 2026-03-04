import json
import math
import os
import tempfile
import zipfile

mpl_config_dir = os.path.join(tempfile.gettempdir(), "matheel-mpl")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

import gradio as gr
import pandas as pd
from gradio_huggingfacehub_search import HuggingfaceHubSearch

from matheel.chunking import available_chunk_aggregations, available_chunking_methods
from matheel.comparison_suite import run_comparison_suite
from matheel.code_metrics import available_code_metric_languages, available_code_metrics
from matheel.model_routing import available_vector_backends
from matheel.preprocessing import available_preprocess_modes
from matheel.similarity import (
    DEFAULT_MODEL_NAME,
    available_runtime_devices,
    calculate_similarity,
    get_sim_list,
    inspect_model_settings,
)
from matheel.vectors import available_pooling_methods, available_similarity_functions


DEVICE_CHOICES = ("auto",) + available_runtime_devices()
DEFAULT_MODEL = DEFAULT_MODEL_NAME
CHUNK_LANGUAGE_CHOICES = [
    "text",
    "python",
    "java",
    "c",
    "cpp",
    "javascript",
    "typescript",
    "go",
    "rust",
]
PREPROCESSING_UI_CHOICES = [mode for mode in available_preprocess_modes() if mode != "none"]
CHONKIE_UI_METHODS = [method for method in available_chunking_methods() if method != "none"]


def build_feature_weights(
    use_semantic,
    ws,
    use_levenshtein,
    wl,
    use_jaro_winkler,
    wj,
    code_metric,
    code_metric_weight,
):
    weights = {}
    fallback_names = []

    if use_semantic:
        weights["semantic"] = max(0.0, float(ws))
        fallback_names.append("semantic")
    if use_levenshtein:
        weights["levenshtein"] = max(0.0, float(wl))
        fallback_names.append("levenshtein")
    if use_jaro_winkler:
        weights["jaro_winkler"] = max(0.0, float(wj))
        fallback_names.append("jaro_winkler")

    numeric_code_metric = max(0.0, float(code_metric_weight))
    if (code_metric or "none") != "none":
        weights["code_metric"] = numeric_code_metric
        fallback_names.append("code_metric")

    return normalize_feature_weights_map(weights, fallback_names=fallback_names)


def normalize_feature_weights_map(weights, fallback_names=None):
    filtered = {name: float(value) for name, value in (weights or {}).items() if float(value) > 0}
    if not filtered:
        fallback = [name for name in (fallback_names or []) if name]
        if fallback:
            raise gr.Error("At least one selected metric must have a positive weight.")
        return {"semantic": 1.0}

    total = sum(filtered.values())
    if abs(total - 1.0) > 0.02:
        raise gr.Error("Active metric weights must sum to 1.")
    return {name: value / total for name, value in filtered.items()}


def parse_weight_values_text(raw_text, expected_count, label):
    text = str(raw_text or "").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise gr.Error(f"{label} must contain exactly {expected_count} comma-separated floats.")

    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise gr.Error(f"{label} must contain only numeric values.") from exc
        if not math.isfinite(value) or value <= 0:
            raise gr.Error(f"{label} must contain positive floats only.")
        values.append(value)

    total = sum(values)
    if total <= 0 or abs(total - 1.0) > 0.02:
        raise gr.Error(f"{label} must sum to 1.")
    return [value / total for value in values]


def parse_positive_values_text(raw_text, expected_count, label):
    text = str(raw_text or "").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise gr.Error(f"{label} must contain exactly {expected_count} comma-separated values.")

    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise gr.Error(f"{label} must contain only numeric values.") from exc
        if not math.isfinite(value) or value <= 0:
            raise gr.Error(f"{label} must contain positive values only.")
        values.append(value)
    return values


def format_weight_values(values):
    formatted = []
    for value in values:
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        formatted.append(text or "0")
    return ",".join(formatted)


def validate_levenshtein_weights_text(raw_text):
    values = parse_positive_values_text(raw_text or "1,1,1", 3, "Levenshtein weights")
    return tuple(max(1, int(round(value))) for value in values), format_weight_values(values)


def validate_codebleu_component_weights_text(raw_text):
    normalized = parse_weight_values_text(
        raw_text or "0.25,0.25,0.25,0.25",
        4,
        "CodeBLEU weights",
    )
    return format_weight_values(normalized)


def profile_status_html(message):
    return f"<p><strong>Status:</strong> {message}</p>"


def model_status_html(settings=None, error_message=None):
    if error_message:
        return f"<p><strong>Model:</strong> {error_message}</p>"

    if not settings:
        return ""

    backend = settings.get("resolved_vector_backend", "auto")
    detected = int(settings.get("detected_max_token_length") or 0)
    configured = settings.get("configured_max_token_length")
    active = int(configured or detected or 0)
    device = settings.get("runtime_device", "auto")
    return (
        f"<p><strong>Resolved:</strong> {backend} | "
        f"<strong>Detected:</strong> {detected} | "
        f"<strong>Active:</strong> {active} | "
        f"<strong>Device:</strong> {device}</p>"
    )


def update_feature_sections(selected_features):
    selected = set(selected_features or [])
    return (
        gr.update(visible="Embedding" in selected),
        gr.update(visible="Levenshtein" in selected),
        gr.update(visible="Jaro-Winkler" in selected),
        gr.update(visible="Code Metric" in selected),
    )


def update_code_preparation_sections(selected_steps):
    selected = set(selected_steps or [])
    return (
        gr.update(visible="Preprocessing" in selected),
        gr.update(visible="Chunking" in selected),
    )


def update_code_metric_sections(code_metric):
    normalized = (code_metric or "codebleu").strip().lower()
    return (
        gr.update(visible=normalized.startswith("codebleu")),
        gr.update(visible=normalized == "crystalbleu"),
    )


def embedding_parameter_updates(vector_backend):
    normalized = (vector_backend or "sentence_transformers").strip().lower()
    if normalized in ("", "auto"):
        normalized = "sentence_transformers"
    return (
        gr.update(visible=normalized != "pylate"),
        gr.update(visible=normalized == "sentence_transformers"),
    )


def sync_model_settings_gradio(
    model_name,
    vector_backend,
    runtime_device,
    similarity_function,
    pooling_method,
    max_token_length,
):
    current_limit = int(max(8, float(max_token_length or 256)))
    current_backend = (vector_backend or "auto").strip() or "auto"
    try:
        settings = inspect_model_settings(
            model_name or "",
            vector_backend=current_backend,
            device=runtime_device,
            similarity_function=similarity_function,
            pooling_method=pooling_method,
            max_token_length=max_token_length,
        )
    except Exception as exc:
        slider_max = max(current_limit, 256)
        return (
            gr.update(value=current_backend),
            gr.update(minimum=8, maximum=slider_max, value=current_limit),
            model_status_html(error_message=str(exc)),
            *embedding_parameter_updates(current_backend),
        )

    detected = max(8, int(settings["detected_max_token_length"]))
    configured = int(settings["configured_max_token_length"] or detected)
    resolved_backend = settings["resolved_vector_backend"]
    return (
        gr.update(value=current_backend),
        gr.update(minimum=8, maximum=detected, value=max(8, min(configured, detected))),
        model_status_html(settings=settings),
        *embedding_parameter_updates(resolved_backend),
    )


SUITE_COLUMNS = [
    "run_name",
    "model_name",
    "vector_backend",
    "similarity_function",
    "pooling_method",
    "max_token_length",
    "preprocess_mode",
    "chunking_method",
    "chunk_size",
    "chunk_overlap",
    "max_chunks",
    "chunk_aggregation",
    "chunk_language",
    "chunker_options",
    "code_metric",
    "code_metric_weight",
    "code_language",
    "codebleu_component_weights",
    "crystalbleu_max_order",
    "crystalbleu_trivial_ngram_count",
    "semantic_weight",
    "levenshtein_weight",
    "levenshtein_weights",
    "jaro_winkler_weight",
    "jaro_winkler_prefix_weight",
    "threshold",
    "number_results",
]

def empty_suite_rows():
    return pd.DataFrame(columns=SUITE_COLUMNS)


def normalize_suite_rows_frame(rows):
    if rows is None:
        return pd.DataFrame(columns=SUITE_COLUMNS)
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        frame = pd.DataFrame(rows, columns=SUITE_COLUMNS)
    for column in SUITE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[SUITE_COLUMNS]


def suite_rows_to_configs(rows):
    frame = normalize_suite_rows_frame(rows)
    if frame.empty:
        return []

    frame = frame.fillna("")
    configs = []
    for index, row in frame.iterrows():
        run_name = str(row.get("run_name", "")).strip() or f"run_{index + 1}"
        code_metric = str(row.get("code_metric", "none") or "none").strip() or "none"
        raw_feature_weights = {
            "semantic": max(0.0, float(row.get("semantic_weight") or 0)),
            "levenshtein": max(0.0, float(row.get("levenshtein_weight") or 0)),
            "jaro_winkler": max(0.0, float(row.get("jaro_winkler_weight") or 0)),
        }
        code_metric_weight = max(0.0, float(row.get("code_metric_weight") or 0))
        if code_metric != "none" and code_metric_weight > 0:
            raw_feature_weights["code_metric"] = code_metric_weight
        feature_weights = normalize_feature_weights_map(raw_feature_weights)

        effective_levenshtein_weights = "1,1,1"
        if feature_weights.get("levenshtein", 0.0) > 0:
            effective_levenshtein_weights, _ = validate_levenshtein_weights_text(
                row.get("levenshtein_weights") or "1,1,1"
            )

        effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
        if code_metric.startswith("codebleu") and feature_weights.get("code_metric", 0.0) > 0:
            effective_codebleu_component_weights = validate_codebleu_component_weights_text(
                row.get("codebleu_component_weights") or "0.25,0.25,0.25,0.25"
            )

        options = {
            "model_name": str(row.get("model_name") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            "vector_backend": str(row.get("vector_backend") or "auto").strip() or "auto",
            "similarity_function": str(row.get("similarity_function") or "cosine").strip() or "cosine",
            "pooling_method": str(row.get("pooling_method") or "mean").strip() or "mean",
            "max_token_length": max(8, int(float(row.get("max_token_length") or 256))),
            "preprocess_mode": str(row.get("preprocess_mode") or "none").strip() or "none",
            "chunking_method": str(row.get("chunking_method") or "none").strip() or "none",
            "chunk_size": max(10, int(float(row.get("chunk_size") or 120))),
            "chunk_overlap": max(0, int(float(row.get("chunk_overlap") or 0))),
            "max_chunks": max(0, int(float(row.get("max_chunks") or 0))),
            "chunk_aggregation": str(row.get("chunk_aggregation") or "mean").strip() or "mean",
            "chunk_language": str(row.get("chunk_language") or "text").strip() or "text",
            "chunker_options": str(row.get("chunker_options") or "").strip(),
            "code_metric": code_metric,
            "code_metric_weight": feature_weights.get("code_metric", 0.0),
            "code_language": str(row.get("code_language") or "python").strip() or "python",
            "codebleu_component_weights": effective_codebleu_component_weights,
            "crystalbleu_max_order": max(1, int(float(row.get("crystalbleu_max_order") or 4))),
            "crystalbleu_trivial_ngram_count": max(
                0, int(float(row.get("crystalbleu_trivial_ngram_count") or 50))
            ),
            "levenshtein_weights": effective_levenshtein_weights,
            "jaro_winkler_prefix_weight": max(
                0.0, min(0.25, float(row.get("jaro_winkler_prefix_weight") or 0.1))
            ),
            "threshold": max(0.0, float(row.get("threshold") or 0.35)),
            "number_results": max(1, int(float(row.get("number_results") or 50))),
            "feature_weights": feature_weights,
        }
        configs.append({"run_name": run_name, "options": options})
    return configs


def _compact_suite_model_name(model_name):
    value = str(model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    short_name = value.split("/")[-1]
    for old, new in ((" ", "_"), (".", "_"), ("-", "_")):
        short_name = short_name.replace(old, new)
    return short_name.lower() or "model"


def ensure_unique_suite_run_name(base_name, rows):
    cleaned = str(base_name or "").strip() or "comparison"
    existing_names = {
        str(value).strip()
        for value in normalize_suite_rows_frame(rows).get("run_name", pd.Series(dtype=str)).tolist()
        if str(value).strip()
    }
    if cleaned not in existing_names:
        return cleaned

    index = 2
    while f"{cleaned}_{index}" in existing_names:
        index += 1
    return f"{cleaned}_{index}"


def default_suite_run_name(
    rows,
    selected_features,
    selected_preparation,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    code_metric,
    preprocess_mode,
    chunking_method,
):
    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    parts = []

    parts.append(_compact_suite_model_name(model_name))

    if "Embedding" in selected:
        backend_map = {
            "sentence_transformers": "dense",
            "model2vec": "static",
            "pylate": "multivector",
            "auto": "auto",
        }
        normalized_backend = str(vector_backend or "auto").strip() or "auto"
        parts.append(backend_map.get(normalized_backend, "embedding"))
        if normalized_backend != "pylate":
            parts.append(str(similarity_function or "cosine").strip().lower() or "cosine")
        if normalized_backend in {"auto", "sentence_transformers"}:
            parts.append(str(pooling_method or "mean").strip().lower() or "mean")
    if "Levenshtein" in selected:
        parts.append("levenshtein")
    if "Jaro-Winkler" in selected:
        parts.append("jaro")
    if "Code Metric" in selected:
        parts.append(str(code_metric or "code_metric").strip().lower() or "code_metric")
    if "Preprocessing" in selected_steps and str(preprocess_mode or "none").strip() not in ("", "none"):
        parts.append(str(preprocess_mode).strip().lower())
    if "Chunking" in selected_steps and str(chunking_method or "none").strip() not in ("", "none"):
        parts.append(str(chunking_method).strip().lower())

    base_name = "_".join(parts) if parts else "comparison"
    return ensure_unique_suite_run_name(base_name, rows)


def preview_suite_run_name(
    rows,
    selected_features,
    selected_preparation,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    code_metric,
    preprocess_mode,
    chunking_method,
):
    return default_suite_run_name(
        rows,
        selected_features,
        selected_preparation,
        model_name,
        vector_backend,
        similarity_function,
        pooling_method,
        code_metric,
        preprocess_mode,
        chunking_method,
    )


def build_suite_run_row_data(
    rows,
    run_name,
    selected_features,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    max_token_length,
    ws,
    wl,
    wj,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
    threshold,
    number_results,
):
    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    requested_run_name = str(run_name or "").strip() or default_suite_run_name(
        rows,
        selected_features,
        selected_preparation,
        model_name,
        vector_backend,
        similarity_function,
        pooling_method,
        code_metric,
        preprocess_mode,
        chunking_method,
    )
    normalized_run_name = ensure_unique_suite_run_name(requested_run_name, rows)
    normalized_code_metric = code_metric if "Code Metric" in selected else "none"
    normalized_levenshtein_weights = "1,1,1"
    if "Levenshtein" in selected:
        _, normalized_levenshtein_weights = validate_levenshtein_weights_text(levenshtein_weights)

    normalized_codebleu_weights = "0.25,0.25,0.25,0.25"
    if normalized_code_metric.startswith("codebleu"):
        normalized_codebleu_weights = validate_codebleu_component_weights_text(codebleu_component_weights)

    feature_weights = build_feature_weights(
        "Embedding" in selected,
        ws,
        "Levenshtein" in selected,
        wl,
        "Jaro-Winkler" in selected,
        wj,
        normalized_code_metric,
        code_metric_weight,
    )

    row = {
        "run_name": normalized_run_name,
        "model_name": str(model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "vector_backend": str(vector_backend or "auto").strip() or "auto",
        "similarity_function": str(similarity_function or "cosine").strip() or "cosine",
        "pooling_method": str(pooling_method or "mean").strip() or "mean",
        "max_token_length": max(8, int(float(max_token_length or 256))),
        "preprocess_mode": preprocess_mode if "Preprocessing" in selected_steps else "none",
        "chunking_method": chunking_method if "Chunking" in selected_steps else "none",
        "chunk_size": max(10, int(float(chunk_size or 120))),
        "chunk_overlap": max(0, int(float(chunk_overlap or 0))),
        "max_chunks": max(0, int(float(max_chunks or 0))),
        "chunk_aggregation": str(chunk_aggregation or "mean").strip() or "mean",
        "chunk_language": str(chunk_language or "text").strip() or "text",
        "chunker_options": str(chunker_options or "").strip(),
        "code_metric": normalized_code_metric,
        "code_metric_weight": feature_weights.get("code_metric", 0.0),
        "code_language": str(code_language or "python").strip() or "python",
        "codebleu_component_weights": normalized_codebleu_weights,
        "crystalbleu_max_order": max(1, int(float(crystalbleu_max_order or 4))),
        "crystalbleu_trivial_ngram_count": max(0, int(float(crystalbleu_trivial_ngram_count or 50))),
        "semantic_weight": feature_weights.get("semantic", 0.0),
        "levenshtein_weight": feature_weights.get("levenshtein", 0.0),
        "levenshtein_weights": normalized_levenshtein_weights,
        "jaro_winkler_weight": feature_weights.get("jaro_winkler", 0.0),
        "jaro_winkler_prefix_weight": max(0.0, min(0.25, float(jaro_winkler_prefix_weight or 0.1))),
        "threshold": max(0.0, float(threshold or 0.35)),
        "number_results": max(1, int(float(number_results or 50))),
    }
    return row


def append_suite_run_gradio(
    rows,
    run_name,
    selected_features,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    max_token_length,
    ws,
    wl,
    wj,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
    threshold,
    number_results,
):
    row = build_suite_run_row_data(
        rows,
        run_name,
        selected_features,
        model_name,
        vector_backend,
        similarity_function,
        pooling_method,
        max_token_length,
        ws,
        wl,
        wj,
        levenshtein_weights,
        jaro_winkler_prefix_weight,
        code_metric,
        code_metric_weight,
        code_language,
        codebleu_component_weights,
        crystalbleu_max_order,
        crystalbleu_trivial_ngram_count,
        selected_preparation,
        preprocess_mode,
        chunking_method,
        chunk_size,
        chunk_overlap,
        max_chunks,
        chunk_aggregation,
        chunk_language,
        chunker_options,
        threshold,
        number_results,
    )

    frame = normalize_suite_rows_frame(rows)
    if frame.empty:
        updated = pd.DataFrame([row], columns=SUITE_COLUMNS)
    else:
        updated = pd.concat([frame, pd.DataFrame([row], columns=SUITE_COLUMNS)], ignore_index=True)
    return (
        updated,
        suite_runs_overview_html(updated),
        profile_status_html(f'Saved "{row["run_name"]}" to the current run list.'),
        preview_suite_run_name(
            updated,
            selected_features,
            selected_preparation,
            model_name,
            vector_backend,
            similarity_function,
            pooling_method,
            code_metric,
            preprocess_mode,
            chunking_method,
        ),
    )

def suite_runs_overview_html(rows):
    frame = normalize_suite_rows_frame(rows)
    if frame.empty:
        return "<p><strong>Configured runs:</strong> none.</p>"

    names = [str(value).strip() for value in frame["run_name"].tolist() if str(value).strip()]
    preview = ", ".join(names[:4])
    if len(names) > 4:
        preview = f"{preview}, +{len(names) - 4} more"
    return f"<p><strong>Configured runs:</strong> {len(names)} | {preview}</p>"


def reset_suite_runs():
    rows = empty_suite_rows()
    return (
        rows,
        suite_runs_overview_html(rows),
        profile_status_html("Run list cleared."),
        preview_suite_run_name(
            rows,
            ["Embedding", "Levenshtein"],
            [],
            DEFAULT_MODEL,
            "auto",
            "cosine",
            "mean",
            "codebleu",
            "basic",
            "code",
        ),
    )


def empty_suite_summary_html():
    return "<p><strong>Comparison Suite:</strong> No runs executed yet.</p>"


def suite_summary_html(summary):
    if summary is None or summary.empty:
        return empty_suite_summary_html()

    best = summary.iloc[0]
    return (
        f"<p><strong>Best run:</strong> {best['run_name']} | "
        f"<strong>Runs:</strong> {len(summary)} | "
        f"<strong>Best mean:</strong> {float(summary['mean_score'].max()):.3f} | "
        f"<strong>Best max:</strong> {float(summary['max_score'].max()):.3f}</p>"
    )


def load_suite_details(run_name, details_store):
    rows = list((details_store or {}).get(run_name) or [])
    if not rows:
        return pd.DataFrame(columns=["file_name_1", "file_name_2", "similarity_score"])
    return pd.DataFrame(rows)

def run_suite_gradio(
    zipped_file,
    run_sheet_rows,
    output_format,
    run_name,
    selected_features,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    max_token_length,
    ws,
    wl,
    wj,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
    threshold,
    number_results,
):
    saved_frame = normalize_suite_rows_frame(run_sheet_rows)
    if zipped_file is None:
        return (
            empty_suite_summary_html(),
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            pd.DataFrame(columns=["file_name_1", "file_name_2", "similarity_score"]),
            {},
            None,
            None,
            None,
            saved_frame,
            suite_runs_overview_html(saved_frame),
            profile_status_html("Upload a ZIP file first."),
        )

    execution_frame = saved_frame
    if saved_frame.empty:
        row = build_suite_run_row_data(
            saved_frame,
            run_name,
            selected_features,
            model_name,
            vector_backend,
            similarity_function,
            pooling_method,
            max_token_length,
            ws,
            wl,
            wj,
            levenshtein_weights,
            jaro_winkler_prefix_weight,
            code_metric,
            code_metric_weight,
            code_language,
            codebleu_component_weights,
            crystalbleu_max_order,
            crystalbleu_trivial_ngram_count,
            selected_preparation,
            preprocess_mode,
            chunking_method,
            chunk_size,
            chunk_overlap,
            max_chunks,
            chunk_aggregation,
            chunk_language,
            chunker_options,
            threshold,
            number_results,
        )
        execution_frame = pd.DataFrame([row], columns=SUITE_COLUMNS)
        status_html = profile_status_html(
            f'Ran the current settings as "{row["run_name"]}". Click "Save Run" if you want to keep it in the suite.'
        )
    else:
        status_html = profile_status_html(f"Completed {len(saved_frame)} saved run(s).")

    run_configs = suite_rows_to_configs(execution_frame)

    export_root = tempfile.mkdtemp(prefix="matheel-suite-")
    summary_name = f"comparison_suite_summary.{output_format}"
    summary_export_path = os.path.join(export_root, summary_name)
    details_dir = os.path.join(export_root, "comparison_suite_details")
    summary, result_frames = run_comparison_suite(
        zipped_file,
        run_configs,
        summary_out=summary_export_path,
        details_dir=details_dir,
        output_format=output_format,
    )
    details_store = {
        run_name: frame.to_dict(orient="records")
        for run_name, frame in result_frames.items()
    }
    first_run = summary.iloc[0]["run_name"] if not summary.empty else None
    first_details = load_suite_details(first_run, details_store)
    details_zip_path = os.path.join(export_root, "comparison_suite_details.zip")
    with zipfile.ZipFile(details_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for run_name, frame in result_frames.items():
            detail_path = os.path.join(export_root, f"{run_name}.csv")
            frame.to_csv(detail_path, index=False)
            archive.write(detail_path, arcname=os.path.basename(detail_path))
    run_sheet_export = os.path.join(export_root, "comparison_suite_runs.json")
    with open(run_sheet_export, "w", encoding="utf-8") as handle:
        json.dump(run_configs, handle, indent=2)
    return (
        suite_summary_html(summary),
        summary,
        gr.update(choices=list(summary["run_name"]) if not summary.empty else [], value=first_run),
        first_details,
        details_store,
        summary_export_path,
        details_zip_path,
        run_sheet_export,
        saved_frame,
        suite_runs_overview_html(saved_frame),
        status_html,
    )


def score_card_html(score, vector_backend="sentence_transformers", runtime_device="auto", code_metric="none"):
    numeric_score = float(score)
    return f"<p><strong>Similarity score:</strong> {numeric_score:.4f}</p>"


def empty_summary_html():
    return "<p><strong>Collection:</strong> No results yet.</p>"


def results_summary_html(results, vector_backend, code_metric, chunking_method, runtime_device):
    if results is None or results.empty:
        return "<p><strong>Collection:</strong> No pairs met the current threshold.</p>"

    scores = results["similarity_score"].astype(float)
    top_row = results.iloc[0]
    return (
        f"<p><strong>Top pair:</strong> {top_row['file_name_1']} vs {top_row['file_name_2']} | "
        f"<strong>Pairs:</strong> {len(results)} | "
        f"<strong>Average:</strong> {scores.mean():.3f} | "
        f"<strong>Top score:</strong> {scores.max():.3f}</p>"
    )


def calculate_similarity_gradio(
    code1,
    code2,
    selected_features,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    max_token_length,
    runtime_device,
    ws,
    wl,
    wj,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
):
    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    use_semantic = "Embedding" in selected
    use_levenshtein = "Levenshtein" in selected
    use_jaro_winkler = "Jaro-Winkler" in selected
    use_code_metric = "Code Metric" in selected
    effective_preprocess_mode = preprocess_mode if "Preprocessing" in selected_steps else "none"
    effective_chunking_method = chunking_method if "Chunking" in selected_steps else "none"
    effective_code_metric = code_metric if use_code_metric else "none"
    effective_code_metric_weight = code_metric_weight if use_code_metric else 0.0
    effective_levenshtein_weights = "1,1,1"
    if use_levenshtein:
        effective_levenshtein_weights, _ = validate_levenshtein_weights_text(levenshtein_weights)

    effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
    if effective_code_metric.startswith("codebleu"):
        effective_codebleu_component_weights = validate_codebleu_component_weights_text(
            codebleu_component_weights
        )

    feature_weights = build_feature_weights(
        use_semantic,
        ws,
        use_levenshtein,
        wl,
        use_jaro_winkler,
        wj,
        effective_code_metric,
        effective_code_metric_weight,
    )
    score = calculate_similarity(
        code1,
        code2,
        model_name=model_name,
        feature_weights=feature_weights,
        preprocess_mode=effective_preprocess_mode,
        chunking_method=effective_chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_aggregation=chunk_aggregation,
        chunk_language=chunk_language,
        chunker_options=chunker_options,
        code_metric=effective_code_metric,
        code_metric_weight=effective_code_metric_weight,
        code_language=code_language,
        codebleu_component_weights=effective_codebleu_component_weights,
        crystalbleu_max_order=crystalbleu_max_order,
        crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        levenshtein_weights=effective_levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        vector_backend=vector_backend,
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
        device=runtime_device,
    )
    return score_card_html(
        score,
        vector_backend=vector_backend,
        runtime_device=runtime_device,
        code_metric=effective_code_metric,
    )


def get_sim_list_gradio(
    zipped_file,
    selected_features,
    model_name,
    vector_backend,
    similarity_function,
    pooling_method,
    max_token_length,
    runtime_device,
    ws,
    wl,
    wj,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
    threshold,
    number_results,
):
    if zipped_file is None:
        return empty_summary_html(), pd.DataFrame(columns=["file_name_1", "file_name_2", "similarity_score"])

    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    use_semantic = "Embedding" in selected
    use_levenshtein = "Levenshtein" in selected
    use_jaro_winkler = "Jaro-Winkler" in selected
    use_code_metric = "Code Metric" in selected
    effective_preprocess_mode = preprocess_mode if "Preprocessing" in selected_steps else "none"
    effective_chunking_method = chunking_method if "Chunking" in selected_steps else "none"
    effective_code_metric = code_metric if use_code_metric else "none"
    effective_code_metric_weight = code_metric_weight if use_code_metric else 0.0
    effective_levenshtein_weights = "1,1,1"
    if use_levenshtein:
        effective_levenshtein_weights, _ = validate_levenshtein_weights_text(levenshtein_weights)

    effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
    if effective_code_metric.startswith("codebleu"):
        effective_codebleu_component_weights = validate_codebleu_component_weights_text(
            codebleu_component_weights
        )

    feature_weights = build_feature_weights(
        use_semantic,
        ws,
        use_levenshtein,
        wl,
        use_jaro_winkler,
        wj,
        effective_code_metric,
        effective_code_metric_weight,
    )
    results = get_sim_list(
        zipped_file,
        model_name=model_name,
        threshold=threshold,
        number_results=number_results,
        feature_weights=feature_weights,
        preprocess_mode=effective_preprocess_mode,
        chunking_method=effective_chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_aggregation=chunk_aggregation,
        chunk_language=chunk_language,
        chunker_options=chunker_options,
        code_metric=effective_code_metric,
        code_metric_weight=effective_code_metric_weight,
        code_language=code_language,
        codebleu_component_weights=effective_codebleu_component_weights,
        crystalbleu_max_order=crystalbleu_max_order,
        crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        levenshtein_weights=effective_levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        vector_backend=vector_backend,
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
        device=runtime_device,
    )
    return results_summary_html(
        results,
        vector_backend,
        effective_code_metric,
        effective_chunking_method,
        runtime_device,
    ), results


with gr.Blocks(title="Matheel Framework") as demo:
    gr.Markdown(
        "# Matheel Framework\n"
        "Measure code similarity with configurable embedding, lexical, and code-aware metrics. "
        "Select only the parts you want to use, then refine their parameters."
    )
    with gr.Tabs():
        with gr.Tab("Pairwise Comparison"):
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("1. Select the metrics you want. 2. Add preprocessing or chunking if needed. 3. Click Run.")
                    with gr.Row():
                        pair_code1 = gr.Textbox(
                            label="Code A",
                            lines=16,
                            placeholder="Paste the first snippet here...",
                        )
                        pair_code2 = gr.Textbox(
                            label="Code B",
                            lines=16,
                            placeholder="Paste the second snippet here...",
                        )
                    pair_run = gr.Button("Run", variant="primary")
                    pair_output = gr.HTML(value=score_card_html(0.0))

                with gr.Column(scale=5):
                    with gr.Accordion("Features", open=True):
                        pair_features = gr.CheckboxGroup(
                            choices=["Embedding", "Levenshtein", "Jaro-Winkler", "Code Metric"],
                            value=["Embedding", "Levenshtein"],
                            label="Metrics",
                        )
                        with gr.Group(visible=True) as pair_embedding_group:
                            pair_model = HuggingfaceHubSearch(
                                value=DEFAULT_MODEL,
                                label="Embedding Model",
                                placeholder="Search Hugging Face models",
                                search_type="model",
                            )
                            pair_model_status = gr.HTML(value=model_status_html())
                            pair_vector_backend = gr.Dropdown(
                                choices=list(available_vector_backends()),
                                value="auto",
                                label="Vector Backend",
                            )
                            with gr.Group(visible=True) as pair_similarity_group:
                                pair_similarity_function = gr.Dropdown(
                                    choices=list(available_similarity_functions()),
                                    value="cosine",
                                    label="Similarity Function",
                                )
                            with gr.Group(visible=True) as pair_pooling_group:
                                pair_pooling_method = gr.Dropdown(
                                    choices=list(available_pooling_methods()),
                                    value="mean",
                                    label="Pooling Method",
                                )
                            pair_max_token_length = gr.Slider(8, 512, value=256, label="Max Tokens", step=1)
                            pair_runtime_device = gr.Dropdown(
                                choices=list(DEVICE_CHOICES),
                                value="auto",
                                label="Runtime Device",
                            )
                            pair_ws = gr.Slider(0, 1, value=0.7, label="Embedding Weight", step=0.05)
                        with gr.Group(visible=True) as pair_levenshtein_group:
                            pair_wl = gr.Slider(0, 1, value=0.3, label="Levenshtein Weight", step=0.05)
                            pair_levenshtein_weights = gr.Textbox(
                                value="1,1,1",
                                label="Insert, Delete, Substitute",
                            )
                        with gr.Group(visible=False) as pair_jaro_group:
                            pair_wj = gr.Slider(0, 1, value=0.1, label="Jaro-Winkler Weight", step=0.05)
                            pair_jaro_prefix_weight = gr.Slider(
                                0.0, 0.25, value=0.1, label="Prefix Weight", step=0.01
                            )
                        with gr.Group(visible=False) as pair_code_group:
                            pair_code_metric = gr.Dropdown(
                                choices=[metric for metric in available_code_metrics() if metric != "none"],
                                value="codebleu",
                                label="Code Metric",
                            )
                            pair_code_metric_weight = gr.Slider(
                                0, 1, value=0.2, label="Code Metric Weight", step=0.05
                            )
                            pair_code_language = gr.Dropdown(
                                choices=list(available_code_metric_languages()),
                                value="python",
                                label="Code Language",
                            )
                            with gr.Group(visible=True) as pair_codebleu_group:
                                pair_codebleu_component_weights = gr.Textbox(
                                    value="0.25,0.25,0.25,0.25",
                                    label="CodeBLEU Weights",
                                )
                            with gr.Group(visible=False) as pair_crystal_group:
                                pair_crystalbleu_max_order = gr.Slider(
                                    1, 8, value=4, label="Max N-gram Order", step=1
                                )
                                pair_crystalbleu_trivial_ngram_count = gr.Slider(
                                    0, 500, value=50, label="Ignored Frequent N-grams", step=5
                                )

                    with gr.Accordion("Code Preparation", open=False):
                        pair_code_preparation = gr.CheckboxGroup(
                            choices=["Preprocessing", "Chunking"],
                            value=[],
                            label="Steps",
                        )
                        with gr.Group(visible=False) as pair_preprocess_group:
                            pair_preprocess_mode = gr.Dropdown(
                                choices=list(PREPROCESSING_UI_CHOICES),
                                value=(PREPROCESSING_UI_CHOICES[0] if PREPROCESSING_UI_CHOICES else "basic"),
                                label="Preprocessing Mode",
                            )
                        with gr.Group(visible=False) as pair_chunk_group:
                            pair_chunking_method = gr.Dropdown(
                                choices=list(CHONKIE_UI_METHODS),
                                value=(CHONKIE_UI_METHODS[0] if CHONKIE_UI_METHODS else "code"),
                                label="Chunking Method",
                            )
                            pair_chunk_size = gr.Slider(10, 400, value=120, label="Chunk Size", step=10)
                            pair_chunk_overlap = gr.Slider(0, 200, value=0, label="Chunk Overlap", step=5)
                            pair_max_chunks = gr.Slider(0, 20, value=0, label="Max Chunks per File", step=1)
                            pair_chunk_aggregation = gr.Dropdown(
                                choices=list(available_chunk_aggregations()),
                                value="mean",
                                label="Chunk Aggregation",
                            )
                            pair_chunk_language = gr.Dropdown(
                                choices=list(CHUNK_LANGUAGE_CHOICES),
                                value="text",
                                label="Chunk Language",
                            )
                            pair_chunker_options = gr.Textbox(
                                value="",
                                label="Chunker Options",
                                placeholder="include_line_numbers=true",
                            )

            pair_features.change(
                update_feature_sections,
                inputs=pair_features,
                outputs=[
                    pair_embedding_group,
                    pair_levenshtein_group,
                    pair_jaro_group,
                    pair_code_group,
                ],
            )
            pair_code_preparation.change(
                update_code_preparation_sections,
                inputs=pair_code_preparation,
                outputs=[pair_preprocess_group, pair_chunk_group],
            )
            pair_code_metric.change(
                update_code_metric_sections,
                inputs=pair_code_metric,
                outputs=[pair_codebleu_group, pair_crystal_group],
            )

            pair_run.click(
                calculate_similarity_gradio,
                inputs=[
                    pair_code1,
                    pair_code2,
                    pair_features,
                    pair_model,
                    pair_vector_backend,
                    pair_similarity_function,
                    pair_pooling_method,
                    pair_max_token_length,
                    pair_runtime_device,
                    pair_ws,
                    pair_wl,
                    pair_wj,
                    pair_levenshtein_weights,
                    pair_jaro_prefix_weight,
                    pair_code_metric,
                    pair_code_metric_weight,
                    pair_code_language,
                    pair_codebleu_component_weights,
                    pair_crystalbleu_max_order,
                    pair_crystalbleu_trivial_ngram_count,
                    pair_code_preparation,
                    pair_preprocess_mode,
                    pair_chunking_method,
                    pair_chunk_size,
                    pair_chunk_overlap,
                    pair_max_chunks,
                    pair_chunk_aggregation,
                    pair_chunk_language,
                    pair_chunker_options,
                ],
                outputs=pair_output,
            )

            for component in (
                pair_model,
                pair_vector_backend,
                pair_runtime_device,
            ):
                component.change(
                    sync_model_settings_gradio,
                    inputs=[
                        pair_model,
                        pair_vector_backend,
                        pair_runtime_device,
                        pair_similarity_function,
                        pair_pooling_method,
                        pair_max_token_length,
                    ],
                    outputs=[
                        pair_vector_backend,
                        pair_max_token_length,
                        pair_model_status,
                        pair_similarity_group,
                        pair_pooling_group,
                    ],
                )

        with gr.Tab("Collection Comparison"):
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("1. Upload one ZIP file. 2. Select the metrics and preparation steps. 3. Click Run.")
                    collection_file = gr.File(label="Code Collection (.zip)", file_types=[".zip"])
                    collection_run = gr.Button("Run", variant="primary")
                    collection_summary = gr.HTML(value=empty_summary_html())
                    collection_output = gr.Dataframe(
                        label="Ranked Pairs",
                        wrap=True,
                        interactive=False,
                    )

                with gr.Column(scale=5):
                    with gr.Accordion("Features", open=True):
                        collection_features = gr.CheckboxGroup(
                            choices=["Embedding", "Levenshtein", "Jaro-Winkler", "Code Metric"],
                            value=["Embedding", "Levenshtein"],
                            label="Metrics",
                        )
                        with gr.Group(visible=True) as collection_embedding_group:
                            collection_model = HuggingfaceHubSearch(
                                value=DEFAULT_MODEL,
                                label="Embedding Model",
                                placeholder="Search Hugging Face models",
                                search_type="model",
                            )
                            collection_model_status = gr.HTML(value=model_status_html())
                            collection_vector_backend = gr.Dropdown(
                                choices=list(available_vector_backends()),
                                value="auto",
                                label="Vector Backend",
                            )
                            with gr.Group(visible=True) as collection_similarity_group:
                                collection_similarity_function = gr.Dropdown(
                                    choices=list(available_similarity_functions()),
                                    value="cosine",
                                    label="Similarity Function",
                                )
                            with gr.Group(visible=True) as collection_pooling_group:
                                collection_pooling_method = gr.Dropdown(
                                    choices=list(available_pooling_methods()),
                                    value="mean",
                                    label="Pooling Method",
                                )
                            collection_max_token_length = gr.Slider(
                                8, 512, value=256, label="Max Tokens", step=1
                            )
                            collection_runtime_device = gr.Dropdown(
                                choices=list(DEVICE_CHOICES),
                                value="auto",
                                label="Runtime Device",
                            )
                            collection_ws = gr.Slider(
                                0, 1, value=0.7, label="Embedding Weight", step=0.05
                            )
                        with gr.Group(visible=True) as collection_levenshtein_group:
                            collection_wl = gr.Slider(
                                0, 1, value=0.3, label="Levenshtein Weight", step=0.05
                            )
                            collection_levenshtein_weights = gr.Textbox(
                                value="1,1,1",
                                label="Insert, Delete, Substitute",
                            )
                        with gr.Group(visible=False) as collection_jaro_group:
                            collection_wj = gr.Slider(
                                0, 1, value=0.1, label="Jaro-Winkler Weight", step=0.05
                            )
                            collection_jaro_prefix_weight = gr.Slider(
                                0.0, 0.25, value=0.1, label="Prefix Weight", step=0.01
                            )
                        with gr.Group(visible=False) as collection_code_group:
                            collection_code_metric = gr.Dropdown(
                                choices=[metric for metric in available_code_metrics() if metric != "none"],
                                value="codebleu",
                                label="Code Metric",
                            )
                            collection_code_metric_weight = gr.Slider(
                                0, 1, value=0.2, label="Code Metric Weight", step=0.05
                            )
                            collection_code_language = gr.Dropdown(
                                choices=list(available_code_metric_languages()),
                                value="python",
                                label="Code Language",
                            )
                            with gr.Group(visible=True) as collection_codebleu_group:
                                collection_codebleu_component_weights = gr.Textbox(
                                    value="0.25,0.25,0.25,0.25",
                                    label="CodeBLEU Weights",
                                )
                            with gr.Group(visible=False) as collection_crystal_group:
                                collection_crystalbleu_max_order = gr.Slider(
                                    1, 8, value=4, label="Max N-gram Order", step=1
                                )
                                collection_crystalbleu_trivial_ngram_count = gr.Slider(
                                    0, 500, value=50, label="Ignored Frequent N-grams", step=5
                                )

                    with gr.Accordion("Code Preparation", open=False):
                        collection_code_preparation = gr.CheckboxGroup(
                            choices=["Preprocessing", "Chunking"],
                            value=[],
                            label="Steps",
                        )
                        with gr.Group(visible=False) as collection_preprocess_group:
                            collection_preprocess_mode = gr.Dropdown(
                                choices=list(PREPROCESSING_UI_CHOICES),
                                value=(PREPROCESSING_UI_CHOICES[0] if PREPROCESSING_UI_CHOICES else "basic"),
                                label="Preprocessing Mode",
                            )
                        with gr.Group(visible=False) as collection_chunk_group:
                            collection_chunking_method = gr.Dropdown(
                                choices=list(CHONKIE_UI_METHODS),
                                value=(CHONKIE_UI_METHODS[0] if CHONKIE_UI_METHODS else "code"),
                                label="Chunking Method",
                            )
                            collection_chunk_size = gr.Slider(
                                10, 400, value=120, label="Chunk Size", step=10
                            )
                            collection_chunk_overlap = gr.Slider(
                                0, 200, value=0, label="Chunk Overlap", step=5
                            )
                            collection_max_chunks = gr.Slider(
                                0, 20, value=0, label="Max Chunks per File", step=1
                            )
                            collection_chunk_aggregation = gr.Dropdown(
                                choices=list(available_chunk_aggregations()),
                                value="mean",
                                label="Chunk Aggregation",
                            )
                            collection_chunk_language = gr.Dropdown(
                                choices=list(CHUNK_LANGUAGE_CHOICES),
                                value="text",
                                label="Chunk Language",
                            )
                            collection_chunker_options = gr.Textbox(
                                value="",
                                label="Chunker Options",
                                placeholder="include_line_numbers=true",
                            )

                    with gr.Accordion("Display", open=False):
                        collection_threshold = gr.Slider(0, 1, value=0.35, label="Threshold", step=0.01)
                        collection_number_results = gr.Slider(
                            1, 1000, value=50, label="Max Pairs", step=1
                        )

            collection_features.change(
                update_feature_sections,
                inputs=collection_features,
                outputs=[
                    collection_embedding_group,
                    collection_levenshtein_group,
                    collection_jaro_group,
                    collection_code_group,
                ],
            )
            collection_code_preparation.change(
                update_code_preparation_sections,
                inputs=collection_code_preparation,
                outputs=[collection_preprocess_group, collection_chunk_group],
            )
            collection_code_metric.change(
                update_code_metric_sections,
                inputs=collection_code_metric,
                outputs=[collection_codebleu_group, collection_crystal_group],
            )

            collection_run.click(
                get_sim_list_gradio,
                inputs=[
                    collection_file,
                    collection_features,
                    collection_model,
                    collection_vector_backend,
                    collection_similarity_function,
                    collection_pooling_method,
                    collection_max_token_length,
                    collection_runtime_device,
                    collection_ws,
                    collection_wl,
                    collection_wj,
                    collection_levenshtein_weights,
                    collection_jaro_prefix_weight,
                    collection_code_metric,
                    collection_code_metric_weight,
                    collection_code_language,
                    collection_codebleu_component_weights,
                    collection_crystalbleu_max_order,
                    collection_crystalbleu_trivial_ngram_count,
                    collection_code_preparation,
                    collection_preprocess_mode,
                    collection_chunking_method,
                    collection_chunk_size,
                    collection_chunk_overlap,
                    collection_max_chunks,
                    collection_chunk_aggregation,
                    collection_chunk_language,
                    collection_chunker_options,
                    collection_threshold,
                    collection_number_results,
                ],
                outputs=[collection_summary, collection_output],
            )

            for component in (
                collection_model,
                collection_vector_backend,
                collection_runtime_device,
            ):
                component.change(
                    sync_model_settings_gradio,
                    inputs=[
                        collection_model,
                        collection_vector_backend,
                        collection_runtime_device,
                        collection_similarity_function,
                        collection_pooling_method,
                        collection_max_token_length,
                    ],
                    outputs=[
                        collection_vector_backend,
                        collection_max_token_length,
                        collection_model_status,
                        collection_similarity_group,
                        collection_pooling_group,
                    ],
                )

        with gr.Tab("Comparison Suite"):
            suite_details_store = gr.State({})
            suite_runs_state = gr.State(empty_suite_rows())

            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("1. Set the current configuration. 2. Save Run to append it. 3. Repeat if needed. 4. Run the suite. If nothing is saved, the current configuration runs once.")
                    suite_file = gr.File(label="Code Collection (.zip)", file_types=[".zip"])
                    suite_summary = gr.HTML(value=empty_suite_summary_html())
                    suite_output = gr.Dataframe(
                        label="Suite Summary",
                        wrap=True,
                        interactive=False,
                    )
                    suite_run_name = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Run Details",
                    )
                    suite_details = gr.Dataframe(
                        label="Selected Run Details",
                        wrap=True,
                        interactive=False,
                    )
                    suite_runs_overview = gr.HTML(value=suite_runs_overview_html(empty_suite_rows()))

                with gr.Column(scale=5):
                    with gr.Accordion("Features", open=True):
                        suite_features = gr.CheckboxGroup(
                            choices=["Embedding", "Levenshtein", "Jaro-Winkler", "Code Metric"],
                            value=["Embedding", "Levenshtein"],
                            label="Metrics",
                        )
                        with gr.Group(visible=True) as suite_embedding_group:
                            suite_model = HuggingfaceHubSearch(
                                value=DEFAULT_MODEL,
                                label="Embedding Model",
                                placeholder="Search Hugging Face models",
                                search_type="model",
                            )
                            suite_model_status = gr.HTML(value=model_status_html())
                            suite_vector_backend = gr.Dropdown(
                                choices=list(available_vector_backends()),
                                value="auto",
                                label="Vector Backend",
                            )
                            with gr.Group(visible=True) as suite_similarity_group:
                                suite_similarity_function = gr.Dropdown(
                                    choices=list(available_similarity_functions()),
                                    value="cosine",
                                    label="Similarity Function",
                                )
                            with gr.Group(visible=True) as suite_pooling_group:
                                suite_pooling_method = gr.Dropdown(
                                    choices=list(available_pooling_methods()),
                                    value="mean",
                                    label="Pooling Method",
                                )
                            suite_max_token_length = gr.Slider(
                                8, 512, value=256, label="Max Tokens", step=1
                            )
                            suite_runtime_device = gr.Dropdown(
                                choices=list(DEVICE_CHOICES),
                                value="auto",
                                label="Runtime Device",
                            )
                            suite_ws = gr.Slider(0, 1, value=0.7, label="Embedding Weight", step=0.05)
                        with gr.Group(visible=True) as suite_levenshtein_group:
                            suite_wl = gr.Slider(0, 1, value=0.3, label="Levenshtein Weight", step=0.05)
                            suite_levenshtein_weights = gr.Textbox(
                                value="1,1,1",
                                label="Insert, Delete, Substitute",
                            )
                        with gr.Group(visible=False) as suite_jaro_group:
                            suite_wj = gr.Slider(0, 1, value=0.1, label="Jaro-Winkler Weight", step=0.05)
                            suite_jaro_prefix_weight = gr.Slider(
                                0.0, 0.25, value=0.1, label="Prefix Weight", step=0.01
                            )
                        with gr.Group(visible=False) as suite_code_group:
                            suite_code_metric = gr.Dropdown(
                                choices=[metric for metric in available_code_metrics() if metric != "none"],
                                value="codebleu",
                                label="Code Metric",
                            )
                            suite_code_metric_weight = gr.Slider(
                                0, 1, value=0.2, label="Code Metric Weight", step=0.05
                            )
                            suite_code_language = gr.Dropdown(
                                choices=list(available_code_metric_languages()),
                                value="python",
                                label="Code Language",
                            )
                            with gr.Group(visible=True) as suite_codebleu_group:
                                suite_codebleu_component_weights = gr.Textbox(
                                    value="0.25,0.25,0.25,0.25",
                                    label="CodeBLEU Weights",
                                )
                            with gr.Group(visible=False) as suite_crystal_group:
                                suite_crystalbleu_max_order = gr.Slider(
                                    1, 8, value=4, label="Max N-gram Order", step=1
                                )
                                suite_crystalbleu_trivial_ngram_count = gr.Slider(
                                    0, 500, value=50, label="Ignored Frequent N-grams", step=5
                                )

                    with gr.Accordion("Code Preparation", open=False):
                        suite_code_preparation = gr.CheckboxGroup(
                            choices=["Preprocessing", "Chunking"],
                            value=[],
                            label="Steps",
                        )
                        with gr.Group(visible=False) as suite_preprocess_group:
                            suite_preprocess_mode = gr.Dropdown(
                                choices=list(PREPROCESSING_UI_CHOICES),
                                value=(PREPROCESSING_UI_CHOICES[0] if PREPROCESSING_UI_CHOICES else "basic"),
                                label="Preprocessing Mode",
                            )
                        with gr.Group(visible=False) as suite_chunk_group:
                            suite_chunking_method = gr.Dropdown(
                                choices=list(CHONKIE_UI_METHODS),
                                value=(CHONKIE_UI_METHODS[0] if CHONKIE_UI_METHODS else "code"),
                                label="Chunking Method",
                            )
                            suite_chunk_size = gr.Slider(10, 400, value=120, label="Chunk Size", step=10)
                            suite_chunk_overlap = gr.Slider(0, 200, value=0, label="Chunk Overlap", step=5)
                            suite_max_chunks = gr.Slider(0, 20, value=0, label="Max Chunks per File", step=1)
                            suite_chunk_aggregation = gr.Dropdown(
                                choices=list(available_chunk_aggregations()),
                                value="mean",
                                label="Chunk Aggregation",
                            )
                            suite_chunk_language = gr.Dropdown(
                                choices=list(CHUNK_LANGUAGE_CHOICES),
                                value="text",
                                label="Chunk Language",
                            )
                            suite_chunker_options = gr.Textbox(
                                value="",
                                label="Chunker Options",
                                placeholder="include_line_numbers=true",
                            )

                    with gr.Group():
                        suite_run_name_input = gr.Textbox(
                            value=preview_suite_run_name(
                                empty_suite_rows(),
                                ["Embedding", "Levenshtein"],
                                [],
                                DEFAULT_MODEL,
                                "auto",
                                "cosine",
                                "mean",
                                "codebleu",
                                "basic",
                                "code",
                            ),
                            label="Run Name",
                            placeholder="Generated automatically from the current configuration",
                        )
                        with gr.Row():
                            suite_add_run = gr.Button("Save Run", variant="secondary")
                            suite_template = gr.Button("Clear Runs", variant="secondary")

                    with gr.Accordion("Display", open=False):
                        suite_threshold = gr.Slider(0, 1, value=0.35, label="Threshold", step=0.01)
                        suite_number_results = gr.Slider(
                            1, 1000, value=50, label="Max Pairs", step=1
                        )

                    suite_run = gr.Button("Run Comparison Suite", variant="primary")
                    suite_output_format = gr.Dropdown(
                        choices=["csv", "json"],
                        value="csv",
                        label="Summary File Format",
                    )
                    suite_summary_download = gr.File(label="Summary Download")
                    suite_details_download = gr.File(label="Details ZIP")
                    suite_runs_download = gr.File(label="Run JSON")
                    suite_status = gr.HTML(
                        value=profile_status_html("Save Run appends the current configuration. Run Comparison Suite executes the saved list, or the current configuration once if the list is empty.")
                    )

            suite_features.change(
                update_feature_sections,
                inputs=suite_features,
                outputs=[
                    suite_embedding_group,
                    suite_levenshtein_group,
                    suite_jaro_group,
                    suite_code_group,
                ],
            )
            suite_code_preparation.change(
                update_code_preparation_sections,
                inputs=suite_code_preparation,
                outputs=[suite_preprocess_group, suite_chunk_group],
            )
            suite_code_metric.change(
                update_code_metric_sections,
                inputs=suite_code_metric,
                outputs=[suite_codebleu_group, suite_crystal_group],
            )

            suite_add_run.click(
                append_suite_run_gradio,
                inputs=[
                    suite_runs_state,
                    suite_run_name_input,
                    suite_features,
                    suite_model,
                    suite_vector_backend,
                    suite_similarity_function,
                    suite_pooling_method,
                    suite_max_token_length,
                    suite_ws,
                    suite_wl,
                    suite_wj,
                    suite_levenshtein_weights,
                    suite_jaro_prefix_weight,
                    suite_code_metric,
                    suite_code_metric_weight,
                    suite_code_language,
                    suite_codebleu_component_weights,
                    suite_crystalbleu_max_order,
                    suite_crystalbleu_trivial_ngram_count,
                    suite_code_preparation,
                    suite_preprocess_mode,
                    suite_chunking_method,
                    suite_chunk_size,
                    suite_chunk_overlap,
                    suite_max_chunks,
                    suite_chunk_aggregation,
                    suite_chunk_language,
                    suite_chunker_options,
                    suite_threshold,
                    suite_number_results,
                ],
                outputs=[suite_runs_state, suite_runs_overview, suite_status, suite_run_name_input],
            )
            suite_template.click(
                reset_suite_runs,
                outputs=[suite_runs_state, suite_runs_overview, suite_status, suite_run_name_input],
            )
            suite_run.click(
                run_suite_gradio,
                inputs=[
                    suite_file,
                    suite_runs_state,
                    suite_output_format,
                    suite_run_name_input,
                    suite_features,
                    suite_model,
                    suite_vector_backend,
                    suite_similarity_function,
                    suite_pooling_method,
                    suite_max_token_length,
                    suite_ws,
                    suite_wl,
                    suite_wj,
                    suite_levenshtein_weights,
                    suite_jaro_prefix_weight,
                    suite_code_metric,
                    suite_code_metric_weight,
                    suite_code_language,
                    suite_codebleu_component_weights,
                    suite_crystalbleu_max_order,
                    suite_crystalbleu_trivial_ngram_count,
                    suite_code_preparation,
                    suite_preprocess_mode,
                    suite_chunking_method,
                    suite_chunk_size,
                    suite_chunk_overlap,
                    suite_max_chunks,
                    suite_chunk_aggregation,
                    suite_chunk_language,
                    suite_chunker_options,
                    suite_threshold,
                    suite_number_results,
                ],
                outputs=[
                    suite_summary,
                    suite_output,
                    suite_run_name,
                    suite_details,
                    suite_details_store,
                    suite_summary_download,
                    suite_details_download,
                    suite_runs_download,
                    suite_runs_state,
                    suite_runs_overview,
                    suite_status,
                ],
            )
            suite_run_name.change(
                load_suite_details,
                inputs=[suite_run_name, suite_details_store],
                outputs=suite_details,
            )
            for component in (
                suite_features,
                suite_code_preparation,
                suite_model,
                suite_vector_backend,
                suite_similarity_function,
                suite_pooling_method,
                suite_code_metric,
                suite_preprocess_mode,
                suite_chunking_method,
            ):
                component.change(
                    preview_suite_run_name,
                    inputs=[
                        suite_runs_state,
                        suite_features,
                        suite_code_preparation,
                        suite_model,
                        suite_vector_backend,
                        suite_similarity_function,
                        suite_pooling_method,
                        suite_code_metric,
                        suite_preprocess_mode,
                        suite_chunking_method,
                    ],
                    outputs=suite_run_name_input,
                )
            for component in (
                suite_model,
                suite_vector_backend,
                suite_runtime_device,
            ):
                component.change(
                    sync_model_settings_gradio,
                    inputs=[
                        suite_model,
                        suite_vector_backend,
                        suite_runtime_device,
                        suite_similarity_function,
                        suite_pooling_method,
                        suite_max_token_length,
                    ],
                    outputs=[
                        suite_vector_backend,
                        suite_max_token_length,
                        suite_model_status,
                        suite_similarity_group,
                        suite_pooling_group,
                    ],
                )

if __name__ == "__main__":
    demo.launch(show_error=True, debug=True)
