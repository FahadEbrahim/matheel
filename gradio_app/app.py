import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import NamedTuple

mpl_config_dir = os.path.join(tempfile.gettempdir(), "matheel-mpl")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

import gradio as gr
import pandas as pd
from gradio_huggingfacehub_search import HuggingfaceHubSearch

try:
    from gradio_app.html_utils import escape_html
except ModuleNotFoundError:
    from html_utils import escape_html

from matheel.chunking import available_chunk_aggregations, available_chunking_methods
from matheel.calibration import write_threshold_tuning_report_artifacts
from matheel.comparison_suite import run_comparison_suite, slugify_run_name
from matheel.code_metrics import available_code_metric_languages, available_code_metrics
from matheel.dataset_validation import write_dataset_validation_report
from matheel.datasets import (
    available_dataset_presets,
    get_dataset_preset,
    load_pair_dataset,
    load_retrieval_dataset,
)
from matheel.evaluation import (
    evaluate_pair_dataset,
    evaluate_pair_resamples,
    evaluate_retrieval_dataset,
    evaluate_retrieval_resamples,
)
from matheel.leaderboard import (
    available_leaderboard_metrics,
    run_leaderboard,
    write_leaderboard_artifacts,
)
from matheel.leaderboard_presets import (
    available_leaderboard_algorithm_presets,
    get_leaderboard_algorithm_preset,
)
from matheel.model_routing import available_vector_backends
from matheel.preprocessing import available_preprocess_modes
from matheel.reports import benchmark_report_html
from matheel.reproducibility import collect_reproducibility_snapshot, write_reproducibility_snapshot
from matheel.resampling import kfold_splits
from matheel._run_metadata import elapsed_seconds_between, perf_counter
from matheel.similarity import (
    DEFAULT_MODEL_NAME,
    available_lexical_tokenizers,
    available_runtime_devices,
    calculate_similarity,
    get_sim_list,
    inspect_model_settings,
)
from matheel.vectors import (
    available_pooling_methods,
    available_similarity_functions,
    detect_model_max_token_length,
)
from matheel.visualization import (
    available_pair_explanation_segment_modes,
    available_projection_methods,
    build_dataset_embedding_map,
    build_pair_explanation,
    pair_explanation_html,
    write_dataset_map_artifacts,
    write_pair_explanation_artifacts,
    write_scored_pair_explanation,
)


DEVICE_CHOICES = ("auto",) + available_runtime_devices()
DEFAULT_MODEL = DEFAULT_MODEL_NAME
CODE_METRIC_CHOICES = [metric for metric in available_code_metrics() if metric != "none"]
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
LEXICAL_TOKENIZER_CHOICES = available_lexical_tokenizers()
DEFAULT_RUBY_MODE = "auto"
DEFAULT_RUBY_GRAPH_TIMEOUT = 1.0
DEFAULT_TSED_COSTS = "1,1,1"
DEFAULT_CODEBERTSCORE_MODEL = "microsoft/codebert-base"
DEFAULT_CODEBERTSCORE_MAX_LENGTH = 0
DEFAULT_CODEBERTSCORE_MODEL_MAX_SEQUENCE = 512
TEMP_WORKSPACE_TTL_SECONDS = int(os.environ.get("MATHEEL_GRADIO_TEMP_TTL_SECONDS", "86400"))
TEMP_WORKSPACE_PREFIXES = (
    "matheel-suite-",
    "matheel-dataset-map-",
    "matheel-pair-explanation-",
    "matheel-leaderboard-inspect-",
    "matheel-ready-leaderboard-upload-",
    "matheel-ready-leaderboard-",
    "matheel-dataset-upload-",
    "matheel-dataset-validation-",
    "matheel-threshold-tuning-",
    "matheel-scored-pair-explanation-",
    "matheel-leaderboard-",
)
FEATURE_UI_CHOICES = [
    "Embedding",
    "Levenshtein",
    "Jaro-Winkler",
    "Winnowing",
    "GST",
    "Code Metric",
]
DEFAULT_FEATURE_SELECTION = ["Embedding", "Levenshtein"]


def cleanup_stale_temp_workspaces(
    temp_root=None,
    ttl_seconds=TEMP_WORKSPACE_TTL_SECONDS,
    prefixes=TEMP_WORKSPACE_PREFIXES,
):
    root = Path(temp_root or tempfile.gettempdir())
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0

    now = time.time()
    cleaned = 0
    for entry in entries:
        if not entry.is_dir() or not any(entry.name.startswith(prefix) for prefix in prefixes):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age < max(0, int(ttl_seconds)):
            continue
        try:
            shutil.rmtree(entry)
        except OSError:
            continue
        cleaned += 1
    return cleaned


def make_temp_workspace(prefix, temp_root=None):
    cleanup_stale_temp_workspaces(temp_root=temp_root)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=temp_root))


def _metric_preset_from_leaderboard_preset(name):
    preset = get_leaderboard_algorithm_preset(name)
    options = dict(preset["similarity_options"])
    weights = dict(options.get("feature_weights") or {})
    features = []
    if weights.get("semantic", 0.0) > 0:
        features.append("Embedding")
    if weights.get("levenshtein", 0.0) > 0:
        features.append("Levenshtein")
    if weights.get("jaro_winkler", 0.0) > 0:
        features.append("Jaro-Winkler")
    if weights.get("winnowing", 0.0) > 0:
        features.append("Winnowing")
    if weights.get("gst", 0.0) > 0:
        features.append("GST")
    if weights.get("code_metric", 0.0) > 0:
        features.append("Code Metric")
    return {
        "features": features or list(DEFAULT_FEATURE_SELECTION),
        "weights": weights,
        "code_metric": str(options.get("code_metric") or "codebleu"),
        "code_metric_weight": float(
            options.get("code_metric_weight") or weights.get("code_metric", 0.0)
        ),
    }


METRIC_PRESETS = {
    name: _metric_preset_from_leaderboard_preset(name)
    for name in available_leaderboard_algorithm_presets()
}
DATASET_TASK_CHOICES = ("Pair Classification", "Retrieval")
DEFAULT_DATASET_TASK = DATASET_TASK_CHOICES[0]
READY_LEADERBOARD_ALGORITHM_CHOICES = tuple(METRIC_PRESETS)
READY_LEADERBOARD_PAIR_METRICS = available_leaderboard_metrics("pair")
READY_LEADERBOARD_RETRIEVAL_METRICS = available_leaderboard_metrics("retrieval")
READY_LEADERBOARD_DEFAULT_METRICS = {
    "pair": "f1",
    "retrieval": "mean_average_precision",
}
READY_LEADERBOARD_TASK_CHOICES = tuple(READY_LEADERBOARD_DEFAULT_METRICS)
READY_LEADERBOARD_ALL_DATASETS = "All datasets"
READY_LEADERBOARD_ALL_ALGORITHMS = "All algorithms"
READY_LEADERBOARD_ALL_SOURCES = "All sources"
READY_LEADERBOARD_AGGREGATE_SORT_COLUMNS = (
    "mean_score",
    "median_score",
    "algorithm_name",
    "dataset_count",
    "sample_count",
    "rank",
)
READY_LEADERBOARD_PER_DATASET_SORT_COLUMNS = (
    "score",
    "dataset_name",
    "algorithm_name",
    "dataset_source",
    "sample_count",
    "rank",
)
READY_LEADERBOARD_SORT_DIRECTIONS = ("Descending", "Ascending")
READY_MADE_LEADERBOARD_PATH = Path(__file__).resolve().parent / "assets" / "ready_leaderboard.json"
THRESHOLD_OPTIMIZE_CHOICES = ("f1", "accuracy", "precision", "recall")
PAIR_DATASET_SCORE_COLUMNS = [
    "left_id",
    "right_id",
    "label",
    "similarity_score",
]
RETRIEVAL_DATASET_SCORE_COLUMNS = [
    "query_id",
    "document_id",
    "relevance",
    "similarity_score",
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "pair_count": "Pair Count",
    "positive_count": "Positive Count",
    "negative_count": "Negative Count",
    "query_count": "Query Count",
    "result_count": "Result Count",
    "relevant_count": "Relevant Count",
    "mean_average_precision": "Mean Average Precision",
    "mean_reciprocal_rank": "Mean Reciprocal Rank",
    "precision_at_k": "Precision at k",
    "recall_at_k": "Recall at k",
    "ndcg_at_k": "NDCG at k",
    "k": "k",
}


def gradio_progress_callback(progress):
    if progress is None:
        return None

    def callback(event):
        total = max(1, int(event.get("total") or 1))
        current = min(total, max(0, int(event.get("current") or 0)))
        description = event.get("message") or event.get("stage") or "Working"
        try:
            progress(current / total, desc=str(description))
        except Exception:
            return

    return callback


SUITE_FEATURE_NAME_MAP = {
    "Embedding": "embedding",
    "Levenshtein": "levenshtein",
    "Jaro-Winkler": "jaro_winkler",
    "Winnowing": "winnowing",
    "GST": "gst",
}
DEFAULT_UI_FEATURE_WEIGHTS = {
    "semantic": 0.7,
    "levenshtein": 0.3,
    "jaro_winkler": 0.1,
    "winnowing": 0.1,
    "gst": 0.1,
    "code_metric": 0.2,
}
APP_CSS = """
.gradio-container {
    width: 100%;
    min-width: 0;
    max-width: 1480px;
    margin: 0 auto;
    padding: 0 18px 48px;
    box-sizing: border-box;
    overflow-x: clip;
}

.matheel-hero {
    position: relative;
    overflow: hidden;
    margin: 6px 0 18px;
    padding: clamp(24px, 4vw, 42px);
    border: 1px solid rgba(148, 210, 196, 0.24);
    border-radius: 22px;
    background:
        radial-gradient(circle at 92% 10%, rgba(45, 212, 191, 0.22), transparent 34%),
        linear-gradient(135deg, #0f172a 0%, #123a42 58%, #0f4c4b 100%);
    color: #f8fafc;
    box-shadow: 0 22px 54px rgba(15, 23, 42, 0.18);
}

.matheel-hero-topline,
.matheel-workflow-kicker {
    margin: 0 0 10px;
    color: #99f6e4;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.matheel-hero h1 {
    margin: 0;
    color: #ffffff;
    font-size: clamp(2rem, 5vw, 3.75rem);
    line-height: 1;
    letter-spacing: -0.045em;
}

.matheel-hero-copy {
    max-width: 760px;
    margin: 14px 0 22px;
    color: #dbeafe;
    font-size: clamp(1rem, 2vw, 1.16rem);
    line-height: 1.65;
}

.matheel-journey {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}

.matheel-journey-step {
    min-width: 0;
    padding: 12px 14px;
    border: 1px solid rgba(226, 232, 240, 0.14);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.28);
    color: #e2e8f0;
    line-height: 1.35;
}

.matheel-journey-step span {
    display: block;
    margin-bottom: 4px;
    color: #5eead4;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

#matheel-workflows {
    gap: 14px;
    min-width: 0;
}

#matheel-workflows > .tab-wrapper {
    position: sticky;
    top: 8px;
    z-index: 20;
}

#matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
    max-width: 100%;
    padding: 6px;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: color-mix(in srgb, var(--background-fill-primary) 92%, transparent);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
    backdrop-filter: blur(14px);
}

#matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] button {
    min-height: 42px;
    border-radius: 10px;
    font-weight: 700;
}

.matheel-workflow-intro {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 0.42fr);
    gap: 18px;
    align-items: center;
    margin: 4px 0 16px;
    padding: 18px 20px;
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: linear-gradient(135deg, var(--block-background-fill), var(--background-fill-secondary));
}

.matheel-workflow-intro h2 {
    margin: 0 0 6px;
    font-size: clamp(1.2rem, 2.5vw, 1.55rem);
    letter-spacing: -0.02em;
}

.matheel-workflow-intro p {
    margin: 0;
    color: var(--body-text-color-subdued);
    line-height: 1.55;
}

.matheel-workflow-kicker {
    color: var(--color-accent, #0f766e);
}

.matheel-workflow-outcome {
    padding: 12px 14px;
    border-left: 3px solid var(--color-accent, #0f766e);
    border-radius: 10px;
    background: var(--background-fill-primary);
    color: var(--body-text-color);
    font-size: 0.9rem;
    line-height: 1.5;
}

.matheel-workflow-outcome strong {
    display: block;
    margin-bottom: 2px;
}

.matheel-workflow-grid {
    align-items: flex-start;
    gap: 18px;
}

#dataset-setup-column {
    order: 1;
}

#dataset-results-column {
    order: 2;
}

.matheel-results-panel,
.matheel-control-panel {
    min-width: 0;
    max-width: 100%;
}

.matheel-control-panel {
    padding: 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: var(--block-background-fill);
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
}

.matheel-control-panel > .form,
.matheel-control-panel > .block {
    margin-bottom: 10px;
}

.matheel-control-panel .wrap {
    border-radius: 12px;
}

.matheel-subtabs > .tab-wrapper > .tab-container {
    margin-bottom: 14px;
    padding: 4px;
    border-radius: 12px;
    background: var(--background-fill-secondary);
}

.matheel-summary,
.matheel-status {
    border: 1px solid #d6d9df;
    border-left: 4px solid #2a7f62;
    border-radius: 8px;
    background: #fbfbf8;
    padding: 10px 12px;
    margin: 0 0 12px;
}

.matheel-status {
    background: #f6faf7;
}

.matheel-empty {
    border-left-color: #8a6f24;
    background: #fffaf0;
}

.matheel-error {
    border-left-color: #b42318;
    background: #fff5f5;
}

.matheel-summary-title {
    display: block;
    margin-bottom: 8px;
    color: #1f2328;
}

.matheel-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px 12px;
}

.matheel-summary-item {
    min-width: 0;
}

.matheel-summary-label {
    display: block;
    color: #5f6773;
    font-size: 0.78rem;
    line-height: 1.2;
}

.matheel-summary-value {
    display: block;
    color: #1f2328;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.matheel-table {
    font-size: 0.92rem;
}

button.primary {
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.18);
}

@media (min-width: 980px) {
    .matheel-control-panel {
        position: sticky;
        top: 76px;
        max-height: calc(100vh - 92px);
        overflow-y: auto;
        overscroll-behavior: contain;
    }
}

@media (max-width: 860px) {
    .gradio-container {
        padding: 0 8px 32px;
    }

    .matheel-journey {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .matheel-workflow-intro {
        grid-template-columns: 1fr;
    }

    #matheel-workflows > .tab-wrapper {
        position: static;
        height: auto !important;
        align-items: stretch !important;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
        position: static !important;
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 4px;
        height: auto !important;
        overflow: visible !important;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] button {
        width: 100% !important;
        min-width: 0;
        justify-content: center;
    }

    .matheel-workflow-grid {
        flex-direction: column;
        flex-wrap: nowrap;
    }

    .matheel-workflow-grid > .column {
        width: 100%;
        min-width: 0;
        flex: 1 1 auto;
    }
}

@media (max-width: 520px) {
    .matheel-hero {
        padding: 22px 18px;
        border-radius: 16px;
    }

    .matheel-journey {
        grid-template-columns: 1fr;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

}
"""

APP_JS = """
() => {
    let animationFrame = null;

    const syncOverflowMenus = () => {
        document.querySelectorAll(".overflow-menu").forEach((overflowMenu) => {
            const trigger = overflowMenu.querySelector(":scope > button");
            const adjacentDropdown = trigger?.nextElementSibling;
            const dropdown =
                overflowMenu.querySelector(":scope > .overflow-dropdown") ||
                (adjacentDropdown?.classList.contains("overflow-dropdown")
                    ? adjacentDropdown
                    : overflowMenu.parentElement?.querySelector(".overflow-dropdown") ||
                      overflowMenu.closest(".tab-wrapper")?.querySelector(".overflow-dropdown"));
            if (!trigger) {
                return;
            }

            trigger.setAttribute("aria-label", "More workflow tabs");
            trigger.setAttribute("aria-haspopup", "menu");
            trigger.setAttribute(
                "aria-expanded",
                String(Boolean(dropdown && !dropdown.classList.contains("hide"))),
            );

            if (dropdown) {
                dropdown.setAttribute("role", "menu");
                dropdown.querySelectorAll("button").forEach((item) => {
                    item.setAttribute("role", "menuitem");
                });
            }
        });
    };

    const scheduleSync = () => {
        if (animationFrame !== null) {
            window.cancelAnimationFrame(animationFrame);
        }
        animationFrame = window.requestAnimationFrame(syncOverflowMenus);
    };

    syncOverflowMenus();
    const observer = new MutationObserver(scheduleSync);
    observer.observe(document.body, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["class", "hidden", "aria-hidden"],
    });
    window.addEventListener("resize", scheduleSync);
}
"""


def app_header_html():
    return (
        '<section class="matheel-hero">'
        '<p class="matheel-hero-topline">Source-code similarity workspace</p>'
        "<h1>Matheel</h1>"
        '<p class="matheel-hero-copy">Move from a quick code comparison to reproducible '
        "dataset evaluation, explanations, and shareable reports without changing tools.</p>"
        '<div class="matheel-journey" aria-label="Recommended workflow">'
        '<div class="matheel-journey-step"><span>01 · Compare</span>Start with two snippets</div>'
        '<div class="matheel-journey-step"><span>02 · Scale</span>Rank a collection or suite</div>'
        '<div class="matheel-journey-step"><span>03 · Evaluate</span>Validate on known datasets</div>'
        '<div class="matheel-journey-step"><span>04 · Explain</span>Export evidence and reports</div>'
        "</div></section>"
    )


def workflow_intro_html(kicker, title, description, outcome):
    return (
        '<section class="matheel-workflow-intro">'
        '<div><p class="matheel-workflow-kicker">'
        f"{escape_html(kicker)}</p><h2>{escape_html(title)}</h2>"
        f"<p>{escape_html(description)}</p></div>"
        '<div class="matheel-workflow-outcome"><strong>What you get</strong>'
        f"{escape_html(outcome)}</div></section>"
    )


def summary_panel_html(title, items, variant="summary"):
    escaped_title = escape_html(title)
    classes = ["matheel-summary"]
    if variant and variant != "summary":
        classes.append(f"matheel-{variant}")
    rendered_items = []
    for label, value in items:
        rendered_items.append(
            '<span class="matheel-summary-item">'
            f'<span class="matheel-summary-label">{escape_html(label)}</span>'
            f'<span class="matheel-summary-value">{escape_html(value)}</span>'
            "</span>"
        )
    return (
        f'<div class="{" ".join(classes)}">'
        f'<strong class="matheel-summary-title">{escaped_title}</strong>'
        f'<div class="matheel-summary-grid">{"".join(rendered_items)}</div>'
        "</div>"
    )


def status_panel_html(label, message, variant="status"):
    classes = ["matheel-status"]
    if variant and variant != "status":
        classes.append(f"matheel-{variant}")
    return (
        f'<div class="{" ".join(classes)}">'
        f"<strong>{escape_html(label)}:</strong> {escape_html(message)}"
        "</div>"
    )


def format_elapsed_seconds(value):
    return f"{float(value):.3f}s"


def score_interpretation(score, threshold=None):
    numeric_score = float(score)
    if numeric_score >= 0.85:
        label = "Very High"
        guidance = "Review this pair first."
    elif numeric_score >= 0.65:
        label = "High"
        guidance = "Likely worth manual review."
    elif numeric_score >= 0.35:
        label = "Moderate"
        guidance = "Use supporting evidence before deciding."
    else:
        label = "Low"
        guidance = "Usually lower priority."

    if threshold is not None:
        decision = "Meets threshold" if numeric_score >= float(threshold) else "Below threshold"
        guidance = f"{decision}. {guidance}"
    return label, guidance


def score_band_label(score, threshold=None):
    label, guidance = score_interpretation(score, threshold=threshold)
    return f"{label}: {guidance}"


def build_feature_weights(
    use_semantic,
    semantic_weight,
    use_levenshtein,
    levenshtein_weight,
    use_jaro_winkler,
    jaro_winkler_weight,
    use_winnowing,
    winnowing_weight,
    use_gst,
    gst_weight,
    code_metric,
    code_metric_weight,
):
    weights = {}
    fallback_names = []

    if not any(
        (
            use_semantic,
            use_levenshtein,
            use_jaro_winkler,
            use_winnowing,
            use_gst,
            (code_metric or "none") != "none",
        )
    ):
        raise gr.Error("Select at least one active metric.")

    if use_semantic:
        weights["semantic"] = max(0.0, float(semantic_weight))
        fallback_names.append("semantic")
    if use_levenshtein:
        weights["levenshtein"] = max(0.0, float(levenshtein_weight))
        fallback_names.append("levenshtein")
    if use_jaro_winkler:
        weights["jaro_winkler"] = max(0.0, float(jaro_winkler_weight))
        fallback_names.append("jaro_winkler")
    if use_winnowing:
        weights["winnowing"] = max(0.0, float(winnowing_weight))
        fallback_names.append("winnowing")
    if use_gst:
        weights["gst"] = max(0.0, float(gst_weight))
        fallback_names.append("gst")

    if (code_metric or "none") != "none":
        weights["code_metric"] = max(0.0, float(code_metric_weight))
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


def validate_positive_int_value(raw_value, label, minimum=1):
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{label} must be an integer.") from exc
    if value < int(minimum):
        raise gr.Error(f"{label} must be at least {int(minimum)}.")
    return value


def resolve_numeric_value(raw_value, default):
    if raw_value in (None, ""):
        return default
    return raw_value


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


def validate_tsed_costs_text(raw_text):
    text = str(raw_text or "").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise gr.Error("TSED costs must contain exactly 3 comma-separated values.")

    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise gr.Error("TSED costs must contain only numeric values.") from exc
        if not math.isfinite(value) or value < 0:
            raise gr.Error("TSED costs must be non-negative.")
        values.append(value)
    return tuple(values), format_weight_values(values)


def resolve_metric_kwargs(
    code_metric,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
):
    normalized = (code_metric or "none").strip().lower()
    if normalized == "ruby":
        mode = str(ruby_mode or DEFAULT_RUBY_MODE).strip().lower()
        if mode not in ("auto", "graph", "tree", "string", "ngram"):
            raise gr.Error("RUBY mode must be one of: auto, graph, tree, string, ngram.")
        timeout_seconds = float(ruby_graph_timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0.1:
            raise gr.Error("RUBY graph timeout must be at least 0.1 seconds.")
        return {
            "ruby_mode": mode,
            "ruby_graph_timeout_seconds": timeout_seconds,
        }
    if normalized == "tsed":
        (delete_cost, insert_cost, rename_cost), _ = validate_tsed_costs_text(tsed_costs)
        return {
            "tsed_delete_cost": delete_cost,
            "tsed_insert_cost": insert_cost,
            "tsed_rename_cost": rename_cost,
        }
    if normalized == "codebertscore":
        model_name, max_length, _ = resolve_codebertscore_model_length(
            codebertscore_model,
            codebertscore_max_length,
        )
        return {
            "codebertscore_model": model_name,
            "codebertscore_max_length": max_length,
        }
    return {}


def resolve_codebertscore_model_length(model_name, max_length):
    resolved_model = str(model_name or DEFAULT_CODEBERTSCORE_MODEL).strip()
    if not resolved_model:
        raise gr.Error("CodeBERTScore model must not be empty.")
    requested = int(float(max_length or DEFAULT_CODEBERTSCORE_MAX_LENGTH))
    if requested < 0:
        raise gr.Error("CodeBERTScore max length must be 0 or a positive integer.")
    detected = max(8, int(detect_model_max_token_length(model_name=resolved_model)))
    if requested == 0:
        selected = 0
    else:
        selected = min(requested, detected)
    return resolved_model, selected, detected


def sync_codebertscore_model_settings_gradio(model_name, codebertscore_max_length):
    current = max(0, int(float(codebertscore_max_length or DEFAULT_CODEBERTSCORE_MAX_LENGTH)))
    try:
        _, selected, detected = resolve_codebertscore_model_length(model_name, current)
    except Exception:
        return gr.update(
            minimum=0, maximum=max(current, DEFAULT_CODEBERTSCORE_MODEL_MAX_SEQUENCE), value=current
        )
    return gr.update(minimum=0, maximum=detected, value=selected)


def profile_status_html(message):
    return status_panel_html("Status", message)


def model_status_html(settings=None, error_message=None):
    if error_message:
        return status_panel_html("Model", error_message, variant="error")

    if not settings:
        return ""

    backend = settings.get("resolved_vector_backend", "auto")
    detected = int(settings.get("detected_max_token_length") or 0)
    configured = settings.get("configured_max_token_length")
    active = int(configured or detected or 0)
    device = settings.get("runtime_device", "auto")
    return summary_panel_html(
        "Model Settings",
        [
            ("Backend", backend),
            ("Detected Tokens", str(detected)),
            ("Active Tokens", str(active)),
            ("Device", device),
        ],
    )


def update_feature_sections(selected_features):
    selected = set(selected_features or [])
    return (
        gr.update(visible="Embedding" in selected),
        gr.update(visible="Levenshtein" in selected),
        gr.update(visible="Jaro-Winkler" in selected),
        gr.update(visible="Winnowing" in selected),
        gr.update(visible="GST" in selected),
        gr.update(visible="Code Metric" in selected),
    )


def update_code_preparation_sections(selected_steps):
    selected = set(selected_steps or [])
    return (
        gr.update(visible="Preprocessing" in selected),
        gr.update(visible="Chunking" in selected),
    )


def update_dataset_task_sections(task_label):
    is_retrieval = str(task_label or "").startswith("Retrieval")
    return gr.update(visible=not is_retrieval), gr.update(visible=is_retrieval)


def update_code_metric_sections(code_metric):
    normalized = (code_metric or "codebleu").strip().lower()
    return (
        gr.update(visible=normalized.startswith("codebleu")),
        gr.update(visible=normalized == "crystalbleu"),
        gr.update(visible=normalized == "ruby"),
        gr.update(visible=normalized == "tsed"),
        gr.update(visible=normalized == "codebertscore"),
    )


def metric_preset_names():
    return tuple(METRIC_PRESETS)


def metric_preset_options(preset_name):
    preset = METRIC_PRESETS.get(str(preset_name or "").strip()) or METRIC_PRESETS["Balanced"]
    weights = dict(preset["weights"])
    return {
        "features": list(preset["features"]),
        "semantic_weight": float(weights.get("semantic", 0.0)),
        "levenshtein_weight": float(weights.get("levenshtein", 0.0)),
        "jaro_winkler_weight": float(weights.get("jaro_winkler", 0.0)),
        "winnowing_weight": float(weights.get("winnowing", 0.0)),
        "gst_weight": float(weights.get("gst", 0.0)),
        "code_metric": str(preset.get("code_metric") or "codebleu"),
        "code_metric_weight": float(
            preset.get("code_metric_weight") or weights.get("code_metric", 0.0)
        ),
    }


def apply_metric_preset_gradio(preset_name):
    options = metric_preset_options(preset_name)
    features = options["features"]
    section_updates = update_feature_sections(features)
    return (
        gr.update(value=features),
        gr.update(value=options["semantic_weight"]),
        gr.update(value=options["levenshtein_weight"]),
        gr.update(value=options["jaro_winkler_weight"]),
        gr.update(value=options["winnowing_weight"]),
        gr.update(value=options["gst_weight"]),
        gr.update(value=options["code_metric"]),
        gr.update(value=options["code_metric_weight"]),
        *section_updates,
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
    "runtime_device",
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
    "ruby_mode",
    "ruby_graph_timeout_seconds",
    "tsed_costs",
    "codebertscore_model",
    "codebertscore_max_length",
    "semantic_weight",
    "levenshtein_weight",
    "levenshtein_weights",
    "jaro_winkler_weight",
    "jaro_winkler_prefix_weight",
    "winnowing_weight",
    "winnowing_kgram",
    "winnowing_window",
    "gst_weight",
    "gst_min_match_length",
    "lexical_tokenizer",
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
            "winnowing": max(0.0, float(row.get("winnowing_weight") or 0)),
            "gst": max(0.0, float(row.get("gst_weight") or 0)),
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
        effective_winnowing_kgram = 5
        effective_winnowing_window = 4
        if feature_weights.get("winnowing", 0.0) > 0:
            effective_winnowing_kgram = validate_positive_int_value(
                row.get("winnowing_kgram") or 5,
                "Winnowing k-gram size",
            )
            effective_winnowing_window = validate_positive_int_value(
                row.get("winnowing_window") or 4,
                "Winnowing window size",
            )
        effective_gst_min_match_length = 5
        if feature_weights.get("gst", 0.0) > 0:
            effective_gst_min_match_length = validate_positive_int_value(
                row.get("gst_min_match_length") or 5,
                "GST minimum match length",
            )

        effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
        if code_metric.startswith("codebleu") and feature_weights.get("code_metric", 0.0) > 0:
            effective_codebleu_component_weights = validate_codebleu_component_weights_text(
                row.get("codebleu_component_weights") or "0.25,0.25,0.25,0.25"
            )
        ruby_mode = str(row.get("ruby_mode") or DEFAULT_RUBY_MODE).strip() or DEFAULT_RUBY_MODE
        ruby_graph_timeout_seconds = float(
            resolve_numeric_value(row.get("ruby_graph_timeout_seconds"), DEFAULT_RUBY_GRAPH_TIMEOUT)
        )
        tsed_costs = str(row.get("tsed_costs") or DEFAULT_TSED_COSTS).strip() or DEFAULT_TSED_COSTS
        codebertscore_model = (
            str(row.get("codebertscore_model") or DEFAULT_CODEBERTSCORE_MODEL).strip()
            or DEFAULT_CODEBERTSCORE_MODEL
        )
        codebertscore_max_length = int(
            float(row.get("codebertscore_max_length") or DEFAULT_CODEBERTSCORE_MAX_LENGTH)
        )
        metric_kwargs = resolve_metric_kwargs(
            code_metric,
            ruby_mode,
            ruby_graph_timeout_seconds,
            tsed_costs,
            codebertscore_model,
            codebertscore_max_length,
        )

        options = {
            "model_name": str(row.get("model_name") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            "vector_backend": str(row.get("vector_backend") or "auto").strip() or "auto",
            "similarity_function": str(row.get("similarity_function") or "cosine").strip()
            or "cosine",
            "pooling_method": str(row.get("pooling_method") or "mean").strip() or "mean",
            "max_token_length": max(8, int(float(row.get("max_token_length") or 256))),
            "device": str(row.get("runtime_device") or "auto").strip() or "auto",
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
                0,
                int(float(resolve_numeric_value(row.get("crystalbleu_trivial_ngram_count"), 50))),
            ),
            "ruby_mode": ruby_mode,
            "ruby_graph_timeout_seconds": ruby_graph_timeout_seconds,
            "codebertscore_model": codebertscore_model,
            "codebertscore_max_length": codebertscore_max_length,
            "levenshtein_weights": effective_levenshtein_weights,
            "jaro_winkler_prefix_weight": max(
                0.0,
                min(0.25, float(resolve_numeric_value(row.get("jaro_winkler_prefix_weight"), 0.1))),
            ),
            "winnowing_kgram": effective_winnowing_kgram,
            "winnowing_window": effective_winnowing_window,
            "gst_min_match_length": effective_gst_min_match_length,
            "lexical_tokenizer": str(row.get("lexical_tokenizer") or "raw").strip() or "raw",
            "threshold": max(0.0, float(resolve_numeric_value(row.get("threshold"), 0.35))),
            "number_results": max(1, int(float(row.get("number_results") or 50))),
            "feature_weights": feature_weights,
        }
        options.update(metric_kwargs)
        configs.append({"run_name": run_name, "options": options})
    return configs


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


def _selected_suite_algorithm_names(selected_features, code_metric):
    selected = set(selected_features or [])
    names = []
    for label in FEATURE_UI_CHOICES:
        if label not in selected:
            continue
        if label == "Code Metric":
            names.append(str(code_metric or "code_metric").strip().lower() or "code_metric")
        else:
            names.append(SUITE_FEATURE_NAME_MAP[label])
    return names


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
    selected_steps = set(selected_preparation or [])
    parts = _selected_suite_algorithm_names(selected_features, code_metric)
    if "Preprocessing" in selected_steps and str(preprocess_mode or "none").strip() not in (
        "",
        "none",
    ):
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
    runtime_device,
    semantic_weight,
    levenshtein_weight,
    jaro_winkler_weight,
    winnowing_weight,
    gst_weight,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    lexical_tokenizer,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
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
    normalized_winnowing_kgram = 5
    normalized_winnowing_window = 4
    if "Winnowing" in selected:
        normalized_winnowing_kgram = validate_positive_int_value(
            winnowing_kgram,
            "Winnowing k-gram size",
        )
        normalized_winnowing_window = validate_positive_int_value(
            winnowing_window,
            "Winnowing window size",
        )
    normalized_gst_min_match_length = 5
    if "GST" in selected:
        normalized_gst_min_match_length = validate_positive_int_value(
            gst_min_match_length,
            "GST minimum match length",
        )

    normalized_codebleu_weights = "0.25,0.25,0.25,0.25"
    if normalized_code_metric.startswith("codebleu"):
        normalized_codebleu_weights = validate_codebleu_component_weights_text(
            codebleu_component_weights
        )
    normalized_ruby_mode = str(ruby_mode or DEFAULT_RUBY_MODE).strip().lower() or DEFAULT_RUBY_MODE
    normalized_ruby_timeout = float(
        resolve_numeric_value(ruby_graph_timeout_seconds, DEFAULT_RUBY_GRAPH_TIMEOUT)
    )
    normalized_tsed_costs_raw = str(tsed_costs or DEFAULT_TSED_COSTS).strip() or DEFAULT_TSED_COSTS
    if normalized_code_metric == "tsed":
        _, normalized_tsed_costs = validate_tsed_costs_text(normalized_tsed_costs_raw)
    else:
        normalized_tsed_costs = normalized_tsed_costs_raw
    normalized_codebertscore_model = (
        str(codebertscore_model or DEFAULT_CODEBERTSCORE_MODEL).strip()
        or DEFAULT_CODEBERTSCORE_MODEL
    )
    normalized_codebertscore_max_length = int(
        float(codebertscore_max_length or DEFAULT_CODEBERTSCORE_MAX_LENGTH)
    )

    resolve_metric_kwargs(
        normalized_code_metric,
        normalized_ruby_mode,
        normalized_ruby_timeout,
        normalized_tsed_costs,
        normalized_codebertscore_model,
        normalized_codebertscore_max_length,
    )

    feature_weights = build_feature_weights(
        "Embedding" in selected,
        semantic_weight,
        "Levenshtein" in selected,
        levenshtein_weight,
        "Jaro-Winkler" in selected,
        jaro_winkler_weight,
        "Winnowing" in selected,
        winnowing_weight,
        "GST" in selected,
        gst_weight,
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
        "runtime_device": str(runtime_device or "auto").strip() or "auto",
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
        "crystalbleu_trivial_ngram_count": max(
            0,
            int(float(resolve_numeric_value(crystalbleu_trivial_ngram_count, 50))),
        ),
        "ruby_mode": normalized_ruby_mode,
        "ruby_graph_timeout_seconds": normalized_ruby_timeout,
        "tsed_costs": normalized_tsed_costs,
        "codebertscore_model": normalized_codebertscore_model,
        "codebertscore_max_length": max(0, normalized_codebertscore_max_length),
        "semantic_weight": feature_weights.get("semantic", 0.0),
        "levenshtein_weight": feature_weights.get("levenshtein", 0.0),
        "levenshtein_weights": normalized_levenshtein_weights,
        "jaro_winkler_weight": feature_weights.get("jaro_winkler", 0.0),
        "jaro_winkler_prefix_weight": max(
            0.0,
            min(0.25, float(resolve_numeric_value(jaro_winkler_prefix_weight, 0.1))),
        ),
        "winnowing_weight": feature_weights.get("winnowing", 0.0),
        "winnowing_kgram": normalized_winnowing_kgram,
        "winnowing_window": normalized_winnowing_window,
        "gst_weight": feature_weights.get("gst", 0.0),
        "gst_min_match_length": normalized_gst_min_match_length,
        "lexical_tokenizer": str(lexical_tokenizer or "raw").strip() or "raw",
        "threshold": max(0.0, float(resolve_numeric_value(threshold, 0.35))),
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
    runtime_device,
    semantic_weight,
    levenshtein_weight,
    jaro_winkler_weight,
    winnowing_weight,
    gst_weight,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    lexical_tokenizer,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
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
        runtime_device,
        semantic_weight,
        levenshtein_weight,
        jaro_winkler_weight,
        winnowing_weight,
        gst_weight,
        levenshtein_weights,
        jaro_winkler_prefix_weight,
        winnowing_kgram,
        winnowing_window,
        gst_min_match_length,
        lexical_tokenizer,
        code_metric,
        code_metric_weight,
        code_language,
        codebleu_component_weights,
        crystalbleu_max_order,
        crystalbleu_trivial_ngram_count,
        ruby_mode,
        ruby_graph_timeout_seconds,
        tsed_costs,
        codebertscore_model,
        codebertscore_max_length,
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
        return summary_panel_html(
            "Configured Runs",
            [("Runs", "0"), ("Run Names", "None")],
            variant="empty",
        )

    names = [str(value).strip() for value in frame["run_name"].tolist() if str(value).strip()]
    preview = ", ".join(names[:4])
    if len(names) > 4:
        preview = f"{preview}, +{len(names) - 4} more"
    return summary_panel_html(
        "Configured Runs",
        [("Runs", str(len(names))), ("Run Names", preview)],
    )


def reset_suite_runs():
    rows = empty_suite_rows()
    return (
        rows,
        suite_runs_overview_html(rows),
        profile_status_html("Run list cleared."),
        preview_suite_run_name(
            rows,
            DEFAULT_FEATURE_SELECTION,
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
    return summary_panel_html(
        "Comparison Suite",
        [("Status", "No runs executed"), ("Best Run", "None")],
        variant="empty",
    )


def suite_summary_html(summary):
    if summary is None or summary.empty:
        return empty_suite_summary_html()

    best = summary.iloc[0]
    elapsed_seconds = float(best.get("elapsed_seconds", 0.0))
    feature_set = best.get("feature_set", "none")
    return summary_panel_html(
        "Comparison Suite",
        [
            ("Best Run", best["run_name"]),
            ("Runs", str(len(summary))),
            ("Best Mean", f"{float(best['mean_score']):.3f}"),
            ("Best Max", f"{float(best['max_score']):.3f}"),
            ("Elapsed", format_elapsed_seconds(elapsed_seconds)),
            ("Features", feature_set),
        ],
    )


def load_suite_details(run_name, details_store):
    rows = list((details_store or {}).get(run_name) or [])
    if not rows:
        return pd.DataFrame(columns=["file_name_1", "file_name_2", "similarity_score"])
    return pd.DataFrame(rows)


def unique_suite_detail_filenames(run_names):
    used_names = set()
    filenames = []
    for run_name in run_names:
        detail_stem = slugify_run_name(run_name) or "run"
        candidate = detail_stem
        suffix = 2
        while candidate.casefold() in used_names:
            candidate = f"{detail_stem}_{suffix}"
            suffix += 1
        used_names.add(candidate.casefold())
        filenames.append(f"{candidate}.csv")
    return filenames


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
    runtime_device,
    semantic_weight,
    levenshtein_weight,
    jaro_winkler_weight,
    winnowing_weight,
    gst_weight,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    lexical_tokenizer,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
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
    progress=gr.Progress(track_tqdm=True),
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
            runtime_device,
            semantic_weight,
            levenshtein_weight,
            jaro_winkler_weight,
            winnowing_weight,
            gst_weight,
            levenshtein_weights,
            jaro_winkler_prefix_weight,
            winnowing_kgram,
            winnowing_window,
            gst_min_match_length,
            lexical_tokenizer,
            code_metric,
            code_metric_weight,
            code_language,
            codebleu_component_weights,
            crystalbleu_max_order,
            crystalbleu_trivial_ngram_count,
            ruby_mode,
            ruby_graph_timeout_seconds,
            tsed_costs,
            codebertscore_model,
            codebertscore_max_length,
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
            f'Ran the current settings as "{row["run_name"]}". Saved runs unchanged.'
        )
    else:
        status_html = profile_status_html(f"Completed {len(saved_frame)} saved run(s).")

    run_configs = suite_rows_to_configs(execution_frame)

    export_root = os.fspath(make_temp_workspace("matheel-suite-"))
    summary_name = f"comparison_suite_summary.{output_format}"
    summary_export_path = os.path.join(export_root, summary_name)
    summary, result_frames = run_comparison_suite(
        zipped_file,
        run_configs,
        summary_out=summary_export_path,
        details_dir=None,
        output_format=output_format,
        progress_callback=gradio_progress_callback(progress),
    )
    details_store = {
        run_name: frame.to_dict(orient="records") for run_name, frame in result_frames.items()
    }
    first_run = summary.iloc[0]["run_name"] if not summary.empty else None
    first_details = load_suite_details(first_run, details_store)
    details_zip_path = os.path.join(export_root, "comparison_suite_details.zip")
    with zipfile.ZipFile(details_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        detail_filenames = unique_suite_detail_filenames(result_frames)
        for (_, frame), detail_filename in zip(result_frames.items(), detail_filenames):
            detail_path = os.path.join(export_root, detail_filename)
            frame.to_csv(detail_path, index=False)
            archive.write(detail_path, arcname=detail_filename)
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


def _uploaded_file_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (list, tuple)):
        if not uploaded_file:
            return None
        uploaded_file = uploaded_file[0]
    if isinstance(uploaded_file, dict):
        uploaded_file = uploaded_file.get("path") or uploaded_file.get("name")
    if isinstance(uploaded_file, (str, os.PathLike)):
        return Path(uploaded_file)
    if hasattr(uploaded_file, "name"):
        uploaded_file = uploaded_file.name
    return Path(uploaded_file)


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _safe_extract_uploaded_zip(archive_path, output_dir):
    archive = Path(archive_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as archive_file:
        for member in archive_file.infolist():
            target = destination / member.filename
            if not _is_relative_to(target, destination):
                raise gr.Error(f"Dataset archive contains an unsafe path: {member.filename}")
        archive_file.extractall(destination)
    return destination


def _zip_artifact_paths(paths, zip_path):
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_paths = sorted({Path(path) for path in paths if path}, key=lambda path: path.name)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in resolved_paths:
            archive.write(path, arcname=path.name)
    return str(destination)


def _dataset_kind_from_task_label(task_label):
    return "retrieval" if str(task_label or "").startswith("Retrieval") else "pair"


def empty_dataset_map_summary_html():
    return summary_panel_html(
        "Dataset Map",
        [("Status", "No map generated"), ("Artifacts", "None")],
        variant="empty",
    )


def empty_pair_explanation_summary_html():
    return summary_panel_html(
        "Pair Explanation",
        [("Status", "No explanation generated"), ("Artifacts", "None")],
        variant="empty",
    )


def empty_leaderboard_inspection_summary_html():
    return summary_panel_html(
        "Leaderboard Inspection",
        [("Status", "No artifact loaded"), ("Artifacts", "None")],
        variant="empty",
    )


def empty_ready_leaderboard_summary_html():
    return summary_panel_html(
        "Ready Leaderboard",
        [("Status", "No leaderboard run"), ("Artifacts", "None")],
        variant="empty",
    )


def empty_dataset_validation_summary_html():
    return summary_panel_html(
        "Dataset Validation",
        [("Status", "No dataset checked"), ("Artifacts", "None")],
        variant="empty",
    )


def empty_threshold_tuning_summary_html():
    return summary_panel_html(
        "Threshold Tuning",
        [("Status", "No pair scores available"), ("Artifacts", "None")],
        variant="empty",
    )


def ready_leaderboard_registered_datasets_frame():
    rows = []
    for preset_name in available_dataset_presets():
        preset = get_dataset_preset(preset_name)
        tasks = tuple(preset.get("task_families") or ())
        metric_default = []
        sampling_default = []
        if "pair" in tasks:
            metric_default.append("f1, accuracy, auroc, average_precision")
            sampling_default.append("full labeled pairs, threshold=0.5")
        if "retrieval" in tasks:
            metric_default.append(
                "mean_average_precision, mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k"
            )
            sampling_default.append("full query ranking, k=10")
        rows.append(
            {
                "Preset": preset_name,
                "Tasks": ", ".join(tasks),
                "Source": preset.get("source", ""),
                "Identifier": preset.get("identifier", ""),
                "Evaluation Metrics": "; ".join(metric_default),
                "Sampling Default": "; ".join(sampling_default),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Preset",
            "Tasks",
            "Source",
            "Identifier",
            "Evaluation Metrics",
            "Sampling Default",
        ],
    )


def dataset_map_display_frame(projection):
    frame = projection.copy() if isinstance(projection, pd.DataFrame) else pd.DataFrame(projection)
    for column in ("x", "y"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float).round(4)
    preferred = [
        column
        for column in ("document_id", "role", "pair_count", "file_path", "x", "y")
        if column in frame.columns
    ]
    remaining = [column for column in frame.columns if column not in preferred]
    return frame[preferred + remaining]


def dataset_map_summary_html(projection, color_column):
    attrs = getattr(projection, "attrs", {})
    actual_method = attrs.get("projection_method", "unknown")
    requested_method = attrs.get("requested_projection_method", "unknown")
    dataset_kind = str(attrs.get("dataset_kind") or "dataset").replace("_", " ")
    return summary_panel_html(
        "Dataset Map",
        [
            ("Dataset", attrs.get("dataset_name", "dataset")),
            ("Kind", dataset_kind),
            ("Documents", str(len(projection))),
            ("Projection", f"{actual_method} (requested {requested_method})"),
            ("Seed", str(attrs.get("seed", ""))),
            ("Vector Dim", str(attrs.get("static_vector_dim", ""))),
            ("Color", color_column or "documents"),
        ],
    )


def generate_dataset_map_gradio(
    dataset_file,
    task_label,
    projection_method,
    seed,
    static_vector_dim,
    color_column,
    progress=gr.Progress(track_tqdm=True),
):
    if dataset_file is None:
        return empty_dataset_map_summary_html(), pd.DataFrame(), "", None

    dataset_root = _dataset_root_from_upload(dataset_file, task_label)
    resolved_dim = validate_positive_int_value(
        static_vector_dim, "Static vector dimension", minimum=8
    )
    resolved_seed = int(float(resolve_numeric_value(seed, 7)))
    if progress is not None:
        progress(0.2, desc="Loading dataset texts")
    try:
        projection = build_dataset_embedding_map(
            dataset_root,
            kind=_dataset_kind_from_task_label(task_label),
            method=projection_method,
            seed=resolved_seed,
            static_vector_dim=resolved_dim,
        )
    except (ImportError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    requested_color = str(color_column or "").strip()
    resolved_color = requested_color or ("role" if "role" in projection.columns else None)
    if resolved_color and resolved_color not in projection.columns:
        raise gr.Error(f"Color column does not exist in the map: {resolved_color}")

    if progress is not None:
        progress(0.7, desc="Writing visualization artifacts")
    export_root = make_temp_workspace("matheel-dataset-map-")
    artifacts = write_dataset_map_artifacts(
        projection,
        export_root,
        title=str(projection.attrs.get("dataset_name") or "Matheel Dataset Map"),
        color_column=resolved_color,
    )
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(), export_root / "dataset_map_artifacts.zip"
    )
    if progress is not None:
        progress(1.0, desc="Dataset map complete")
    return (
        dataset_map_summary_html(projection, resolved_color),
        dataset_map_display_frame(projection),
        artifacts["html"].read_text(encoding="utf-8"),
        artifacts_zip,
    )


def pair_explanation_matches_frame(explanation):
    matches = list(explanation.get("matches") or [])
    frame = pd.DataFrame(matches)
    if frame.empty:
        return pd.DataFrame(columns=["Match", "Level", "Score", "Left Segment", "Right Segment"])
    display = frame.copy()
    display["left_index"] = display["left_index"].astype(int) + 1
    display["right_index"] = display["right_index"].astype(int) + 1
    display["score"] = display["score"].astype(float).round(4)
    return display.rename(
        columns={
            "match_id": "Match",
            "level": "Level",
            "score": "Score",
            "left_index": "Left Segment",
            "right_index": "Right Segment",
        }
    )[["Match", "Level", "Score", "Left Segment", "Right Segment"]]


def pair_explanation_summary_html(explanation):
    metadata = explanation.get("metadata", {})
    matches = list(explanation.get("matches") or [])
    level_counts = pd.Series([match.get("level", "none") for match in matches]).value_counts()
    return summary_panel_html(
        "Pair Explanation",
        [
            ("Left", metadata.get("left_id", "left")),
            ("Right", metadata.get("right_id", "right")),
            ("Mode", metadata.get("segment_mode", "line")),
            ("Matches", str(len(matches))),
            ("High", str(int(level_counts.get("high", 0)))),
            ("Medium", str(int(level_counts.get("medium", 0)))),
            ("Low", str(int(level_counts.get("low", 0)))),
        ],
    )


def generate_pair_explanation_gradio(
    left_code,
    right_code,
    segment_mode,
    high_threshold,
    medium_threshold,
    low_threshold,
    chunk_size,
):
    if not str(left_code or "").strip() or not str(right_code or "").strip():
        raise gr.Error("Paste both snippets before generating a pair explanation.")
    try:
        explanation = build_pair_explanation(
            left_code,
            right_code,
            left_id="Code A",
            right_id="Code B",
            segment_mode=segment_mode,
            high_threshold=float(high_threshold),
            medium_threshold=float(medium_threshold),
            low_threshold=float(low_threshold),
            chunk_size=int(float(chunk_size or 1)),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    export_root = make_temp_workspace("matheel-pair-explanation-")
    artifacts = write_pair_explanation_artifacts(
        explanation,
        export_root,
        basename="pair_explanation",
        title="Code A vs Code B",
    )
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "pair_explanation_artifacts.zip",
    )
    return (
        pair_explanation_summary_html(explanation),
        pair_explanation_matches_frame(explanation),
        pair_explanation_html(explanation, title="Code A vs Code B"),
        artifacts_zip,
    )


def _looks_like_leaderboard_payload(payload):
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("aggregate"), list)
        and isinstance(payload.get("per_dataset"), list)
        and isinstance(payload.get("metadata"), dict)
    )


def _load_leaderboard_payload_from_upload(uploaded_file):
    uploaded_path = _uploaded_file_path(uploaded_file)
    if uploaded_path is None:
        raise gr.Error("Upload a leaderboard JSON file or artifact ZIP first.")
    if uploaded_path.is_dir():
        candidates = sorted(path for path in uploaded_path.rglob("*.json") if path.is_file())
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if _looks_like_leaderboard_payload(payload):
                return payload, candidate.name
        raise gr.Error("Could not find a leaderboard JSON artifact in the uploaded directory.")
    if zipfile.is_zipfile(uploaded_path):
        with zipfile.ZipFile(uploaded_path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/") or not name.lower().endswith(".json"):
                    continue
                try:
                    payload = json.loads(archive.read(name).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if _looks_like_leaderboard_payload(payload):
                    return payload, Path(name).name
        raise gr.Error("Could not find a leaderboard JSON artifact in the uploaded ZIP.")
    if uploaded_path.suffix.lower() != ".json":
        raise gr.Error("Leaderboard inspection accepts JSON artifacts or ZIP archives.")
    try:
        payload = json.loads(uploaded_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise gr.Error("Leaderboard artifact must be valid JSON.") from exc
    if not _looks_like_leaderboard_payload(payload):
        raise gr.Error("Uploaded JSON is not a Matheel leaderboard artifact.")
    return payload, uploaded_path.name


def _leaderboard_report_from_payload(payload):
    metadata = dict(payload.get("metadata") or {})
    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
    cards = payload.get("cards") if isinstance(payload.get("cards"), dict) else {}
    return {
        "metadata": metadata,
        "manifest": manifest,
        "cards": {
            "datasets": list(cards.get("datasets") or []),
            "algorithms": list(cards.get("algorithms") or []),
        },
        "per_dataset": pd.DataFrame(payload.get("per_dataset") or []),
        "aggregate": pd.DataFrame(payload.get("aggregate") or []),
    }


def load_ready_made_leaderboard_payload(artifact_path=READY_MADE_LEADERBOARD_PATH):
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ready-made leaderboard artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Ready-made leaderboard artifact must be valid JSON.") from exc
    if not _looks_like_leaderboard_payload(payload):
        raise ValueError("Ready-made leaderboard artifact is not a Matheel leaderboard payload.")
    return payload


def ready_made_leaderboard_summary_html(report):
    profile = dict(report["metadata"].get("benchmark_profile") or {})
    dataset_presets = list(profile.get("dataset_presets") or [])
    algorithms = list(profile.get("algorithm_presets") or [])
    generated_at = str(profile.get("generated_at") or "unknown")
    if "T" in generated_at:
        generated_at = generated_at.split("T", 1)[0]
    sample_plan = (
        f"≤{profile.get('pair_sample_limit', '?')} pairs; "
        f"≤{profile.get('retrieval_query_limit', '?')} queries × "
        f"{profile.get('retrieval_corpus_limit', '?')} documents"
    )
    return summary_panel_html(
        "Ready-made Leaderboard",
        [
            ("Dataset Presets", str(len(dataset_presets))),
            (
                "Task Views",
                str(profile.get("dataset_task_count") or len(report["cards"]["datasets"])),
            ),
            ("Algorithms", str(len(algorithms) or len(report["cards"]["algorithms"]))),
            ("Backend", str(profile.get("vector_backend") or "unknown")),
            ("Sampling", sample_plan),
            ("Seed", str(profile.get("seed", ""))),
            ("Generated", generated_at),
        ],
    )


def ready_made_leaderboard_coverage_frame(report):
    rows = []
    for card in report.get("cards", {}).get("datasets", []):
        metadata = dict(card.get("metadata") or {})
        source = dict(card.get("source") or {})
        counts = dict(card.get("counts") or {})
        if card.get("task_family") == "retrieval":
            sample = f"{counts.get('queries', 0)} queries × {counts.get('documents', 0)} documents"
        else:
            sample = f"{counts.get('pairs', 0)} pairs"
        rows.append(
            {
                "Dataset": card.get("name", ""),
                "Preset": source.get("preset") or metadata.get("source_preset", ""),
                "Task": card.get("task_family", ""),
                "Sample": sample,
                "Source": source.get("source", ""),
                "License": card.get("license", "unknown"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Dataset", "Preset", "Task", "Sample", "Source", "License"],
    )


def _ready_made_task_family(task_family):
    task = str(task_family or "").strip().lower()
    if task not in READY_LEADERBOARD_DEFAULT_METRICS:
        return READY_LEADERBOARD_TASK_CHOICES[0]
    return task


def _ready_made_unique_values(frame, column):
    if not isinstance(frame, pd.DataFrame) or column not in frame:
        return []
    return sorted({str(value) for value in frame[column].dropna().tolist() if str(value).strip()})


def _ready_made_filter_options_from_report(report, task_family):
    task = _ready_made_task_family(task_family)
    per_dataset = report["per_dataset"]
    task_rows = per_dataset[per_dataset["task_family"] == task]
    configured_metrics = (
        READY_LEADERBOARD_PAIR_METRICS if task == "pair" else READY_LEADERBOARD_RETRIEVAL_METRICS
    )
    available_metrics = set(_ready_made_unique_values(task_rows, "metric"))
    metrics = [metric for metric in configured_metrics if metric in available_metrics]
    if not metrics:
        metrics = list(configured_metrics)
    default_metric = READY_LEADERBOARD_DEFAULT_METRICS[task]
    if default_metric not in metrics:
        default_metric = metrics[0]
    return {
        "task": task,
        "metric": default_metric,
        "metrics": metrics,
        "datasets": [
            READY_LEADERBOARD_ALL_DATASETS,
            *_ready_made_unique_values(task_rows, "dataset_name"),
        ],
        "algorithms": [
            READY_LEADERBOARD_ALL_ALGORITHMS,
            *_ready_made_unique_values(task_rows, "algorithm_name"),
        ],
        "sources": [
            READY_LEADERBOARD_ALL_SOURCES,
            *_ready_made_unique_values(task_rows, "dataset_source"),
        ],
    }


def ready_made_leaderboard_filter_options(
    task_family=READY_LEADERBOARD_TASK_CHOICES[0],
    artifact_path=READY_MADE_LEADERBOARD_PATH,
):
    task = _ready_made_task_family(task_family)
    try:
        report = _leaderboard_report_from_payload(
            load_ready_made_leaderboard_payload(artifact_path)
        )
    except (FileNotFoundError, OSError, ValueError):
        metrics = (
            READY_LEADERBOARD_PAIR_METRICS
            if task == "pair"
            else READY_LEADERBOARD_RETRIEVAL_METRICS
        )
        return {
            "task": task,
            "metric": READY_LEADERBOARD_DEFAULT_METRICS[task],
            "metrics": list(metrics),
            "datasets": [READY_LEADERBOARD_ALL_DATASETS],
            "algorithms": [
                READY_LEADERBOARD_ALL_ALGORITHMS,
                *READY_LEADERBOARD_ALGORITHM_CHOICES,
            ],
            "sources": [READY_LEADERBOARD_ALL_SOURCES],
        }
    return _ready_made_filter_options_from_report(report, task)


def _filter_ready_made_leaderboard_report(
    report,
    task_family,
    metric,
    dataset_name,
    algorithm_name,
    dataset_source,
    aggregate_sort="mean_score",
    aggregate_direction=READY_LEADERBOARD_SORT_DIRECTIONS[0],
    per_dataset_sort="score",
    per_dataset_direction=READY_LEADERBOARD_SORT_DIRECTIONS[0],
):
    options = _ready_made_filter_options_from_report(report, task_family)
    task = options["task"]
    selected_metric = str(metric or "")
    if selected_metric not in options["metrics"]:
        selected_metric = options["metric"]

    per_dataset = report["per_dataset"].copy()
    per_dataset = per_dataset[
        (per_dataset["task_family"] == task) & (per_dataset["metric"] == selected_metric)
    ]
    if dataset_name and dataset_name != READY_LEADERBOARD_ALL_DATASETS:
        per_dataset = per_dataset[per_dataset["dataset_name"] == dataset_name]
    if dataset_source and dataset_source != READY_LEADERBOARD_ALL_SOURCES:
        per_dataset = per_dataset[per_dataset["dataset_source"] == dataset_source]

    if per_dataset.empty:
        aggregate = report["aggregate"].iloc[0:0].copy()
    else:
        aggregate = per_dataset.groupby(
            ["task_family", "algorithm_name", "metric"],
            as_index=False,
        ).agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            dataset_count=("dataset_name", "nunique"),
            sample_count=("sample_count", "sum"),
        )
        aggregate["rank"] = (
            aggregate.groupby(["task_family", "metric"])["mean_score"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype(int)
        )

    if algorithm_name and algorithm_name != READY_LEADERBOARD_ALL_ALGORITHMS:
        per_dataset = per_dataset[per_dataset["algorithm_name"] == algorithm_name]
        aggregate = aggregate[aggregate["algorithm_name"] == algorithm_name]

    aggregate = _sort_ready_made_leaderboard_frame(
        aggregate,
        aggregate_sort,
        aggregate_direction,
        READY_LEADERBOARD_AGGREGATE_SORT_COLUMNS,
        tie_breakers=("algorithm_name",),
    )
    per_dataset = _sort_ready_made_leaderboard_frame(
        per_dataset,
        per_dataset_sort,
        per_dataset_direction,
        READY_LEADERBOARD_PER_DATASET_SORT_COLUMNS,
        tie_breakers=("dataset_name", "algorithm_name"),
    )
    return leaderboard_display_frame(aggregate), leaderboard_display_frame(per_dataset)


def _sort_ready_made_leaderboard_frame(
    frame,
    sort_column,
    direction,
    allowed_columns,
    *,
    tie_breakers,
):
    selected_column = str(sort_column or "")
    if selected_column not in allowed_columns:
        selected_column = allowed_columns[0]
    sort_columns = [selected_column]
    sort_columns.extend(
        column for column in tie_breakers if column != selected_column and column in frame.columns
    )
    primary_ascending = str(direction) == "Ascending"
    return frame.sort_values(
        sort_columns,
        ascending=[primary_ascending, *([True] * (len(sort_columns) - 1))],
        na_position="last",
        ignore_index=True,
    )


def filter_ready_made_leaderboard(
    task_family,
    metric,
    dataset_name=READY_LEADERBOARD_ALL_DATASETS,
    algorithm_name=READY_LEADERBOARD_ALL_ALGORITHMS,
    dataset_source=READY_LEADERBOARD_ALL_SOURCES,
    aggregate_sort="mean_score",
    aggregate_direction=READY_LEADERBOARD_SORT_DIRECTIONS[0],
    per_dataset_sort="score",
    per_dataset_direction=READY_LEADERBOARD_SORT_DIRECTIONS[0],
    artifact_path=READY_MADE_LEADERBOARD_PATH,
):
    try:
        report = _leaderboard_report_from_payload(
            load_ready_made_leaderboard_payload(artifact_path)
        )
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(), pd.DataFrame()
    return _filter_ready_made_leaderboard_report(
        report,
        task_family,
        metric,
        dataset_name,
        algorithm_name,
        dataset_source,
        aggregate_sort,
        aggregate_direction,
        per_dataset_sort,
        per_dataset_direction,
    )


def change_ready_made_leaderboard_task(
    task_family,
    aggregate_sort,
    aggregate_direction,
    per_dataset_sort,
    per_dataset_direction,
):
    options = ready_made_leaderboard_filter_options(task_family)
    aggregate, per_dataset = filter_ready_made_leaderboard(
        options["task"],
        options["metric"],
        aggregate_sort=aggregate_sort,
        aggregate_direction=aggregate_direction,
        per_dataset_sort=per_dataset_sort,
        per_dataset_direction=per_dataset_direction,
    )
    return (
        gr.update(choices=options["metrics"], value=options["metric"]),
        gr.update(choices=options["datasets"], value=READY_LEADERBOARD_ALL_DATASETS),
        gr.update(choices=options["algorithms"], value=READY_LEADERBOARD_ALL_ALGORITHMS),
        gr.update(choices=options["sources"], value=READY_LEADERBOARD_ALL_SOURCES),
        aggregate,
        per_dataset,
    )


def reset_ready_made_leaderboard_filters():
    task = READY_LEADERBOARD_TASK_CHOICES[0]
    options = ready_made_leaderboard_filter_options(task)
    aggregate, per_dataset = filter_ready_made_leaderboard(task, options["metric"])
    return (
        gr.update(value=task),
        gr.update(choices=options["metrics"], value=options["metric"]),
        gr.update(choices=options["datasets"], value=READY_LEADERBOARD_ALL_DATASETS),
        gr.update(choices=options["algorithms"], value=READY_LEADERBOARD_ALL_ALGORITHMS),
        gr.update(choices=options["sources"], value=READY_LEADERBOARD_ALL_SOURCES),
        gr.update(value="mean_score"),
        gr.update(value=READY_LEADERBOARD_SORT_DIRECTIONS[0]),
        gr.update(value="score"),
        gr.update(value=READY_LEADERBOARD_SORT_DIRECTIONS[0]),
        aggregate,
        per_dataset,
    )


def ready_made_leaderboard_values(artifact_path=READY_MADE_LEADERBOARD_PATH):
    try:
        payload = load_ready_made_leaderboard_payload(artifact_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return (
            summary_panel_html(
                "Ready-made Leaderboard",
                [("Status", "Snapshot unavailable"), ("Reason", str(exc))],
                variant="empty",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
        )
    report = _leaderboard_report_from_payload(payload)
    title = str(report["metadata"].get("name") or "Matheel Ready-made Leaderboard")
    task = READY_LEADERBOARD_TASK_CHOICES[0]
    aggregate, per_dataset = _filter_ready_made_leaderboard_report(
        report,
        task,
        READY_LEADERBOARD_DEFAULT_METRICS[task],
        READY_LEADERBOARD_ALL_DATASETS,
        READY_LEADERBOARD_ALL_ALGORITHMS,
        READY_LEADERBOARD_ALL_SOURCES,
    )
    return (
        ready_made_leaderboard_summary_html(report),
        aggregate,
        per_dataset,
        ready_made_leaderboard_coverage_frame(report),
        benchmark_report_html(report, title=title),
        os.fspath(Path(artifact_path)),
    )


def leaderboard_display_frame(frame):
    display = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    for column in ("score", "mean_score", "median_score"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
    return display


def leaderboard_inspection_summary_html(report, source_name):
    aggregate = report["aggregate"]
    per_dataset = report["per_dataset"]
    cards = report.get("cards") or {}
    dataset_count = len(cards.get("datasets") or []) or (
        per_dataset["dataset_name"].nunique() if "dataset_name" in per_dataset else 0
    )
    algorithm_count = len(cards.get("algorithms") or []) or (
        aggregate["algorithm_name"].nunique() if "algorithm_name" in aggregate else 0
    )
    metric_count = aggregate["metric"].nunique() if "metric" in aggregate else 0
    return summary_panel_html(
        "Leaderboard Inspection",
        [
            ("Report", report["metadata"].get("name", "leaderboard")),
            ("Source", source_name),
            ("Datasets", str(dataset_count)),
            ("Algorithms", str(algorithm_count)),
            ("Metrics", str(metric_count)),
            ("Aggregate Rows", str(len(aggregate))),
            ("Per-Dataset Rows", str(len(per_dataset))),
        ],
    )


def inspect_leaderboard_artifacts_gradio(leaderboard_file):
    if leaderboard_file is None:
        return empty_leaderboard_inspection_summary_html(), pd.DataFrame(), pd.DataFrame(), "", None

    payload, source_name = _load_leaderboard_payload_from_upload(leaderboard_file)
    report = _leaderboard_report_from_payload(payload)
    title = str(report["metadata"].get("name") or "Matheel Leaderboard")
    html_report = benchmark_report_html(report, title=title)
    export_root = make_temp_workspace("matheel-leaderboard-inspect-")
    artifacts = write_leaderboard_artifacts(
        report,
        export_root,
        basename="leaderboard_report",
    )
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "leaderboard_report_artifacts.zip",
    )
    return (
        leaderboard_inspection_summary_html(report, source_name),
        leaderboard_display_frame(report["aggregate"]),
        leaderboard_display_frame(report["per_dataset"]),
        html_report,
        artifacts_zip,
    )


def ready_leaderboard_algorithm_configs(
    selected_algorithms,
    model_name,
    vector_backend,
    runtime_device,
    preprocess_mode,
    code_language,
    lexical_tokenizer,
):
    if selected_algorithms is None:
        selected = list(READY_LEADERBOARD_ALGORITHM_CHOICES)
    else:
        selected = list(selected_algorithms)
    if not selected:
        raise gr.Error("Select at least one algorithm preset.")
    configs = []
    for name in selected:
        options = dataset_similarity_options(
            name,
            model_name,
            vector_backend,
            runtime_device,
            preprocess_mode,
            code_language,
            lexical_tokenizer,
        )
        configs.append({"name": str(name), **options})
    return configs


def _uploaded_file_paths(uploaded_files):
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        return [
            path
            for path in (_uploaded_file_path(item) for item in uploaded_files)
            if path is not None
        ]
    path = _uploaded_file_path(uploaded_files)
    return [] if path is None else [path]


def _detect_normalized_dataset_task(root):
    is_pair = _looks_like_pair_dataset_root(root)
    is_retrieval = _looks_like_retrieval_dataset_root(root)
    if is_pair and is_retrieval:
        raise gr.Error("Dataset archive contains an ambiguous normalized dataset.")
    if is_pair:
        return "pair"
    if is_retrieval:
        return "retrieval"
    return None


def _find_normalized_dataset_roots(root):
    path = Path(root)
    task = _detect_normalized_dataset_task(path)
    if task is not None:
        return [(path, task)]

    candidates = []
    seen = set()
    for metadata_path in sorted(path.rglob("metadata.json")):
        candidate = metadata_path.parent
        task = _detect_normalized_dataset_task(candidate)
        if task is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append((candidate, task))
    if not candidates:
        raise gr.Error("Could not find a normalized Matheel dataset in the uploaded archive.")
    return candidates


def _normalized_dataset_roots_from_upload(uploaded_path):
    path = Path(uploaded_path)
    if path.is_dir():
        return _find_normalized_dataset_roots(path)
    if not zipfile.is_zipfile(path):
        raise gr.Error(
            "Ready leaderboard inputs must be normalized dataset ZIP archives or directories."
        )
    extract_root = make_temp_workspace("matheel-ready-leaderboard-upload-")
    _safe_extract_uploaded_zip(path, extract_root)
    return _find_normalized_dataset_roots(extract_root)


def _normalized_dataset_metadata_name(root):
    metadata_path = Path(root) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    return str(metadata.get("name") or Path(root).name)


def _unique_dataset_name(name, used_names):
    base = str(name or "dataset").strip() or "dataset"
    candidate = base
    index = 2
    while candidate in used_names:
        candidate = f"{base}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def ready_leaderboard_dataset_configs(uploaded_files, pair_threshold, retrieval_k):
    paths = _uploaded_file_paths(uploaded_files)
    if not paths:
        raise gr.Error("Upload at least one normalized dataset ZIP.")

    datasets = []
    used_roots = set()
    used_names = set()
    for path in paths:
        for dataset_root, task_family in _normalized_dataset_roots_from_upload(path):
            resolved = Path(dataset_root).resolve()
            if resolved in used_roots:
                continue
            used_roots.add(resolved)
            name = _unique_dataset_name(_normalized_dataset_metadata_name(dataset_root), used_names)
            config = {"name": name, "task": task_family, "path": os.fspath(dataset_root)}
            if task_family == "retrieval":
                config["k"] = int(float(retrieval_k or 10))
            else:
                config["threshold"] = float(pair_threshold)
            datasets.append(config)

    if not datasets:
        raise gr.Error("Could not find a normalized Matheel dataset in the uploaded files.")
    return datasets


def ready_leaderboard_summary_html(report, uploaded_count, algorithm_count):
    aggregate = report["aggregate"]
    per_dataset = report["per_dataset"]
    dataset_count = per_dataset["dataset_name"].nunique() if "dataset_name" in per_dataset else 0
    metric_count = aggregate["metric"].nunique() if "metric" in aggregate else 0
    return summary_panel_html(
        "Ready Leaderboard",
        [
            ("Datasets", str(dataset_count)),
            ("Uploads", str(uploaded_count)),
            ("Algorithms", str(algorithm_count)),
            ("Metrics", str(metric_count)),
            ("Aggregate Rows", str(len(aggregate))),
            ("Per-Dataset Rows", str(len(per_dataset))),
        ],
    )


def run_ready_leaderboard_gradio(
    dataset_files,
    selected_algorithms,
    model_name,
    vector_backend,
    runtime_device,
    preprocess_mode,
    code_language,
    lexical_tokenizer,
    pair_threshold,
    retrieval_k,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if not dataset_files:
        return empty_ready_leaderboard_summary_html(), pd.DataFrame(), pd.DataFrame(), "", None

    if progress is not None:
        progress(0.1, desc="Preparing datasets")
    datasets = ready_leaderboard_dataset_configs(dataset_files, pair_threshold, retrieval_k)
    algorithms = ready_leaderboard_algorithm_configs(
        selected_algorithms,
        model_name,
        vector_backend,
        runtime_device,
        preprocess_mode,
        code_language,
        lexical_tokenizer,
    )
    manifest = {
        "name": "gradio_ready_leaderboard",
        "seed": int(float(resolve_numeric_value(seed, 7))),
        "pair_metrics": list(READY_LEADERBOARD_PAIR_METRICS),
        "retrieval_metrics": list(READY_LEADERBOARD_RETRIEVAL_METRICS),
        "datasets": datasets,
        "algorithms": algorithms,
    }
    if progress is not None:
        progress(0.35, desc="Running leaderboard")
    export_root = make_temp_workspace("matheel-ready-leaderboard-")
    try:
        report, artifacts = run_leaderboard(
            manifest,
            output_dir=export_root,
            basename="ready_leaderboard",
        )
    except (ImportError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    if progress is not None:
        progress(0.9, desc="Packaging artifacts")
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "ready_leaderboard_artifacts.zip",
    )
    if progress is not None:
        progress(1.0, desc="Ready leaderboard complete")
    return (
        ready_leaderboard_summary_html(
            report, len(_uploaded_file_paths(dataset_files)), len(algorithms)
        ),
        leaderboard_display_frame(report["aggregate"]),
        leaderboard_display_frame(report["per_dataset"]),
        artifacts["html"].read_text(encoding="utf-8"),
        artifacts_zip,
    )


def _looks_like_pair_dataset_root(root):
    path = Path(root)
    return all((path / name).exists() for name in ("metadata.json", "files.csv", "pairs.csv"))


def _looks_like_retrieval_dataset_root(root):
    path = Path(root)
    return all(
        (path / name).exists()
        for name in ("metadata.json", "files.csv", "queries.csv", "corpus.csv", "qrels.csv")
    )


def _find_normalized_dataset_root(root, task_label):
    expected_task = _dataset_kind_from_task_label(task_label)
    candidates = [
        candidate
        for candidate, task_family in _find_normalized_dataset_roots(root)
        if task_family == expected_task
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise gr.Error(
            "Dataset archive contains multiple normalized datasets. Upload one dataset at a time."
        )
    raise gr.Error("Could not find a normalized Matheel dataset in the uploaded archive.")


def _dataset_root_from_upload(uploaded_file, task_label):
    uploaded_path = _uploaded_file_path(uploaded_file)
    if uploaded_path is None:
        raise gr.Error("Upload a normalized dataset ZIP first.")
    if uploaded_path.is_dir():
        return _find_normalized_dataset_root(uploaded_path, task_label)
    if not zipfile.is_zipfile(uploaded_path):
        raise gr.Error("Dataset upload must be a ZIP archive or a normalized dataset directory.")

    extract_root = make_temp_workspace("matheel-dataset-upload-")
    _safe_extract_uploaded_zip(uploaded_path, extract_root)
    return _find_normalized_dataset_root(extract_root, task_label)


def dataset_similarity_options(
    preset_name,
    model_name,
    vector_backend,
    runtime_device,
    preprocess_mode,
    code_language,
    lexical_tokenizer="raw",
):
    preset = metric_preset_options(preset_name)
    feature_weights = {
        "semantic": preset["semantic_weight"],
        "levenshtein": preset["levenshtein_weight"],
        "jaro_winkler": preset["jaro_winkler_weight"],
        "winnowing": preset["winnowing_weight"],
        "gst": preset["gst_weight"],
    }
    if "Code Metric" in preset["features"]:
        feature_weights["code_metric"] = preset["code_metric_weight"]

    code_metric = preset["code_metric"] if "Code Metric" in preset["features"] else "none"
    return {
        "feature_weights": normalize_feature_weights_map(feature_weights),
        "model_name": str(model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "vector_backend": str(vector_backend or "auto").strip() or "auto",
        "device": str(runtime_device or "auto").strip() or "auto",
        "preprocess_mode": str(preprocess_mode or "none").strip() or "none",
        "code_language": str(code_language or "python").strip() or "python",
        "lexical_tokenizer": str(lexical_tokenizer or "raw").strip() or "raw",
        "code_metric": code_metric,
        "code_metric_weight": preset["code_metric_weight"] if code_metric != "none" else 0.0,
    }


def metrics_dict_frame(metrics):
    rows = []
    for key, value in (metrics or {}).items():
        label = METRIC_LABELS.get(str(key), str(key).replace("_", " ").title())
        if isinstance(value, float):
            display_value = round(value, 4)
        else:
            display_value = value
        rows.append({"Metric": label, "Value": display_value})
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def dataset_scores_display_frame(scored, task_label):
    frame = scored.copy() if isinstance(scored, pd.DataFrame) else pd.DataFrame(scored)
    raw_records = frame.to_dict(orient="records")
    columns = (
        RETRIEVAL_DATASET_SCORE_COLUMNS
        if str(task_label).startswith("Retrieval")
        else PAIR_DATASET_SCORE_COLUMNS
    )
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        frame.attrs["raw_scored_records"] = raw_records
        return frame
    display = frame[selected].copy()
    if "similarity_score" in display.columns:
        raw_scores = display["similarity_score"].astype(float)
        display["Interpretation"] = raw_scores.map(score_band_label)
        display["similarity_score"] = raw_scores.round(4)
    display = display.rename(
        columns={
            "left_id": "Left File",
            "right_id": "Right File",
            "label": "Label",
            "query_id": "Query",
            "document_id": "Document",
            "relevance": "Relevance",
            "similarity_score": "Similarity Score",
        }
    )
    display.attrs["raw_scored_records"] = raw_records
    return display


def _dataset_count_items(task_label, dataset, scored):
    if str(task_label).startswith("Retrieval"):
        return [
            ("Queries", str(len(dataset.queries))),
            ("Documents", str(len(dataset.corpus))),
            ("Results", str(len(scored))),
        ]
    positives = int(scored["label"].sum()) if "label" in scored else 0
    return [
        ("Pairs", str(len(scored))),
        ("Positive Labels", str(positives)),
        ("Negative Labels", str(max(0, len(scored) - positives))),
    ]


def dataset_evaluation_summary_html(
    task_label, dataset, scored, metrics, elapsed_seconds, preset_name
):
    scores = (
        scored["similarity_score"].astype(float)
        if "similarity_score" in scored
        else pd.Series(dtype=float)
    )
    top_score = scores.max() if not scores.empty else 0.0
    task_name = "Retrieval Dataset" if str(task_label).startswith("Retrieval") else "Pair Dataset"
    items = [
        ("Dataset", str(dataset.metadata.get("name") or dataset.root.name)),
        ("Task", task_name),
        ("Metric Preset", str(preset_name or "Balanced")),
        ("Top Score", f"{float(top_score):.3f}"),
        ("Top Interpretation", score_band_label(float(top_score))),
        ("Elapsed", format_elapsed_seconds(elapsed_seconds)),
    ]
    items.extend(_dataset_count_items(task_label, dataset, scored))
    if str(task_label).startswith("Retrieval"):
        items.append(("MAP", f"{float(metrics.get('mean_average_precision', 0.0)):.3f}"))
        items.append(("NDCG at k", f"{float(metrics.get('ndcg_at_k', 0.0)):.3f}"))
    else:
        items.append(("F1", f"{float(metrics.get('f1', 0.0)):.3f}"))
        items.append(("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.3f}"))
    return summary_panel_html("Dataset Evaluation", items)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return Path(path)


def dataset_evaluation_leaderboard_payload(
    task_label,
    dataset,
    scored,
    metrics,
    algorithm_name,
    similarity_options,
):
    """Build an Inspector-compatible leaderboard payload for one dataset run."""
    task_family = "retrieval" if str(task_label).startswith("Retrieval") else "pair"
    dataset_name = str(dataset.metadata.get("name") or dataset.root.name)
    resolved_algorithm = str(algorithm_name or "Custom Dataset Evaluation")
    dataset_source = str(dataset.metadata.get("source") or "uploaded")
    sample_count = int(len(scored))
    metric_rows = []
    for metric, value in sorted(dict(metrics or {}).items()):
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        metric_rows.append(
            {
                "task_family": task_family,
                "dataset_name": dataset_name,
                "algorithm_name": resolved_algorithm,
                "metric": str(metric),
                "score": score,
                "sample_count": sample_count,
                "dataset_source": dataset_source,
                "algorithm_kind": "metric_preset",
                "rank": 1,
            }
        )
    aggregate_rows = [
        {
            "task_family": row["task_family"],
            "algorithm_name": row["algorithm_name"],
            "metric": row["metric"],
            "mean_score": row["score"],
            "median_score": row["score"],
            "dataset_count": 1,
            "sample_count": row["sample_count"],
            "rank": 1,
        }
        for row in metric_rows
    ]
    return {
        "schema_version": 1,
        "metadata": {
            "schema_version": 1,
            "name": f"{dataset_name} Dataset Evaluation",
            "pair_metrics": [row["metric"] for row in metric_rows]
            if task_family == "pair"
            else [],
            "retrieval_metrics": [row["metric"] for row in metric_rows]
            if task_family == "retrieval"
            else [],
        },
        "manifest": {
            "schema_version": 1,
            "name": f"{dataset_name}_dataset_evaluation",
            "datasets": [{"name": dataset_name, "task_family": task_family}],
            "algorithms": [
                {
                    "name": resolved_algorithm,
                    "similarity_options": similarity_options,
                }
            ],
        },
        "cards": {
            "datasets": [
                {
                    "name": dataset_name,
                    "card_type": "dataset",
                    "task_family": task_family,
                }
            ],
            "algorithms": [
                {
                    "name": resolved_algorithm,
                    "card_type": "algorithm",
                    "algorithm_kind": "metric_preset",
                }
            ],
        },
        "per_dataset": metric_rows,
        "aggregate": aggregate_rows,
    }


def _write_leaderboard_artifacts(
    output_dir,
    task_label,
    dataset,
    scored,
    metrics,
    similarity_options,
    algorithm_name=None,
    resample_metrics=None,
    resample_summary=None,
):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "retrieval" if str(task_label).startswith("Retrieval") else "pair"
    scored_path = root / f"{prefix}_scored_rows.csv"
    metrics_path = root / f"{prefix}_metrics.json"
    manifest_path = root / "leaderboard_manifest.json"
    leaderboard_report_path = root / "leaderboard_report.json"
    reproducibility_path = root / "reproducibility.json"
    scored.to_csv(scored_path, index=False)
    _write_json(metrics_path, metrics)

    files = {
        "scored_rows": scored_path.name,
        "metrics": metrics_path.name,
        "reproducibility": reproducibility_path.name,
    }
    if resample_metrics is not None and not resample_metrics.empty:
        resample_metrics_path = root / f"{prefix}_resampling_metrics.csv"
        resample_metrics.to_csv(resample_metrics_path, index=False)
        files["resampling_metrics"] = resample_metrics_path.name
    if resample_summary is not None and not resample_summary.empty:
        resample_summary_path = root / f"{prefix}_resampling_summary.csv"
        resample_summary.to_csv(resample_summary_path, index=False)
        files["resampling_summary"] = resample_summary_path.name
    if prefix == "pair":
        _, threshold_artifacts = write_threshold_tuning_report_artifacts(
            scored,
            root,
            score_key="similarity_score",
            label_key="label",
            basename="pair_threshold_tuning",
        )
        for name, path in threshold_artifacts.items():
            files[f"threshold_{name}"] = Path(path).name

    manifest = {
        "schema_version": 1,
        "workflow": "gradio_dataset_evaluation",
        "dataset_kind": "retrieval"
        if str(task_label).startswith("Retrieval")
        else "pair_classification",
        "dataset_name": str(dataset.metadata.get("name") or dataset.root.name),
        "files": files,
        "similarity_options": similarity_options,
    }
    _write_json(manifest_path, manifest)
    files["manifest"] = manifest_path.name
    _write_json(
        leaderboard_report_path,
        dataset_evaluation_leaderboard_payload(
            task_label,
            dataset,
            scored,
            metrics,
            algorithm_name,
            similarity_options,
        ),
    )
    files["leaderboard_report"] = leaderboard_report_path.name

    snapshot = collect_reproducibility_snapshot(
        dataset.root,
        run_configs=[manifest],
        result_attrs={"metrics": metrics},
    )
    write_reproducibility_snapshot(snapshot, reproducibility_path)

    zip_path = root / "leaderboard_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in sorted(files.values()):
            archive.write(root / file_name, arcname=file_name)
    return zip_path


def _resample_pair_scores(scored, folds, seed, threshold):
    if not folds:
        return pd.DataFrame(), pd.DataFrame()
    _validate_pair_resampling_folds(scored, int(folds))
    splits = kfold_splits(
        len(scored),
        n_splits=int(folds),
        shuffle=True,
        seed=int(seed),
        labels=scored["label"].tolist() if "label" in scored else None,
    )
    return evaluate_pair_resamples(scored, splits, threshold=threshold)


def _resample_retrieval_scores(scored, dataset, folds, seed, k):
    if not folds:
        return pd.DataFrame(), pd.DataFrame()
    query_ids = tuple(sorted(scored["query_id"].astype(str).unique().tolist()))
    _validate_retrieval_resampling_folds(query_ids, int(folds))
    splits = kfold_splits(query_ids, n_splits=int(folds), shuffle=True, seed=int(seed))
    return evaluate_retrieval_resamples(scored, splits, qrels=dataset.qrels, k=k)


def _validate_resampling_fold_count(item_count, folds, item_label):
    if folds < 2:
        raise gr.Error(
            "K-fold resampling needs at least 2 folds, or set folds to 0 to turn it off."
        )
    if folds > item_count:
        raise gr.Error(f"K-fold resampling folds must not exceed the number of {item_label}.")


def _validate_pair_resampling_folds(scored, folds):
    _validate_resampling_fold_count(len(scored), folds, "scored pair rows")
    if "label" not in scored:
        return
    label_counts = scored["label"].value_counts()
    if not label_counts.empty and folds > int(label_counts.min()):
        raise gr.Error("K-fold resampling folds must not exceed the smallest label count.")


def _validate_retrieval_resampling_folds(query_ids, folds):
    _validate_resampling_fold_count(len(query_ids), folds, "queries")


def dataset_validation_summary_html(report):
    return summary_panel_html(
        "Dataset Validation",
        [
            ("Dataset", str(report.get("dataset_name") or "dataset")),
            ("Kind", str(report.get("dataset_kind") or "unknown").replace("_", " ")),
            ("Status", str(report.get("status") or "unknown")),
            ("Errors", str(report.get("error_count", 0))),
            ("Warnings", str(report.get("warning_count", 0))),
        ],
        variant="error"
        if report.get("error_count")
        else "empty"
        if report.get("warning_count")
        else "summary",
    )


def dataset_validation_issues_frame(report):
    issues = list((report or {}).get("issues") or [])
    if not issues:
        return pd.DataFrame(columns=["Severity", "Code", "Message", "Count"])
    frame = pd.DataFrame(issues)
    return frame.rename(
        columns={
            "severity": "Severity",
            "code": "Code",
            "message": "Message",
            "count": "Count",
        }
    )[["Severity", "Code", "Message", "Count"]]


def validate_dataset_gradio(dataset_file, task_label, progress=gr.Progress(track_tqdm=True)):
    if dataset_file is None:
        return empty_dataset_validation_summary_html(), pd.DataFrame(), "", None
    dataset_root = _dataset_root_from_upload(dataset_file, task_label)
    if progress is not None:
        progress(0.3, desc="Inspecting dataset")
    export_root = make_temp_workspace("matheel-dataset-validation-")
    report, artifacts = write_dataset_validation_report(
        dataset_root,
        export_root,
        kind=_dataset_kind_from_task_label(task_label),
        basename="dataset_validation",
    )
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "dataset_validation_artifacts.zip",
    )
    if progress is not None:
        progress(1.0, desc="Dataset validation complete")
    return (
        dataset_validation_summary_html(report),
        dataset_validation_issues_frame(report),
        artifacts["report_html"].read_text(encoding="utf-8"),
        artifacts_zip,
    )


def dataset_upload_identity(dataset_file, task_label):
    """Return a stable identity used to bind validation to one upload and task."""
    uploaded_path = _uploaded_file_path(dataset_file)
    if uploaded_path is None or not uploaded_path.exists():
        return None
    digest = hashlib.sha256()
    with uploaded_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "sha256": digest.hexdigest(),
        "task": _dataset_kind_from_task_label(task_label),
    }


def dataset_journey_status_html(step, message, variant="status"):
    return status_panel_html(f"Step {int(step)}", message, variant=variant)


def validate_dataset_journey_gradio(
    dataset_file, task_label, progress=gr.Progress(track_tqdm=True)
):
    outputs = validate_dataset_gradio(dataset_file, task_label, progress=progress)
    issues = outputs[1]
    error_count = 0
    if isinstance(issues, pd.DataFrame) and not issues.empty and "Severity" in issues:
        error_count = int(issues["Severity"].astype(str).str.lower().eq("error").sum())
    identity = dataset_upload_identity(dataset_file, task_label) if error_count == 0 else None
    if identity is None:
        status = dataset_journey_status_html(
            2,
            "Validation must pass before evaluation. Fix the reported errors, then validate again.",
            variant="error" if dataset_file is not None else "empty",
        )
    else:
        status = dataset_journey_status_html(
            2,
            "Validation passed. Review any warnings, then run the dataset evaluation.",
        )
    return (*outputs, identity, gr.update(interactive=identity is not None), status)


def evaluate_dataset_journey_gradio(
    validation_state,
    dataset_file,
    task_label,
    *evaluation_args,
    progress=gr.Progress(track_tqdm=True),
):
    current_identity = dataset_upload_identity(dataset_file, task_label)
    if not validation_state or validation_state != current_identity:
        raise gr.Error("Validate this dataset and task successfully before running evaluation.")
    outputs = evaluate_dataset_gradio_for_journey(
        dataset_file,
        task_label,
        *evaluation_args,
        progress=progress,
    )
    is_pair = not str(task_label).startswith("Retrieval")
    status = dataset_journey_status_html(
        3,
        (
            "Evaluation complete. Tune the pair threshold or explain a scored row, then download artifacts."
            if is_pair
            else "Evaluation complete. Review ranked rows and metrics, then download the artifacts."
        ),
    )
    return (
        gr.update(value=outputs[0], visible=True),
        gr.update(value=outputs[1], visible=True),
        gr.update(value=outputs[2], visible=True),
        gr.update(value=outputs[3], visible=True),
        gr.update(value=outputs[4], visible=True),
        outputs[5],
        outputs[6],
        status,
        gr.update(visible=is_pair),
        gr.update(visible=is_pair),
        gr.update(visible=is_pair),
        gr.update(visible=True),
    )


def handoff_dataset_to_explain(dataset_file, task_label):
    if dataset_file is None:
        raise gr.Error("Run a dataset evaluation before continuing to Explain.")
    return (
        dataset_file,
        task_label,
        gr.update(selected="explain"),
        gr.update(selected="dataset-map"),
    )


def handoff_dataset_to_reports(artifacts_path):
    if artifacts_path is None:
        raise gr.Error("Run a dataset evaluation before continuing to Reports.")
    return (
        artifacts_path,
        gr.update(selected="reports"),
        gr.update(selected="inspect-artifacts"),
    )


def threshold_tuning_journey_gradio(scored_state, optimize, progress=gr.Progress(track_tqdm=True)):
    outputs = threshold_tuning_gradio(scored_state, optimize, progress=progress)
    return (
        gr.update(value=outputs[0], visible=True),
        gr.update(value=outputs[1], visible=True),
        gr.update(value=outputs[2], visible=True),
        outputs[3],
        dataset_journey_status_html(
            4,
            "Threshold tuning complete. Review the sweep, then download calibration artifacts.",
        ),
    )


def explain_scored_pair_journey_gradio(
    scored_state,
    row_index,
    segment_mode,
    high_threshold,
    medium_threshold,
    low_threshold,
    chunk_size,
    progress=gr.Progress(track_tqdm=True),
):
    outputs = explain_scored_pair_gradio(
        scored_state,
        row_index,
        segment_mode,
        high_threshold,
        medium_threshold,
        low_threshold,
        chunk_size,
        progress=progress,
    )
    return (
        gr.update(value=outputs[0], visible=True),
        gr.update(value=outputs[1], visible=True),
        gr.update(value=outputs[2], visible=True),
        outputs[3],
        dataset_journey_status_html(
            4,
            "Pair explanation complete. Review the matched regions, then download the evidence bundle.",
        ),
    )


def reset_dataset_results():
    """Clear evaluation and every dependent stage after a scoring option changes."""
    return (
        None,
        gr.update(
            value=summary_panel_html(
                "Dataset Evaluation",
                [("Status", "Ready to evaluate"), ("Artifacts", "None")],
                variant="empty",
            ),
            visible=False,
        ),
        gr.update(value=pd.DataFrame(columns=["Metric", "Value"]), visible=False),
        gr.update(
            value=pd.DataFrame(columns=["Similarity Score", "Interpretation"]),
            visible=False,
        ),
        gr.update(value=pd.DataFrame(), visible=False),
        gr.update(value=pd.DataFrame(), visible=False),
        None,
        gr.update(value=empty_threshold_tuning_summary_html(), visible=False),
        gr.update(value=pd.DataFrame(), visible=False),
        gr.update(value="", visible=False),
        None,
        gr.update(value=empty_pair_explanation_summary_html(), visible=False),
        gr.update(value=pd.DataFrame(), visible=False),
        gr.update(value="", visible=False),
        None,
        dataset_journey_status_html(3, "Settings changed. Run evaluation to refresh every result."),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def reset_dataset_threshold_stage():
    return (
        gr.update(value=empty_threshold_tuning_summary_html(), visible=False),
        gr.update(value=pd.DataFrame(), visible=False),
        gr.update(value="", visible=False),
        None,
        dataset_journey_status_html(
            4, "Threshold options changed. Tune again to refresh calibration."
        ),
    )


def reset_dataset_explanation_stage():
    return (
        gr.update(value=empty_pair_explanation_summary_html(), visible=False),
        gr.update(value=pd.DataFrame(), visible=False),
        gr.update(value="", visible=False),
        None,
        dataset_journey_status_html(
            4, "Explanation options changed. Explain again to refresh evidence."
        ),
    )


def reset_dataset_upload_journey():
    """Invalidate validation and downstream stages after upload/task changes."""
    return (
        None,
        empty_dataset_validation_summary_html(),
        pd.DataFrame(columns=["Severity", "Code", "Message", "Count"]),
        "",
        None,
        gr.update(interactive=False),
        dataset_journey_status_html(1, "Upload a dataset, choose its task, then validate it."),
        *reset_dataset_results()[:-5],
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def threshold_tuning_summary_html(report):
    summary = report["summary"]
    optimized = summary["optimized_threshold"]
    items = [
        ("Pairs", str(summary["pair_count"])),
        ("Positive Labels", str(summary["positive_count"])),
        ("Negative Labels", str(summary["negative_count"])),
        ("Optimized Metric", str(summary["optimized_metric"])),
        ("Best Threshold", f"{float(optimized['threshold']):.4f}"),
        ("Best F1", f"{float(optimized['f1']):.3f}"),
        ("Best Accuracy", f"{float(optimized['accuracy']):.3f}"),
    ]
    if summary.get("auroc") is not None:
        items.append(("AUROC", f"{float(summary['auroc']):.3f}"))
    if summary.get("average_precision") is not None:
        items.append(("Average Precision", f"{float(summary['average_precision']):.3f}"))
    return summary_panel_html(
        "Threshold Tuning",
        items,
        variant="empty" if summary.get("warnings") else "summary",
    )


def threshold_sweep_display_frame(report):
    frame = report["threshold_sweep"].copy()
    if frame.empty:
        return pd.DataFrame(columns=["Threshold", "Precision", "Recall", "F1", "Accuracy"])
    for column in ("threshold", "precision", "recall", "f1", "accuracy"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float).round(4)
    columns = [
        column
        for column in (
            "threshold",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "accuracy",
        )
        if column in frame.columns
    ]
    return frame[columns].rename(
        columns={
            "threshold": "Threshold",
            "true_positive": "TP",
            "false_positive": "FP",
            "true_negative": "TN",
            "false_negative": "FN",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "accuracy": "Accuracy",
        }
    )


def threshold_tuning_gradio(scored_state, optimize, progress=gr.Progress(track_tqdm=True)):
    if not scored_state:
        return empty_threshold_tuning_summary_html(), pd.DataFrame(), "", None
    if scored_state.get("task") != "pair":
        raise gr.Error("Threshold tuning is only available for pair-classification scores.")
    scored = pd.DataFrame(scored_state.get("scored") or [])
    if scored.empty:
        return empty_threshold_tuning_summary_html(), pd.DataFrame(), "", None
    if progress is not None:
        progress(0.35, desc="Sweeping thresholds")
    export_root = make_temp_workspace("matheel-threshold-tuning-")
    try:
        report, artifacts = write_threshold_tuning_report_artifacts(
            scored,
            export_root,
            score_key="similarity_score",
            label_key="label",
            optimize=optimize,
            basename="threshold_tuning",
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "threshold_tuning_artifacts.zip",
    )
    if progress is not None:
        progress(1.0, desc="Threshold tuning complete")
    return (
        threshold_tuning_summary_html(report),
        threshold_sweep_display_frame(report),
        artifacts["report_html"].read_text(encoding="utf-8"),
        artifacts_zip,
    )


def explain_scored_pair_gradio(
    scored_state,
    row_index,
    segment_mode,
    high_threshold,
    medium_threshold,
    low_threshold,
    chunk_size,
    progress=gr.Progress(track_tqdm=True),
):
    if not scored_state:
        return empty_pair_explanation_summary_html(), pd.DataFrame(), "", None
    if scored_state.get("task") != "pair":
        raise gr.Error(
            "Scored pair explanations are only available for pair-classification scores."
        )
    dataset_root = scored_state.get("dataset_root")
    if not dataset_root:
        raise gr.Error("Run pair dataset evaluation before explaining a scored pair.")
    scored = pd.DataFrame(scored_state.get("scored") or [])
    if scored.empty:
        return empty_pair_explanation_summary_html(), pd.DataFrame(), "", None
    if progress is not None:
        progress(0.35, desc="Building pair explanation")
    export_root = make_temp_workspace("matheel-scored-pair-explanation-")
    try:
        explanation, artifacts = write_scored_pair_explanation(
            scored,
            dataset_root,
            export_root,
            row_index=int(float(row_index or 0)),
            segment_mode=segment_mode,
            high_threshold=float(high_threshold),
            medium_threshold=float(medium_threshold),
            low_threshold=float(low_threshold),
            chunk_size=int(float(chunk_size or 1)),
            basename="scored_pair_explanation",
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    artifacts_zip = _zip_artifact_paths(
        artifacts.values(),
        export_root / "scored_pair_explanation_artifacts.zip",
    )
    if progress is not None:
        progress(1.0, desc="Pair explanation complete")
    return (
        pair_explanation_summary_html(explanation),
        pair_explanation_matches_frame(explanation),
        artifacts["html"].read_text(encoding="utf-8"),
        artifacts_zip,
    )


def evaluate_dataset_gradio(
    dataset_file,
    task_label,
    metric_preset,
    model_name,
    vector_backend,
    runtime_device,
    preprocess_mode,
    code_language,
    lexical_tokenizer,
    threshold,
    retrieval_k,
    resampling_folds,
    resampling_seed,
    progress=gr.Progress(track_tqdm=True),
):
    if dataset_file is None:
        empty_metrics = pd.DataFrame(columns=["Metric", "Value"])
        empty_scores = pd.DataFrame(columns=["Similarity Score", "Interpretation"])
        return (
            summary_panel_html(
                "Dataset Evaluation",
                [("Status", "No dataset uploaded"), ("Artifacts", "None")],
                variant="empty",
            ),
            empty_metrics,
            empty_scores,
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        )

    dataset_root = _dataset_root_from_upload(dataset_file, task_label)
    similarity_options = dataset_similarity_options(
        metric_preset,
        model_name,
        vector_backend,
        runtime_device,
        preprocess_mode,
        code_language,
        lexical_tokenizer,
    )
    start_time = perf_counter()
    if str(task_label).startswith("Retrieval"):
        dataset = load_retrieval_dataset(dataset_root)
        scored, metrics = evaluate_retrieval_dataset(
            dataset,
            k=int(retrieval_k),
            similarity_options=similarity_options,
        )
        resample_metrics, resample_summary = _resample_retrieval_scores(
            scored,
            dataset,
            int(resampling_folds or 0),
            int(resampling_seed),
            int(retrieval_k),
        )
    else:
        dataset = load_pair_dataset(dataset_root)
        scored, metrics = evaluate_pair_dataset(
            dataset,
            threshold=float(threshold),
            similarity_options=similarity_options,
        )
        resample_metrics, resample_summary = _resample_pair_scores(
            scored,
            int(resampling_folds or 0),
            int(resampling_seed),
            float(threshold),
        )
    elapsed_seconds = elapsed_seconds_between(start_time, perf_counter())
    if progress is not None:
        progress(1.0, desc="Dataset evaluation complete")

    export_root = os.fspath(make_temp_workspace("matheel-leaderboard-"))
    artifacts_zip = _write_leaderboard_artifacts(
        export_root,
        task_label,
        dataset,
        scored,
        metrics,
        similarity_options,
        algorithm_name=metric_preset,
        resample_metrics=resample_metrics,
        resample_summary=resample_summary,
    )
    return (
        dataset_evaluation_summary_html(
            task_label, dataset, scored, metrics, elapsed_seconds, metric_preset
        ),
        metrics_dict_frame(metrics),
        dataset_scores_display_frame(scored, task_label),
        resample_metrics,
        resample_summary,
        str(artifacts_zip),
    )


def evaluate_dataset_gradio_with_state(
    dataset_file,
    task_label,
    metric_preset,
    model_name,
    vector_backend,
    runtime_device,
    preprocess_mode,
    code_language,
    lexical_tokenizer,
    threshold,
    retrieval_k,
    resampling_folds,
    resampling_seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Preserve the historical six-output public dataset evaluation contract."""
    return evaluate_dataset_gradio(
        dataset_file,
        task_label,
        metric_preset,
        model_name,
        vector_backend,
        runtime_device,
        preprocess_mode,
        code_language,
        lexical_tokenizer,
        threshold,
        retrieval_k,
        resampling_folds,
        resampling_seed,
        progress=progress,
    )


def evaluate_dataset_gradio_for_journey(*args, **kwargs):
    """Add private UI state without changing the historical public endpoint."""
    outputs = evaluate_dataset_gradio_with_state(*args, **kwargs)
    scored = outputs[2]
    task_label = args[1] if len(args) > 1 else kwargs.get("task_label", DEFAULT_DATASET_TASK)
    dataset_file = args[0] if args else kwargs.get("dataset_file")
    dataset_root = None
    if dataset_file is not None:
        try:
            dataset_root = os.fspath(_dataset_root_from_upload(dataset_file, task_label))
        except gr.Error:
            dataset_root = None
    state = {
        "task": "retrieval" if str(task_label).startswith("Retrieval") else "pair",
        "dataset_root": dataset_root,
        "scored": list(scored.attrs.get("raw_scored_records") or _state_scored_records(scored)),
    }
    return (*outputs, state)


def _state_scored_records(display_frame):
    frame = (
        display_frame.copy()
        if isinstance(display_frame, pd.DataFrame)
        else pd.DataFrame(display_frame)
    )
    if frame.empty:
        return []
    rename = {
        "Left File": "left_id",
        "Right File": "right_id",
        "Label": "label",
        "Similarity Score": "similarity_score",
        "Query": "query_id",
        "Document": "document_id",
        "Relevance": "relevance",
    }
    frame = frame.rename(columns=rename)
    return frame.to_dict(orient="records")


def score_card_html(
    score,
    vector_backend="sentence_transformers",
    runtime_device="auto",
    code_metric="none",
    elapsed_seconds=None,
    threshold=None,
):
    numeric_score = float(score)
    items = [("Similarity Score", f"{numeric_score:.4f}")]
    label, guidance = score_interpretation(numeric_score, threshold=threshold)
    items.append(("Interpretation", label))
    items.append(("Review Hint", guidance))
    if elapsed_seconds is not None:
        items.append(("Elapsed", format_elapsed_seconds(elapsed_seconds)))
    if code_metric and code_metric != "none":
        items.append(("Code Metric", code_metric))
    items.append(("Backend", vector_backend))
    items.append(("Device", runtime_device))
    return summary_panel_html("Pairwise Result", items)


def empty_pair_summary_html():
    return summary_panel_html(
        "Pairwise Result",
        [("Status", "No comparison run"), ("Similarity Score", "None")],
        variant="empty",
    )


def empty_summary_html():
    return summary_panel_html(
        "Collection Results",
        [("Status", "No collection run"), ("Top Pair", "None")],
        variant="empty",
    )


def results_summary_html(results, vector_backend, code_metric, chunking_method, runtime_device):
    if results is None or results.empty:
        return summary_panel_html(
            "Collection Results",
            [("Status", "No pairs met the threshold"), ("Top Pair", "None")],
            variant="empty",
        )

    scores = results["similarity_score"].astype(float)
    top_row = results.iloc[0]
    elapsed_seconds = float(results.attrs.get("elapsed_seconds", 0.0))
    feature_set = results.attrs.get("feature_set", "none")
    high_priority = int((scores >= 0.65).sum())
    return summary_panel_html(
        "Collection Results",
        [
            ("Top Pair", f"{top_row['file_name_1']} vs {top_row['file_name_2']}"),
            ("Top Interpretation", score_band_label(top_row["similarity_score"])),
            ("Pairs", str(len(results))),
            ("High Priority", str(high_priority)),
            ("Average", f"{scores.mean():.3f}"),
            ("Top Score", f"{scores.max():.3f}"),
            ("Elapsed", format_elapsed_seconds(elapsed_seconds)),
            ("Features", feature_set),
            ("Backend", vector_backend),
            ("Code Metric", code_metric),
            ("Chunking", chunking_method),
            ("Device", runtime_device),
        ],
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
    semantic_weight,
    levenshtein_weight,
    jaro_winkler_weight,
    winnowing_weight,
    gst_weight,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    lexical_tokenizer,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
    selected_preparation,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_aggregation,
    chunk_language,
    chunker_options,
    progress=gr.Progress(track_tqdm=True),
):
    if not str(code1 or "").strip() or not str(code2 or "").strip():
        raise gr.Error("Paste both snippets before running pairwise comparison.")

    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    use_semantic = "Embedding" in selected
    use_levenshtein = "Levenshtein" in selected
    use_jaro_winkler = "Jaro-Winkler" in selected
    use_winnowing = "Winnowing" in selected
    use_gst = "GST" in selected
    use_code_metric = "Code Metric" in selected
    effective_preprocess_mode = preprocess_mode if "Preprocessing" in selected_steps else "none"
    effective_chunking_method = chunking_method if "Chunking" in selected_steps else "none"
    effective_code_metric = code_metric if use_code_metric else "none"
    effective_code_metric_weight = code_metric_weight if use_code_metric else 0.0
    effective_levenshtein_weights = "1,1,1"
    if use_levenshtein:
        effective_levenshtein_weights, _ = validate_levenshtein_weights_text(levenshtein_weights)
    effective_winnowing_kgram = 5
    effective_winnowing_window = 4
    if use_winnowing:
        effective_winnowing_kgram = validate_positive_int_value(
            winnowing_kgram,
            "Winnowing k-gram size",
        )
        effective_winnowing_window = validate_positive_int_value(
            winnowing_window,
            "Winnowing window size",
        )
    effective_gst_min_match_length = 5
    if use_gst:
        effective_gst_min_match_length = validate_positive_int_value(
            gst_min_match_length,
            "GST minimum match length",
        )

    effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
    if effective_code_metric.startswith("codebleu"):
        effective_codebleu_component_weights = validate_codebleu_component_weights_text(
            codebleu_component_weights
        )
    metric_kwargs = resolve_metric_kwargs(
        effective_code_metric,
        ruby_mode,
        ruby_graph_timeout_seconds,
        tsed_costs,
        codebertscore_model,
        codebertscore_max_length,
    )

    feature_weights = build_feature_weights(
        use_semantic,
        semantic_weight,
        use_levenshtein,
        levenshtein_weight,
        use_jaro_winkler,
        jaro_winkler_weight,
        use_winnowing,
        winnowing_weight,
        use_gst,
        gst_weight,
        effective_code_metric,
        effective_code_metric_weight,
    )
    start_time = perf_counter()
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
        winnowing_kgram=effective_winnowing_kgram,
        winnowing_window=effective_winnowing_window,
        gst_min_match_length=effective_gst_min_match_length,
        lexical_tokenizer=lexical_tokenizer,
        vector_backend=vector_backend,
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
        device=runtime_device,
        progress_callback=gradio_progress_callback(progress),
        **metric_kwargs,
    )
    elapsed_seconds = elapsed_seconds_between(start_time, perf_counter())
    return score_card_html(
        score,
        vector_backend=vector_backend,
        runtime_device=runtime_device,
        code_metric=effective_code_metric,
        elapsed_seconds=elapsed_seconds,
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
    semantic_weight,
    levenshtein_weight,
    jaro_winkler_weight,
    winnowing_weight,
    gst_weight,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    lexical_tokenizer,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_mode,
    ruby_graph_timeout_seconds,
    tsed_costs,
    codebertscore_model,
    codebertscore_max_length,
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
    progress=gr.Progress(track_tqdm=True),
):
    if zipped_file is None:
        return empty_summary_html(), pd.DataFrame(
            columns=["file_name_1", "file_name_2", "similarity_score"]
        )

    selected = set(selected_features or [])
    selected_steps = set(selected_preparation or [])
    use_semantic = "Embedding" in selected
    use_levenshtein = "Levenshtein" in selected
    use_jaro_winkler = "Jaro-Winkler" in selected
    use_winnowing = "Winnowing" in selected
    use_gst = "GST" in selected
    use_code_metric = "Code Metric" in selected
    effective_preprocess_mode = preprocess_mode if "Preprocessing" in selected_steps else "none"
    effective_chunking_method = chunking_method if "Chunking" in selected_steps else "none"
    effective_code_metric = code_metric if use_code_metric else "none"
    effective_code_metric_weight = code_metric_weight if use_code_metric else 0.0
    effective_levenshtein_weights = "1,1,1"
    if use_levenshtein:
        effective_levenshtein_weights, _ = validate_levenshtein_weights_text(levenshtein_weights)
    effective_winnowing_kgram = 5
    effective_winnowing_window = 4
    if use_winnowing:
        effective_winnowing_kgram = validate_positive_int_value(
            winnowing_kgram,
            "Winnowing k-gram size",
        )
        effective_winnowing_window = validate_positive_int_value(
            winnowing_window,
            "Winnowing window size",
        )
    effective_gst_min_match_length = 5
    if use_gst:
        effective_gst_min_match_length = validate_positive_int_value(
            gst_min_match_length,
            "GST minimum match length",
        )

    effective_codebleu_component_weights = "0.25,0.25,0.25,0.25"
    if effective_code_metric.startswith("codebleu"):
        effective_codebleu_component_weights = validate_codebleu_component_weights_text(
            codebleu_component_weights
        )
    metric_kwargs = resolve_metric_kwargs(
        effective_code_metric,
        ruby_mode,
        ruby_graph_timeout_seconds,
        tsed_costs,
        codebertscore_model,
        codebertscore_max_length,
    )

    feature_weights = build_feature_weights(
        use_semantic,
        semantic_weight,
        use_levenshtein,
        levenshtein_weight,
        use_jaro_winkler,
        jaro_winkler_weight,
        use_winnowing,
        winnowing_weight,
        use_gst,
        gst_weight,
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
        winnowing_kgram=effective_winnowing_kgram,
        winnowing_window=effective_winnowing_window,
        gst_min_match_length=effective_gst_min_match_length,
        lexical_tokenizer=lexical_tokenizer,
        vector_backend=vector_backend,
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
        device=runtime_device,
        progress_callback=gradio_progress_callback(progress),
        **metric_kwargs,
    )
    return results_summary_html(
        results,
        vector_backend,
        effective_code_metric,
        effective_chunking_method,
        runtime_device,
    ), results


ready_made_initial = ready_made_leaderboard_values()
ready_made_filter_initial = ready_made_leaderboard_filter_options()


class ModelRoutingControls(NamedTuple):
    group: object
    model: object
    status: object
    vector_backend: object
    similarity_group: object
    similarity_function: object
    pooling_group: object
    pooling_method: object
    max_token_length: object
    runtime_device: object
    semantic_weight: object


class CodeMetricControls(NamedTuple):
    group: object
    metric: object
    weight: object
    language: object
    codebleu_group: object
    codebleu_component_weights: object
    crystal_group: object
    crystalbleu_max_order: object
    crystalbleu_trivial_ngram_count: object
    ruby_group: object
    ruby_mode: object
    ruby_graph_timeout_seconds: object
    tsed_group: object
    tsed_costs: object
    codebertscore_group: object
    codebertscore_model: object
    codebertscore_max_length: object


class PreparationControls(NamedTuple):
    selected: object
    preprocess_group: object
    preprocess_mode: object
    chunk_group: object
    chunking_method: object
    chunk_size: object
    chunk_overlap: object
    max_chunks: object
    chunk_aggregation: object
    chunk_language: object
    chunker_options: object


class ScoringControls(NamedTuple):
    metric_preset: object
    features: object
    model: ModelRoutingControls
    levenshtein_group: object
    levenshtein_weight: object
    levenshtein_weights: object
    jaro_group: object
    jaro_winkler_weight: object
    jaro_prefix_weight: object
    winnowing_group: object
    winnowing_weight: object
    winnowing_kgram: object
    winnowing_window: object
    gst_group: object
    gst_weight: object
    gst_min_match_length: object
    lexical_tokenizer: object
    code: CodeMetricControls
    preparation: PreparationControls

    def handler_inputs(self):
        """Return the stable Pair/Collection/Suite scoring argument order."""
        return [
            self.features,
            self.model.model,
            self.model.vector_backend,
            self.model.similarity_function,
            self.model.pooling_method,
            self.model.max_token_length,
            self.model.runtime_device,
            self.model.semantic_weight,
            self.levenshtein_weight,
            self.jaro_winkler_weight,
            self.winnowing_weight,
            self.gst_weight,
            self.levenshtein_weights,
            self.jaro_prefix_weight,
            self.winnowing_kgram,
            self.winnowing_window,
            self.gst_min_match_length,
            self.lexical_tokenizer,
            self.code.metric,
            self.code.weight,
            self.code.language,
            self.code.codebleu_component_weights,
            self.code.crystalbleu_max_order,
            self.code.crystalbleu_trivial_ngram_count,
            self.code.ruby_mode,
            self.code.ruby_graph_timeout_seconds,
            self.code.tsed_costs,
            self.code.codebertscore_model,
            self.code.codebertscore_max_length,
            self.preparation.selected,
            self.preparation.preprocess_mode,
            self.preparation.chunking_method,
            self.preparation.chunk_size,
            self.preparation.chunk_overlap,
            self.preparation.max_chunks,
            self.preparation.chunk_aggregation,
            self.preparation.chunk_language,
            self.preparation.chunker_options,
        ]


def build_model_routing_controls():
    with gr.Group(visible=True) as group:
        model = HuggingfaceHubSearch(
            value=DEFAULT_MODEL,
            label="Embedding Model",
            placeholder="Search Hugging Face models",
            search_type="model",
        )
        status = gr.HTML(value=model_status_html(), padding=False)
        vector_backend = gr.Dropdown(
            choices=list(available_vector_backends()),
            value="auto",
            label="Vector Backend",
        )
        with gr.Group(visible=True) as similarity_group:
            similarity_function = gr.Dropdown(
                choices=list(available_similarity_functions()),
                value="cosine",
                label="Similarity Function",
            )
        with gr.Group(visible=True) as pooling_group:
            pooling_method = gr.Dropdown(
                choices=list(available_pooling_methods()),
                value="mean",
                label="Pooling Method",
            )
        max_token_length = gr.Slider(8, 512, value=256, label="Max Tokens per Input", step=1)
        runtime_device = gr.Dropdown(
            choices=list(DEVICE_CHOICES),
            value="auto",
            label="Runtime Device",
        )
        semantic_weight = gr.Slider(
            0,
            1,
            value=DEFAULT_UI_FEATURE_WEIGHTS["semantic"],
            label="Embedding Weight",
            step=0.05,
        )
    return ModelRoutingControls(
        group,
        model,
        status,
        vector_backend,
        similarity_group,
        similarity_function,
        pooling_group,
        pooling_method,
        max_token_length,
        runtime_device,
        semantic_weight,
    )


def build_code_metric_controls():
    with gr.Group(visible=False) as group:
        metric = gr.Dropdown(
            choices=list(CODE_METRIC_CHOICES), value="codebleu", label="Code Metric"
        )
        weight = gr.Slider(
            0,
            1,
            value=DEFAULT_UI_FEATURE_WEIGHTS["code_metric"],
            label="Code Metric Weight",
            step=0.05,
        )
        language = gr.Dropdown(
            choices=list(available_code_metric_languages()),
            value="python",
            label="Code Language",
        )
        with gr.Group(visible=True) as codebleu_group:
            codebleu_component_weights = gr.Textbox(
                value="0.25,0.25,0.25,0.25",
                label="CodeBLEU Weights",
            )
        with gr.Group(visible=False) as crystal_group:
            crystalbleu_max_order = gr.Slider(1, 8, value=4, label="Max N-gram Order", step=1)
            crystalbleu_trivial_ngram_count = gr.Slider(
                0,
                500,
                value=50,
                label="Ignored Frequent N-grams",
                step=5,
            )
        with gr.Group(visible=False) as ruby_group:
            ruby_mode = gr.Dropdown(
                choices=["auto", "graph", "tree", "string", "ngram"],
                value=DEFAULT_RUBY_MODE,
                label="RUBY Mode",
            )
            ruby_graph_timeout_seconds = gr.Number(
                value=DEFAULT_RUBY_GRAPH_TIMEOUT,
                precision=2,
                label="Graph Timeout (seconds)",
            )
        with gr.Group(visible=False) as tsed_group:
            tsed_costs = gr.Textbox(
                value=DEFAULT_TSED_COSTS,
                label="Delete, Insert, Rename Costs",
                placeholder="1,1,1",
            )
        with gr.Group(visible=False) as codebertscore_group:
            codebertscore_model = HuggingfaceHubSearch(
                value=DEFAULT_CODEBERTSCORE_MODEL,
                label="CodeBERTScore Model",
                placeholder="Search Hugging Face models",
                search_type="model",
            )
            codebertscore_max_length = gr.Slider(
                0,
                DEFAULT_CODEBERTSCORE_MODEL_MAX_SEQUENCE,
                value=DEFAULT_CODEBERTSCORE_MAX_LENGTH,
                step=1,
                label="CodeBERTScore Max Length (0 = model default)",
            )
    return CodeMetricControls(
        group,
        metric,
        weight,
        language,
        codebleu_group,
        codebleu_component_weights,
        crystal_group,
        crystalbleu_max_order,
        crystalbleu_trivial_ngram_count,
        ruby_group,
        ruby_mode,
        ruby_graph_timeout_seconds,
        tsed_group,
        tsed_costs,
        codebertscore_group,
        codebertscore_model,
        codebertscore_max_length,
    )


def build_preparation_controls():
    with gr.Accordion("Advanced preparation", open=False):
        selected = gr.CheckboxGroup(
            choices=["Preprocessing", "Chunking"],
            value=[],
            label="Steps",
        )
        with gr.Group(visible=False) as preprocess_group:
            preprocess_mode = gr.Dropdown(
                choices=list(PREPROCESSING_UI_CHOICES),
                value=(PREPROCESSING_UI_CHOICES[0] if PREPROCESSING_UI_CHOICES else "basic"),
                label="Preprocessing Mode",
            )
        with gr.Group(visible=False) as chunk_group:
            chunking_method = gr.Dropdown(
                choices=list(CHONKIE_UI_METHODS),
                value=(CHONKIE_UI_METHODS[0] if CHONKIE_UI_METHODS else "code"),
                label="Chunking Method",
            )
            chunk_size = gr.Slider(10, 400, value=120, label="Chunk Size", step=10)
            chunk_overlap = gr.Slider(0, 200, value=0, label="Chunk Overlap", step=5)
            max_chunks = gr.Slider(0, 20, value=0, label="Max Chunks per File", step=1)
            chunk_aggregation = gr.Dropdown(
                choices=list(available_chunk_aggregations()),
                value="mean",
                label="Chunk Aggregation",
            )
            chunk_language = gr.Dropdown(
                choices=list(CHUNK_LANGUAGE_CHOICES),
                value="text",
                label="Chunk Language",
            )
            chunker_options = gr.Textbox(
                value="",
                label="Chunker Options",
                placeholder="include_line_numbers=true",
            )
    return PreparationControls(
        selected,
        preprocess_group,
        preprocess_mode,
        chunk_group,
        chunking_method,
        chunk_size,
        chunk_overlap,
        max_chunks,
        chunk_aggregation,
        chunk_language,
        chunker_options,
    )


def build_scoring_controls():
    with gr.Accordion("Scoring setup", open=True):
        metric_preset = gr.Dropdown(
            choices=list(metric_preset_names()),
            value="Balanced",
            label="Metric Preset",
        )
        features = gr.CheckboxGroup(
            choices=FEATURE_UI_CHOICES,
            value=DEFAULT_FEATURE_SELECTION,
            label="Active Metrics",
        )
        model = build_model_routing_controls()
        with gr.Group(visible=True) as levenshtein_group:
            levenshtein_weight = gr.Slider(
                0,
                1,
                value=DEFAULT_UI_FEATURE_WEIGHTS["levenshtein"],
                label="Levenshtein Weight",
                step=0.05,
            )
            levenshtein_weights = gr.Textbox(value="1,1,1", label="Insert, Delete, Substitute")
        with gr.Group(visible=False) as jaro_group:
            jaro_winkler_weight = gr.Slider(
                0,
                1,
                value=DEFAULT_UI_FEATURE_WEIGHTS["jaro_winkler"],
                label="Jaro-Winkler Weight",
                step=0.05,
            )
            jaro_prefix_weight = gr.Slider(
                0.0,
                0.25,
                value=0.1,
                label="Prefix Weight",
                step=0.01,
            )
        with gr.Group(visible=False) as winnowing_group:
            winnowing_weight = gr.Slider(
                0,
                1,
                value=DEFAULT_UI_FEATURE_WEIGHTS["winnowing"],
                label="Winnowing Baseline Weight",
                step=0.05,
            )
            winnowing_kgram = gr.Slider(1, 32, value=5, label="k-gram Size", step=1)
            winnowing_window = gr.Slider(1, 32, value=4, label="Window Size", step=1)
        with gr.Group(visible=False) as gst_group:
            gst_weight = gr.Slider(
                0,
                1,
                value=DEFAULT_UI_FEATURE_WEIGHTS["gst"],
                label="GST Baseline Weight",
                step=0.05,
            )
            gst_min_match_length = gr.Slider(1, 32, value=5, label="Min Match Length", step=1)
        lexical_tokenizer = gr.Dropdown(
            choices=list(LEXICAL_TOKENIZER_CHOICES),
            value="raw",
            label="Lexical Tokenizer",
        )
        code = build_code_metric_controls()
    preparation = build_preparation_controls()
    return ScoringControls(
        metric_preset,
        features,
        model,
        levenshtein_group,
        levenshtein_weight,
        levenshtein_weights,
        jaro_group,
        jaro_winkler_weight,
        jaro_prefix_weight,
        winnowing_group,
        winnowing_weight,
        winnowing_kgram,
        winnowing_window,
        gst_group,
        gst_weight,
        gst_min_match_length,
        lexical_tokenizer,
        code,
        preparation,
    )


def wire_scoring_controls(controls):
    """Wire shared progressive-disclosure and model-routing behavior."""
    controls.features.change(
        update_feature_sections,
        inputs=controls.features,
        outputs=[
            controls.model.group,
            controls.levenshtein_group,
            controls.jaro_group,
            controls.winnowing_group,
            controls.gst_group,
            controls.code.group,
        ],
    )
    controls.metric_preset.change(
        apply_metric_preset_gradio,
        inputs=controls.metric_preset,
        outputs=[
            controls.features,
            controls.model.semantic_weight,
            controls.levenshtein_weight,
            controls.jaro_winkler_weight,
            controls.winnowing_weight,
            controls.gst_weight,
            controls.code.metric,
            controls.code.weight,
            controls.model.group,
            controls.levenshtein_group,
            controls.jaro_group,
            controls.winnowing_group,
            controls.gst_group,
            controls.code.group,
        ],
    )
    controls.preparation.selected.change(
        update_code_preparation_sections,
        inputs=controls.preparation.selected,
        outputs=[controls.preparation.preprocess_group, controls.preparation.chunk_group],
    )
    controls.code.metric.change(
        update_code_metric_sections,
        inputs=controls.code.metric,
        outputs=[
            controls.code.codebleu_group,
            controls.code.crystal_group,
            controls.code.ruby_group,
            controls.code.tsed_group,
            controls.code.codebertscore_group,
        ],
    )
    controls.code.codebertscore_model.change(
        sync_codebertscore_model_settings_gradio,
        inputs=[controls.code.codebertscore_model, controls.code.codebertscore_max_length],
        outputs=[controls.code.codebertscore_max_length],
    )
    for component in (
        controls.model.model,
        controls.model.vector_backend,
        controls.model.runtime_device,
    ):
        component.change(
            sync_model_settings_gradio,
            inputs=[
                controls.model.model,
                controls.model.vector_backend,
                controls.model.runtime_device,
                controls.model.similarity_function,
                controls.model.pooling_method,
                controls.model.max_token_length,
            ],
            outputs=[
                controls.model.vector_backend,
                controls.model.max_token_length,
                controls.model.status,
                controls.model.similarity_group,
                controls.model.pooling_group,
            ],
        )


with gr.Blocks(
    title="Matheel Framework",
    css=APP_CSS,
    js=APP_JS,
    fill_width=True,
) as demo:
    gr.HTML(value=app_header_html(), container=False, padding=False)
    with gr.Tabs(elem_id="matheel-workflows") as workflow_tabs:
        with gr.Tab("Compare"):
            gr.HTML(
                value=workflow_intro_html(
                    "Fast path",
                    "Compare two snippets",
                    "Paste code side by side, choose a scoring preset, and review the evidence.",
                    "A similarity score, interpretation, and active metric breakdown.",
                ),
                container=False,
                padding=False,
            )
            with gr.Row(elem_classes=["matheel-workflow-grid"]):
                with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                    with gr.Row():
                        pair_code1 = gr.Textbox(
                            label="Code A",
                            lines=16,
                            placeholder="First snippet",
                        )
                        pair_code2 = gr.Textbox(
                            label="Code B",
                            lines=16,
                            placeholder="Second snippet",
                        )
                    pair_run = gr.Button("Run Pair", variant="primary")
                    pair_output = gr.HTML(value=empty_pair_summary_html(), padding=False)

                with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                    pair_controls = build_scoring_controls()
            wire_scoring_controls(pair_controls)
            pair_run.click(
                calculate_similarity_gradio,
                inputs=[
                    pair_code1,
                    pair_code2,
                    *pair_controls.handler_inputs(),
                ],
                outputs=pair_output,
            )

        with gr.Tab("Collection"):
            gr.HTML(
                value=workflow_intro_html(
                    "Scale up",
                    "Rank similar files in a collection",
                    "Upload a source ZIP and apply one scoring configuration across its files.",
                    "A ranked pair table with run metadata and reproducible settings.",
                ),
                container=False,
                padding=False,
            )
            with gr.Row(elem_classes=["matheel-workflow-grid"]):
                with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                    collection_file = gr.File(label="Code ZIP", file_types=[".zip"])
                    collection_run = gr.Button("Run Collection", variant="primary")
                    collection_summary = gr.HTML(value=empty_summary_html(), padding=False)
                    collection_output = gr.Dataframe(
                        label="Ranked Pairs",
                        wrap=False,
                        interactive=False,
                        max_height=460,
                        row_count=1,
                        show_search="filter",
                        column_widths=["34%", "34%", "18%"],
                        elem_classes=["matheel-table"],
                    )

                with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                    collection_controls = build_scoring_controls()
                    with gr.Accordion("Result limits", open=False):
                        collection_threshold = gr.Slider(
                            0, 1, value=0.35, label="Threshold", step=0.01
                        )
                        collection_number_results = gr.Slider(
                            1, 1000, value=50, label="Max Pairs", step=1
                        )

            wire_scoring_controls(collection_controls)
            collection_run.click(
                get_sim_list_gradio,
                inputs=[
                    collection_file,
                    *collection_controls.handler_inputs(),
                    collection_threshold,
                    collection_number_results,
                ],
                outputs=[collection_summary, collection_output],
            )

        with gr.Tab("Suites"):
            suite_details_store = gr.State({})
            suite_runs_state = gr.State(empty_suite_rows())

            gr.HTML(
                value=workflow_intro_html(
                    "Compare configurations",
                    "Run a reproducible scoring suite",
                    "Save multiple scoring profiles, run them on one collection, and compare results.",
                    "A suite summary, per-run details, and downloadable run definitions.",
                ),
                container=False,
                padding=False,
            )
            with gr.Row(elem_classes=["matheel-workflow-grid"]):
                with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                    suite_file = gr.File(label="Code ZIP", file_types=[".zip"])
                    suite_summary = gr.HTML(value=empty_suite_summary_html(), padding=False)
                    suite_output = gr.Dataframe(
                        label="Suite Summary",
                        wrap=False,
                        interactive=False,
                        max_height=360,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    suite_run_name = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Run Details",
                    )
                    suite_details = gr.Dataframe(
                        label="Selected Run Details",
                        wrap=False,
                        interactive=False,
                        max_height=420,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    suite_runs_overview = gr.HTML(
                        value=suite_runs_overview_html(empty_suite_rows()),
                        padding=False,
                    )

                with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                    suite_controls = build_scoring_controls()
                    with gr.Group():
                        suite_run_name_input = gr.Textbox(
                            value=preview_suite_run_name(
                                empty_suite_rows(),
                                DEFAULT_FEATURE_SELECTION,
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

                    with gr.Accordion("Result limits", open=False):
                        suite_threshold = gr.Slider(0, 1, value=0.35, label="Threshold", step=0.01)
                        suite_number_results = gr.Slider(
                            1, 1000, value=50, label="Max Pairs", step=1
                        )

                    suite_run = gr.Button("Run Suite", variant="primary")
                    suite_output_format = gr.Dropdown(
                        choices=["csv", "json"],
                        value="csv",
                        label="Summary File Format",
                    )
                    suite_summary_download = gr.File(label="Summary Download")
                    suite_details_download = gr.File(label="Details ZIP")
                    suite_runs_download = gr.File(label="Run JSON")
                    suite_status = gr.HTML(
                        value=profile_status_html(
                            "No saved runs yet. Save a configuration or run the current one."
                        ),
                        padding=False,
                    )

            wire_scoring_controls(suite_controls)

            suite_add_run.click(
                append_suite_run_gradio,
                inputs=[
                    suite_runs_state,
                    suite_run_name_input,
                    *suite_controls.handler_inputs(),
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
                    *suite_controls.handler_inputs(),
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
                suite_controls.features,
                suite_controls.preparation.selected,
                suite_controls.model.model,
                suite_controls.model.vector_backend,
                suite_controls.model.similarity_function,
                suite_controls.model.pooling_method,
                suite_controls.code.metric,
                suite_controls.preparation.preprocess_mode,
                suite_controls.preparation.chunking_method,
            ):
                component.change(
                    preview_suite_run_name,
                    inputs=[
                        suite_runs_state,
                        suite_controls.features,
                        suite_controls.preparation.selected,
                        suite_controls.model.model,
                        suite_controls.model.vector_backend,
                        suite_controls.model.similarity_function,
                        suite_controls.model.pooling_method,
                        suite_controls.code.metric,
                        suite_controls.preparation.preprocess_mode,
                        suite_controls.preparation.chunking_method,
                    ],
                    outputs=suite_run_name_input,
                )
        with gr.Tab("Datasets", id="datasets"):
            dataset_scored_state = gr.State(None)
            dataset_validation_state = gr.State(None)
            gr.HTML(
                value=workflow_intro_html(
                    "Measure quality",
                    "Validate and evaluate normalized datasets",
                    "Check dataset structure before scoring, then tune thresholds or inspect scored pairs.",
                    "Validation evidence, evaluation metrics, scored rows, and report artifacts.",
                ),
                container=False,
                padding=False,
            )
            dataset_journey_status = gr.HTML(
                value=dataset_journey_status_html(
                    1,
                    "Upload a dataset, choose its task, then validate it.",
                ),
                padding=False,
            )
            with gr.Row(elem_classes=["matheel-workflow-grid"]):
                with gr.Column(
                    scale=5,
                    elem_id="dataset-setup-column",
                    elem_classes=["matheel-control-panel"],
                ):
                    with gr.Accordion("Dataset setup", open=True):
                        dataset_file = gr.File(
                            label="Normalized Dataset ZIP",
                            file_types=[".zip"],
                        )
                        dataset_task = gr.Radio(
                            choices=list(DATASET_TASK_CHOICES),
                            value=DEFAULT_DATASET_TASK,
                            label="Task",
                        )
                        dataset_validate = gr.Button(
                            "Validate Dataset",
                            variant="secondary",
                        )
                        dataset_validate_api = gr.Button(
                            "Dataset Validation API",
                            visible=False,
                        )
                        dataset_metric_preset = gr.Dropdown(
                            choices=list(metric_preset_names()),
                            value="Lexical Only",
                            label="Metric Preset",
                        )
                        dataset_model = HuggingfaceHubSearch(
                            value=DEFAULT_MODEL,
                            label="Embedding Model",
                            placeholder="Search Hugging Face models",
                            search_type="model",
                        )
                        dataset_vector_backend = gr.Dropdown(
                            choices=list(available_vector_backends()),
                            value="auto",
                            label="Vector Backend",
                        )
                        dataset_runtime_device = gr.Dropdown(
                            choices=list(DEVICE_CHOICES),
                            value="auto",
                            label="Runtime Device",
                        )
                        dataset_preprocess_mode = gr.Dropdown(
                            choices=list(available_preprocess_modes()),
                            value="none",
                            label="Preprocessing Mode",
                        )
                        dataset_code_language = gr.Dropdown(
                            choices=list(available_code_metric_languages()),
                            value="python",
                            label="Code Language",
                        )
                        dataset_lexical_tokenizer = gr.Dropdown(
                            choices=list(LEXICAL_TOKENIZER_CHOICES),
                            value="raw",
                            label="Lexical Tokenizer",
                        )
                    with gr.Accordion("Evaluation options", open=True):
                        with gr.Group(visible=True) as dataset_pair_group:
                            dataset_threshold = gr.Slider(
                                0,
                                1,
                                value=0.5,
                                label="Pair Threshold",
                                step=0.01,
                            )
                        with gr.Group(visible=False) as dataset_retrieval_group:
                            dataset_k = gr.Slider(
                                1,
                                100,
                                value=10,
                                label="Retrieval k",
                                step=1,
                            )
                        dataset_resampling_folds = gr.Slider(
                            0,
                            10,
                            value=0,
                            label="K-fold Resampling Folds (0 = off)",
                            step=1,
                        )
                        dataset_resampling_seed = gr.Number(
                            value=7,
                            precision=0,
                            label="Resampling Seed",
                        )
                        dataset_threshold_optimize = gr.Dropdown(
                            choices=list(THRESHOLD_OPTIMIZE_CHOICES),
                            value="f1",
                            label="Threshold Tuning Metric",
                            visible=False,
                        )
                        dataset_threshold_tune = gr.Button(
                            "Tune Pair Threshold",
                            variant="secondary",
                            visible=False,
                        )
                        dataset_run = gr.Button(
                            "Run Dataset Evaluation",
                            variant="primary",
                            interactive=False,
                        )
                        dataset_evaluate_api = gr.Button(
                            "Dataset Evaluation API",
                            visible=False,
                        )
                    with gr.Accordion(
                        "Scored Pair Explanation",
                        open=False,
                        visible=False,
                    ) as dataset_pair_explanation_actions:
                        dataset_pair_explain_row = gr.Number(
                            value=0,
                            precision=0,
                            label="Scored Row",
                        )
                        dataset_pair_explain_segment_mode = gr.Dropdown(
                            choices=list(available_pair_explanation_segment_modes()),
                            value="line",
                            label="Segment Mode",
                        )
                        dataset_pair_explain_high_threshold = gr.Slider(
                            0,
                            1,
                            value=0.85,
                            step=0.01,
                            label="High Similarity Threshold",
                        )
                        dataset_pair_explain_medium_threshold = gr.Slider(
                            0,
                            1,
                            value=0.6,
                            step=0.01,
                            label="Medium Similarity Threshold",
                        )
                        dataset_pair_explain_low_threshold = gr.Slider(
                            0,
                            1,
                            value=0.3,
                            step=0.01,
                            label="Low Similarity Threshold",
                        )
                        dataset_pair_explain_chunk_size = gr.Slider(
                            1,
                            50,
                            value=5,
                            step=1,
                            label="Chunk Lines",
                        )
                        dataset_pair_explain_run = gr.Button(
                            "Explain Scored Pair", variant="secondary"
                        )

                with gr.Column(
                    scale=8,
                    elem_id="dataset-results-column",
                    elem_classes=["matheel-results-panel"],
                ):
                    dataset_validation_summary = gr.HTML(
                        value=empty_dataset_validation_summary_html(),
                        padding=False,
                    )
                    dataset_validation_issues = gr.Dataframe(
                        label="Validation Issues",
                        wrap=False,
                        interactive=False,
                        max_height=220,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    dataset_validation_report = gr.HTML(value="", padding=False)
                    dataset_summary = gr.HTML(
                        value=summary_panel_html(
                            "Dataset Evaluation",
                            [("Status", "No dataset uploaded"), ("Artifacts", "None")],
                            variant="empty",
                        ),
                        padding=False,
                        visible=False,
                    )
                    dataset_metrics = gr.Dataframe(
                        label="Metrics",
                        wrap=False,
                        interactive=False,
                        max_height=280,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_scores = gr.Dataframe(
                        label="Scored Rows",
                        wrap=False,
                        interactive=False,
                        max_height=420,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_resampling_metrics = gr.Dataframe(
                        label="Resampling Metrics",
                        wrap=False,
                        interactive=False,
                        max_height=260,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_resampling_summary = gr.Dataframe(
                        label="Resampling Summary",
                        wrap=False,
                        interactive=False,
                        max_height=260,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_threshold_summary = gr.HTML(
                        value=empty_threshold_tuning_summary_html(),
                        padding=False,
                        visible=False,
                    )
                    dataset_threshold_sweep = gr.Dataframe(
                        label="Threshold Sweep",
                        wrap=False,
                        interactive=False,
                        max_height=300,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_threshold_report = gr.HTML(value="", padding=False, visible=False)
                    dataset_pair_explain_summary = gr.HTML(
                        value=empty_pair_explanation_summary_html(),
                        padding=False,
                        visible=False,
                    )
                    dataset_pair_explain_matches = gr.Dataframe(
                        label="Scored Pair Matches",
                        wrap=False,
                        interactive=False,
                        max_height=260,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                        visible=False,
                    )
                    dataset_pair_explain_report = gr.HTML(value="", padding=False, visible=False)
            with gr.Accordion(
                "Artifact Workspace",
                open=True,
                elem_id="dataset-artifact-workspace",
            ):
                gr.Markdown(
                    "Validation, evaluation, calibration, and explanation downloads stay together here."
                )
                dataset_validation_artifacts = gr.File(label="Validation Artifacts")
                dataset_artifacts = gr.File(label="Leaderboard Artifacts")
                dataset_threshold_artifacts = gr.File(label="Threshold Tuning Artifacts")
                dataset_pair_explain_artifacts = gr.File(label="Scored Pair Explanation Artifacts")
            with gr.Group(
                visible=False,
                elem_id="dataset-next-actions",
            ) as dataset_next_actions:
                gr.Markdown("Continue with this dataset and its generated evaluation artifacts.")
                with gr.Row():
                    dataset_continue_explain = gr.Button(
                        "Continue to Explain",
                        variant="secondary",
                    )
                    dataset_continue_reports = gr.Button(
                        "Open Evaluation in Reports",
                        variant="secondary",
                    )

            dataset_task.change(
                update_dataset_task_sections,
                inputs=dataset_task,
                outputs=[dataset_pair_group, dataset_retrieval_group],
            )
            dataset_validate.click(
                validate_dataset_journey_gradio,
                inputs=[dataset_file, dataset_task],
                outputs=[
                    dataset_validation_summary,
                    dataset_validation_issues,
                    dataset_validation_report,
                    dataset_validation_artifacts,
                    dataset_validation_state,
                    dataset_run,
                    dataset_journey_status,
                ],
                api_name=False,
            )
            dataset_run.click(
                evaluate_dataset_journey_gradio,
                inputs=[
                    dataset_validation_state,
                    dataset_file,
                    dataset_task,
                    dataset_metric_preset,
                    dataset_model,
                    dataset_vector_backend,
                    dataset_runtime_device,
                    dataset_preprocess_mode,
                    dataset_code_language,
                    dataset_lexical_tokenizer,
                    dataset_threshold,
                    dataset_k,
                    dataset_resampling_folds,
                    dataset_resampling_seed,
                ],
                outputs=[
                    dataset_summary,
                    dataset_metrics,
                    dataset_scores,
                    dataset_resampling_metrics,
                    dataset_resampling_summary,
                    dataset_artifacts,
                    dataset_scored_state,
                    dataset_journey_status,
                    dataset_threshold_optimize,
                    dataset_threshold_tune,
                    dataset_pair_explanation_actions,
                    dataset_next_actions,
                ],
                api_name=False,
            )
            dataset_validate_api.click(
                validate_dataset_gradio,
                inputs=[dataset_file, dataset_task],
                outputs=[
                    dataset_validation_summary,
                    dataset_validation_issues,
                    dataset_validation_report,
                    dataset_validation_artifacts,
                ],
                api_name="validate_dataset_gradio",
            )
            dataset_evaluate_api.click(
                evaluate_dataset_gradio_with_state,
                inputs=[
                    dataset_file,
                    dataset_task,
                    dataset_metric_preset,
                    dataset_model,
                    dataset_vector_backend,
                    dataset_runtime_device,
                    dataset_preprocess_mode,
                    dataset_code_language,
                    dataset_lexical_tokenizer,
                    dataset_threshold,
                    dataset_k,
                    dataset_resampling_folds,
                    dataset_resampling_seed,
                ],
                outputs=[
                    dataset_summary,
                    dataset_metrics,
                    dataset_scores,
                    dataset_resampling_metrics,
                    dataset_resampling_summary,
                    dataset_artifacts,
                ],
                api_name="evaluate_dataset_gradio_with_state",
            )
            dataset_threshold_tune.click(
                threshold_tuning_journey_gradio,
                inputs=[dataset_scored_state, dataset_threshold_optimize],
                outputs=[
                    dataset_threshold_summary,
                    dataset_threshold_sweep,
                    dataset_threshold_report,
                    dataset_threshold_artifacts,
                    dataset_journey_status,
                ],
            )
            dataset_pair_explain_run.click(
                explain_scored_pair_journey_gradio,
                inputs=[
                    dataset_scored_state,
                    dataset_pair_explain_row,
                    dataset_pair_explain_segment_mode,
                    dataset_pair_explain_high_threshold,
                    dataset_pair_explain_medium_threshold,
                    dataset_pair_explain_low_threshold,
                    dataset_pair_explain_chunk_size,
                ],
                outputs=[
                    dataset_pair_explain_summary,
                    dataset_pair_explain_matches,
                    dataset_pair_explain_report,
                    dataset_pair_explain_artifacts,
                    dataset_journey_status,
                ],
            )

            dataset_upload_reset_outputs = [
                dataset_validation_state,
                dataset_validation_summary,
                dataset_validation_issues,
                dataset_validation_report,
                dataset_validation_artifacts,
                dataset_run,
                dataset_journey_status,
                dataset_scored_state,
                dataset_summary,
                dataset_metrics,
                dataset_scores,
                dataset_resampling_metrics,
                dataset_resampling_summary,
                dataset_artifacts,
                dataset_threshold_summary,
                dataset_threshold_sweep,
                dataset_threshold_report,
                dataset_threshold_artifacts,
                dataset_pair_explain_summary,
                dataset_pair_explain_matches,
                dataset_pair_explain_report,
                dataset_pair_explain_artifacts,
                dataset_threshold_optimize,
                dataset_threshold_tune,
                dataset_pair_explanation_actions,
                dataset_next_actions,
            ]
            dataset_file.change(
                reset_dataset_upload_journey,
                outputs=dataset_upload_reset_outputs,
                show_api=False,
            )
            dataset_task.change(
                reset_dataset_upload_journey,
                outputs=dataset_upload_reset_outputs,
                show_api=False,
            )

            dataset_result_reset_outputs = [
                dataset_scored_state,
                dataset_summary,
                dataset_metrics,
                dataset_scores,
                dataset_resampling_metrics,
                dataset_resampling_summary,
                dataset_artifacts,
                dataset_threshold_summary,
                dataset_threshold_sweep,
                dataset_threshold_report,
                dataset_threshold_artifacts,
                dataset_pair_explain_summary,
                dataset_pair_explain_matches,
                dataset_pair_explain_report,
                dataset_pair_explain_artifacts,
                dataset_journey_status,
                dataset_threshold_optimize,
                dataset_threshold_tune,
                dataset_pair_explanation_actions,
                dataset_next_actions,
            ]
            for component in (
                dataset_metric_preset,
                dataset_model,
                dataset_vector_backend,
                dataset_runtime_device,
                dataset_preprocess_mode,
                dataset_code_language,
                dataset_lexical_tokenizer,
                dataset_threshold,
                dataset_k,
                dataset_resampling_folds,
                dataset_resampling_seed,
            ):
                component.input(
                    reset_dataset_results,
                    outputs=dataset_result_reset_outputs,
                    show_api=False,
                )

            dataset_threshold_optimize.input(
                reset_dataset_threshold_stage,
                outputs=[
                    dataset_threshold_summary,
                    dataset_threshold_sweep,
                    dataset_threshold_report,
                    dataset_threshold_artifacts,
                    dataset_journey_status,
                ],
                show_api=False,
            )
            for component in (
                dataset_pair_explain_row,
                dataset_pair_explain_segment_mode,
                dataset_pair_explain_high_threshold,
                dataset_pair_explain_medium_threshold,
                dataset_pair_explain_low_threshold,
                dataset_pair_explain_chunk_size,
            ):
                component.input(
                    reset_dataset_explanation_stage,
                    outputs=[
                        dataset_pair_explain_summary,
                        dataset_pair_explain_matches,
                        dataset_pair_explain_report,
                        dataset_pair_explain_artifacts,
                        dataset_journey_status,
                    ],
                    show_api=False,
                )

        with gr.Tab("Explain", id="explain"):
            gr.HTML(
                value=workflow_intro_html(
                    "Inspect evidence",
                    "Explain patterns and matched regions",
                    "Map a dataset or inspect where two snippets share similar regions.",
                    "Interactive visual evidence plus portable explanation artifacts.",
                ),
                container=False,
                padding=False,
            )
            with gr.Tabs(
                selected="dataset-map",
                elem_classes=["matheel-subtabs"],
            ) as explain_tabs:
                with gr.Tab("Dataset Map", id="dataset-map"):
                    with gr.Row(elem_classes=["matheel-workflow-grid"]):
                        with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                            map_dataset_file = gr.File(
                                label="Normalized Dataset ZIP",
                                file_types=[".zip"],
                            )
                            map_run = gr.Button("Generate Map", variant="primary")
                            map_summary = gr.HTML(
                                value=empty_dataset_map_summary_html(), padding=False
                            )
                            map_points = gr.Dataframe(
                                label="Projected Documents",
                                wrap=False,
                                interactive=False,
                                max_height=320,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            map_html = gr.HTML(value="", padding=False)
                            map_artifacts = gr.File(label="Dataset Map Artifacts")

                        with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                            with gr.Accordion("Map setup", open=True):
                                map_task = gr.Radio(
                                    choices=list(DATASET_TASK_CHOICES),
                                    value=DEFAULT_DATASET_TASK,
                                    label="Task",
                                )
                                map_projection_method = gr.Dropdown(
                                    choices=list(available_projection_methods()),
                                    value="pca",
                                    label="Projection Method",
                                )
                                map_seed = gr.Number(value=7, precision=0, label="Projection Seed")
                                map_static_vector_dim = gr.Slider(
                                    8,
                                    1024,
                                    value=256,
                                    step=8,
                                    label="Static Vector Dimension",
                                )
                                map_color_column = gr.Textbox(
                                    value="role",
                                    label="Color Column",
                                )

                    map_run.click(
                        generate_dataset_map_gradio,
                        inputs=[
                            map_dataset_file,
                            map_task,
                            map_projection_method,
                            map_seed,
                            map_static_vector_dim,
                            map_color_column,
                        ],
                        outputs=[map_summary, map_points, map_html, map_artifacts],
                    )

                with gr.Tab("Pair Explanation", id="pair-explanation"):
                    with gr.Row(elem_classes=["matheel-workflow-grid"]):
                        with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                            with gr.Row():
                                explain_left_code = gr.Textbox(
                                    label="Code A",
                                    lines=14,
                                    placeholder="First snippet",
                                )
                                explain_right_code = gr.Textbox(
                                    label="Code B",
                                    lines=14,
                                    placeholder="Second snippet",
                                )
                            explain_run = gr.Button("Generate Explanation", variant="primary")
                            explain_summary = gr.HTML(
                                value=empty_pair_explanation_summary_html(),
                                padding=False,
                            )
                            explain_matches = gr.Dataframe(
                                label="Matched Regions",
                                wrap=False,
                                interactive=False,
                                max_height=260,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            explain_html = gr.HTML(value="", padding=False)
                            explain_artifacts = gr.File(label="Pair Explanation Artifacts")

                        with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                            with gr.Accordion("Explanation setup", open=True):
                                explain_segment_mode = gr.Dropdown(
                                    choices=list(available_pair_explanation_segment_modes()),
                                    value="line",
                                    label="Segment Mode",
                                )
                                explain_high_threshold = gr.Slider(
                                    0,
                                    1,
                                    value=0.85,
                                    step=0.01,
                                    label="High Similarity Threshold",
                                )
                                explain_medium_threshold = gr.Slider(
                                    0,
                                    1,
                                    value=0.6,
                                    step=0.01,
                                    label="Medium Similarity Threshold",
                                )
                                explain_low_threshold = gr.Slider(
                                    0,
                                    1,
                                    value=0.3,
                                    step=0.01,
                                    label="Low Similarity Threshold",
                                )
                                explain_chunk_size = gr.Slider(
                                    1,
                                    50,
                                    value=5,
                                    step=1,
                                    label="Chunk Lines",
                                )

                    explain_run.click(
                        generate_pair_explanation_gradio,
                        inputs=[
                            explain_left_code,
                            explain_right_code,
                            explain_segment_mode,
                            explain_high_threshold,
                            explain_medium_threshold,
                            explain_low_threshold,
                            explain_chunk_size,
                        ],
                        outputs=[explain_summary, explain_matches, explain_html, explain_artifacts],
                    )

        with gr.Tab("Reports", id="reports"):
            gr.HTML(
                value=workflow_intro_html(
                    "Share results",
                    "Build leaderboards and inspect report bundles",
                    "Compare algorithm presets across datasets or reopen a generated report artifact.",
                    "Ranked algorithms, per-dataset evidence, and downloadable report bundles.",
                ),
                container=False,
                padding=False,
            )
            with gr.Tabs(
                selected="ready-made-leaderboard",
                elem_classes=["matheel-subtabs"],
            ) as reports_tabs:
                with gr.Tab("Ready-made Leaderboard", id="ready-made-leaderboard"):
                    gr.Markdown(
                        "Scores are precomputed across every registered public dataset preset and "
                        "every built-in algorithm preset. The snapshot uses deterministic samples "
                        "and the offline `static_hash` vector backend; use Build Leaderboard for a "
                        "full-corpus or custom run."
                    )
                    ready_made_summary = gr.HTML(value=ready_made_initial[0], padding=False)
                    with gr.Accordion("Explore rankings", open=True):
                        gr.Markdown(
                            "Pair classification defaults to **F1**; retrieval defaults to "
                            "**Mean Average Precision**. Both rankings start in descending score "
                            "order. Use the controls below to filter the snapshot, the table filter "
                            "to search any visible column, and the explicit sort controls to reorder "
                            "either table."
                        )
                        with gr.Row():
                            ready_made_task = gr.Dropdown(
                                choices=list(READY_LEADERBOARD_TASK_CHOICES),
                                value=ready_made_filter_initial["task"],
                                label="Task",
                            )
                            ready_made_metric = gr.Dropdown(
                                choices=ready_made_filter_initial["metrics"],
                                value=ready_made_filter_initial["metric"],
                                label="Metric (higher is better)",
                            )
                            ready_made_dataset = gr.Dropdown(
                                choices=ready_made_filter_initial["datasets"],
                                value=READY_LEADERBOARD_ALL_DATASETS,
                                label="Dataset",
                            )
                        with gr.Row():
                            ready_made_algorithm = gr.Dropdown(
                                choices=ready_made_filter_initial["algorithms"],
                                value=READY_LEADERBOARD_ALL_ALGORITHMS,
                                label="Algorithm",
                            )
                            ready_made_source = gr.Dropdown(
                                choices=ready_made_filter_initial["sources"],
                                value=READY_LEADERBOARD_ALL_SOURCES,
                                label="Dataset Source",
                            )
                            ready_made_reset = gr.Button("Reset Leaderboard Filters")
                        with gr.Row():
                            ready_made_aggregate_sort = gr.Dropdown(
                                choices=[
                                    ("Mean score", "mean_score"),
                                    ("Median score", "median_score"),
                                    ("Algorithm", "algorithm_name"),
                                    ("Dataset count", "dataset_count"),
                                    ("Sample count", "sample_count"),
                                    ("Rank", "rank"),
                                ],
                                value="mean_score",
                                label="Sort Aggregate By",
                            )
                            ready_made_aggregate_direction = gr.Dropdown(
                                choices=list(READY_LEADERBOARD_SORT_DIRECTIONS),
                                value=READY_LEADERBOARD_SORT_DIRECTIONS[0],
                                label="Aggregate Direction",
                            )
                            ready_made_per_dataset_sort = gr.Dropdown(
                                choices=[
                                    ("Score", "score"),
                                    ("Dataset", "dataset_name"),
                                    ("Algorithm", "algorithm_name"),
                                    ("Dataset source", "dataset_source"),
                                    ("Sample count", "sample_count"),
                                    ("Rank", "rank"),
                                ],
                                value="score",
                                label="Sort Per-Dataset By",
                            )
                            ready_made_per_dataset_direction = gr.Dropdown(
                                choices=list(READY_LEADERBOARD_SORT_DIRECTIONS),
                                value=READY_LEADERBOARD_SORT_DIRECTIONS[0],
                                label="Per-Dataset Direction",
                            )
                    ready_made_aggregate = gr.Dataframe(
                        value=ready_made_initial[1],
                        label="Interactive Aggregate Ranking",
                        wrap=False,
                        interactive=False,
                        max_height=360,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    ready_made_per_dataset = gr.Dataframe(
                        value=ready_made_initial[2],
                        label="Interactive Per-Dataset Ranking",
                        wrap=False,
                        interactive=False,
                        max_height=420,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    ready_made_coverage = gr.Dataframe(
                        value=ready_made_initial[3],
                        label="Dataset Coverage",
                        wrap=False,
                        interactive=False,
                        max_height=280,
                        row_count=1,
                        show_search="filter",
                        elem_classes=["matheel-table"],
                    )
                    with gr.Accordion("Static Report", open=False):
                        ready_made_report = gr.HTML(value=ready_made_initial[4], padding=False)
                    ready_made_artifact = gr.File(
                        value=ready_made_initial[5],
                        label="Ready-made Leaderboard JSON",
                        interactive=False,
                    )

                    ready_made_task.input(
                        change_ready_made_leaderboard_task,
                        inputs=[
                            ready_made_task,
                            ready_made_aggregate_sort,
                            ready_made_aggregate_direction,
                            ready_made_per_dataset_sort,
                            ready_made_per_dataset_direction,
                        ],
                        outputs=[
                            ready_made_metric,
                            ready_made_dataset,
                            ready_made_algorithm,
                            ready_made_source,
                            ready_made_aggregate,
                            ready_made_per_dataset,
                        ],
                    )
                    for ready_made_filter in (
                        ready_made_metric,
                        ready_made_dataset,
                        ready_made_algorithm,
                        ready_made_source,
                        ready_made_aggregate_sort,
                        ready_made_aggregate_direction,
                        ready_made_per_dataset_sort,
                        ready_made_per_dataset_direction,
                    ):
                        ready_made_filter.input(
                            filter_ready_made_leaderboard,
                            inputs=[
                                ready_made_task,
                                ready_made_metric,
                                ready_made_dataset,
                                ready_made_algorithm,
                                ready_made_source,
                                ready_made_aggregate_sort,
                                ready_made_aggregate_direction,
                                ready_made_per_dataset_sort,
                                ready_made_per_dataset_direction,
                            ],
                            outputs=[ready_made_aggregate, ready_made_per_dataset],
                        )
                    ready_made_reset.click(
                        reset_ready_made_leaderboard_filters,
                        outputs=[
                            ready_made_task,
                            ready_made_metric,
                            ready_made_dataset,
                            ready_made_algorithm,
                            ready_made_source,
                            ready_made_aggregate_sort,
                            ready_made_aggregate_direction,
                            ready_made_per_dataset_sort,
                            ready_made_per_dataset_direction,
                            ready_made_aggregate,
                            ready_made_per_dataset,
                        ],
                    )

                with gr.Tab("Build Leaderboard", id="build-leaderboard"):
                    with gr.Row(elem_classes=["matheel-workflow-grid"]):
                        with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                            ready_dataset_files = gr.File(
                                label="Normalized Dataset ZIPs",
                                file_types=[".zip"],
                                file_count="multiple",
                            )
                            ready_run = gr.Button("Run Custom Leaderboard", variant="primary")
                            ready_summary = gr.HTML(
                                value=empty_ready_leaderboard_summary_html(),
                                padding=False,
                            )
                            ready_aggregate = gr.Dataframe(
                                label="Ranked Algorithms",
                                wrap=False,
                                interactive=False,
                                max_height=320,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            ready_per_dataset = gr.Dataframe(
                                label="Per-Dataset Ranking",
                                wrap=False,
                                interactive=False,
                                max_height=360,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            ready_report = gr.HTML(value="", padding=False)
                            ready_artifacts = gr.File(label="Ready Leaderboard Artifacts")
                            ready_registered_datasets = gr.Dataframe(
                                value=ready_leaderboard_registered_datasets_frame(),
                                label="Registered Datasets",
                                wrap=False,
                                interactive=False,
                                max_height=280,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )

                        with gr.Column(scale=5, elem_classes=["matheel-control-panel"]):
                            with gr.Accordion("Algorithms", open=True):
                                ready_algorithms = gr.CheckboxGroup(
                                    choices=list(READY_LEADERBOARD_ALGORITHM_CHOICES),
                                    value=list(READY_LEADERBOARD_ALGORITHM_CHOICES),
                                    label="Algorithm Presets",
                                )
                                ready_model = HuggingfaceHubSearch(
                                    value=DEFAULT_MODEL,
                                    label="Embedding Model",
                                    placeholder="Search Hugging Face models",
                                    search_type="model",
                                )
                                ready_vector_backend = gr.Dropdown(
                                    choices=list(available_vector_backends()),
                                    value="auto",
                                    label="Vector Backend",
                                )
                                ready_runtime_device = gr.Dropdown(
                                    choices=list(DEVICE_CHOICES),
                                    value="auto",
                                    label="Runtime Device",
                                )
                                ready_preprocess_mode = gr.Dropdown(
                                    choices=list(available_preprocess_modes()),
                                    value="none",
                                    label="Preprocessing Mode",
                                )
                                ready_code_language = gr.Dropdown(
                                    choices=list(available_code_metric_languages()),
                                    value="python",
                                    label="Code Language",
                                )
                                ready_lexical_tokenizer = gr.Dropdown(
                                    choices=list(LEXICAL_TOKENIZER_CHOICES),
                                    value="raw",
                                    label="Lexical Tokenizer",
                                )

                            with gr.Accordion("Task defaults", open=True):
                                ready_pair_threshold = gr.Slider(
                                    0,
                                    1,
                                    value=0.5,
                                    label="Pair Threshold",
                                    step=0.01,
                                )
                                ready_retrieval_k = gr.Slider(
                                    1,
                                    100,
                                    value=10,
                                    label="Retrieval k",
                                    step=1,
                                )
                                ready_seed = gr.Number(value=7, precision=0, label="Seed")

                    ready_run.click(
                        run_ready_leaderboard_gradio,
                        inputs=[
                            ready_dataset_files,
                            ready_algorithms,
                            ready_model,
                            ready_vector_backend,
                            ready_runtime_device,
                            ready_preprocess_mode,
                            ready_code_language,
                            ready_lexical_tokenizer,
                            ready_pair_threshold,
                            ready_retrieval_k,
                            ready_seed,
                        ],
                        outputs=[
                            ready_summary,
                            ready_aggregate,
                            ready_per_dataset,
                            ready_report,
                            ready_artifacts,
                        ],
                    )

                with gr.Tab("Inspect Artifacts", id="inspect-artifacts"):
                    with gr.Row(elem_classes=["matheel-workflow-grid"]):
                        with gr.Column(scale=8, elem_classes=["matheel-results-panel"]):
                            leaderboard_file = gr.File(
                                label="Leaderboard JSON or ZIP",
                                file_types=[".json", ".zip"],
                            )
                            leaderboard_inspect = gr.Button(
                                "Inspect Leaderboard", variant="primary"
                            )
                            leaderboard_summary = gr.HTML(
                                value=empty_leaderboard_inspection_summary_html(),
                                padding=False,
                            )
                            leaderboard_aggregate = gr.Dataframe(
                                label="Aggregate Ranking",
                                wrap=False,
                                interactive=False,
                                max_height=320,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            leaderboard_per_dataset = gr.Dataframe(
                                label="Per-Dataset Ranking",
                                wrap=False,
                                interactive=False,
                                max_height=360,
                                row_count=1,
                                show_search="filter",
                                elem_classes=["matheel-table"],
                            )
                            leaderboard_report = gr.HTML(value="", padding=False)
                            leaderboard_artifacts = gr.File(label="Leaderboard Report Artifacts")

                    leaderboard_inspect.click(
                        inspect_leaderboard_artifacts_gradio,
                        inputs=leaderboard_file,
                        outputs=[
                            leaderboard_summary,
                            leaderboard_aggregate,
                            leaderboard_per_dataset,
                            leaderboard_report,
                            leaderboard_artifacts,
                        ],
                    )

    dataset_continue_explain.click(
        handoff_dataset_to_explain,
        inputs=[dataset_file, dataset_task],
        outputs=[map_dataset_file, map_task, workflow_tabs, explain_tabs],
        api_name=False,
    )
    dataset_continue_reports.click(
        handoff_dataset_to_reports,
        inputs=dataset_artifacts,
        outputs=[leaderboard_file, workflow_tabs, reports_tabs],
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(show_error=True, debug=True)
