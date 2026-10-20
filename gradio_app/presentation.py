"""Escaped HTML presentation helpers shared by Gradio workflows."""

try:
    from gradio_app.html_utils import escape_html
except ModuleNotFoundError:
    from html_utils import escape_html


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
