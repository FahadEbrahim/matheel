"""Result inspection UI over uploaded source-free scored rows."""

import hashlib
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from matheel.result_analysis import analyze_scored_results, write_result_analysis

try:
    from gradio_app.presentation import summary_panel_html
except ModuleNotFoundError:
    from presentation import summary_panel_html


def analyze_results_gradio(scores, reference, task, threshold, k, group_column, rounds):
    empty = pd.DataFrame()
    try:
        if not scores:
            raise ValueError("Upload scored result CSVs before analyzing.")
        for path in (scores, reference):
            if path and Path(path).stat().st_size > 50_000_000:
                raise ValueError("Use the CLI for score files larger than 50 MB.")
            if path and len(pd.read_csv(path, nrows=100001)) > 100000:
                raise ValueError("Use the CLI for score files with more than 100,000 rows.")
        report = analyze_scored_results(
            scores, reference_rows=reference, task=task, threshold=float(threshold),
            k=int(k), group_column=str(group_column or "").strip() or None, rounds=int(rounds),
        )
        artifacts = write_result_analysis(report, tempfile.mkdtemp(prefix="matheel-result-analysis-"))
        summary = summary_panel_html("Result analysis", [
            ("Task", report["task_family"]), ("Rows", str(report["rows"])),
            ("Grouping", report["grouping"]), ("Groups", str(report["group_count"])),
            ("Threshold", str(report["threshold"])), ("Retrieval k", str(report["k"])),
            ("Score CSV SHA256", hashlib.sha256(Path(scores).read_bytes()).hexdigest()),
            ("Uncertainty", "95% conditional grouped bootstrap; unavailable with insufficient groups"),
        ])
        return (summary, pd.DataFrame(report["metrics"]).round(6), pd.DataFrame(report["errors"]).head(1000).round(6),
                pd.DataFrame(report["slices"]).round(6), pd.DataFrame(report["per_query"]).round(6),
                [str(path) for path in artifacts.values()])
    except (OSError, ValueError, TypeError) as exc:
        return (summary_panel_html("Result analysis", [("Details", str(exc))], variant="error"),
                empty, empty, empty, empty, None)


def build_result_analysis_tab():
    with gr.Tab("Analyze Results", id="analyze-results"):
        gr.Markdown(
            "Inspect classification errors, retrieval misses and method disagreements from scored CSVs. "
            "Use complete candidate results. Pair uncertainty groups shared source IDs; retrieval "
            "uncertainty resamples queries. Thresholds are supplied explicitly and are never fitted here."
        )
        with gr.Row():
            scores = gr.File(label="Scored Results CSV", file_types=[".csv"], type="filepath")
            reference = gr.File(label="Reference Method CSV (optional)", file_types=[".csv"], type="filepath")
        with gr.Row():
            task = gr.Dropdown(["auto", "pair", "retrieval"], value="auto", label="Analysis Task")
            threshold = gr.Number(value=.5, label="Analysis Threshold")
            k = gr.Number(value=10, minimum=1, precision=0, label="Analysis Retrieval k")
            group = gr.Textbox(label="Pair Group Column (optional)", placeholder="assignment")
            rounds = gr.Slider(25, 1000, value=200, step=25, label="Bootstrap Rounds")
        button = gr.Button("Analyze Scored Results", variant="primary")
        summary = gr.HTML()
        metrics = gr.Dataframe(label="Metrics and Paired Uncertainty", interactive=False, wrap=False,
                               show_search="filter", max_height=320)
        errors = gr.Dataframe(label="Errors and Disagreements (first 1000 rows)", interactive=False,
                              wrap=False, show_search="filter", max_height=420)
        slices = gr.Dataframe(label="Available Dataset Slices", interactive=False, wrap=False,
                              show_search="filter", max_height=300)
        queries = gr.Dataframe(label="Per-Query Metrics", interactive=False, wrap=False,
                               show_search="filter", max_height=300)
        downloads = gr.File(label="Analysis Artifacts", file_count="multiple", interactive=False)
        gr.Markdown(
            "Filter `outcome` for false positives or false negatives, or `decision_disagreement` "
            "for comparisons. Download complete CSVs and the HTML report. To inspect source regions, "
            "use the Explain tab with your local submissions, or run `matheel explain-pair` with "
            "the original scores and the displayed `score_row_index`. Explanations are local matching "
            "aids and do not attribute a detector's internal reasoning."
        )
        inputs = [scores, reference, task, threshold, k, group, rounds]
        controls = inputs + [button]
        outputs = [summary, metrics, errors, slices, queries, downloads]

        def run(*values):
            return (*analyze_results_gradio(*values),
                    *[gr.update(interactive=True) for _ in controls])

        # Inputs remain fixed during an analysis so its result cannot overwrite
        # a newer input selection. User changes after completion clear the view.
        button.click(lambda: [gr.update(interactive=False) for _ in controls],
                     outputs=controls, queue=False, api_name=False).then(
            run, inputs=inputs, outputs=outputs + controls,
            concurrency_limit=1, trigger_mode="once", api_name=False,
        )
        gr.on([scores.upload, scores.clear, reference.upload, reference.clear,
               task.input, threshold.input, k.input, group.input, rounds.input],
              lambda: ("", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None),
              outputs=outputs, queue=False, api_name=False)
