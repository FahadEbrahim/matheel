"""Typed clone and AI-authorship benchmark views."""

import hashlib
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

try:
    from gradio_app.presentation import summary_panel_html
except ModuleNotFoundError:
    from presentation import summary_panel_html


RELEASE_RESULTS = Path(__file__).with_name("benchmark_data") / "research-080" / "summary.json"


def _upload_path(value, name, maximum_bytes):
    if not value:
        raise ValueError(f"Upload the {name} CSV.")
    path = Path(value)
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"Use the CLI for {name} CSVs larger than {maximum_bytes // 1_000_000} MB.")
    return path


def published_research_rows(path=RELEASE_RESULTS):
    if not Path(path).is_file():
        return pd.DataFrame(), None
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(evidence["rows"]), str(path)


def similcode_slices_gradio(scores):
    from matheel.similarity_datasets import similcode_score_slices

    try:
        scores_path = _upload_path(scores, "SimilCode scores", 2_000_000)
        slices = similcode_score_slices(pd.read_csv(scores_path))
        return "", slices
    except (OSError, TypeError, ValueError) as exc:
        return summary_panel_html("SimilCode score slices", [("Details", str(exc))], variant="error"), pd.DataFrame()


def evaluate_clones_gradio(references, findings, coverage):
    from matheel.clone_discovery import (
        evaluate_clone_discovery, load_clone_findings, load_clone_references,
    )
    from matheel.external_detectors.common import file_sha256

    try:
        reference_path = _upload_path(references, "reference", 25_000_000)
        finding_path = _upload_path(findings, "findings", 25_000_000)
        report = evaluate_clone_discovery(
            load_clone_references(reference_path), load_clone_findings(finding_path),
            coverage_threshold=float(coverage),
        )
        report["input_sha256"] = {
            "references": file_sha256(reference_path), "findings": file_sha256(finding_path),
        }
        output = Path(tempfile.mkdtemp(prefix="matheel-clone-recall-")) / "clone_recall.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        overall = report["overall"]
        summary = summary_panel_html("Clone discovery recall", [
            ("Matched references", f"{overall['matched']} / {overall['eligible']}"),
            ("Detector findings", str(report["reported_pairs"])),
            ("Coverage threshold", f"{float(coverage):.0%}"),
        ])
        return summary, pd.DataFrame(report["strata"]), str(output)
    except (OSError, TypeError, ValueError) as exc:
        return summary_panel_html("Clone discovery recall", [("Details", str(exc))], variant="error"), pd.DataFrame(), None


def benchmark_authorship_gradio(human, generated, target_fpr):
    from matheel.authorship import benchmark_authorship, prepare_aigcodeset, write_authorship_report

    try:
        human_path = _upload_path(human, "human", 10_000_000)
        generated_path = _upload_path(generated, "generated", 10_000_000)
        dataset = prepare_aigcodeset(human_path, generated_path, verify_source=True)
        report = benchmark_authorship(dataset, target_fpr=float(target_fpr))
        output = Path(tempfile.mkdtemp(prefix="matheel-authorship-"))
        write_authorship_report(report, output)
        summary = summary_panel_html("AI-authorship evaluation", [
            ("Prepared samples", str(len(dataset.samples))),
            ("Target validation false-positive rate", f"{float(target_fpr):.1%}"),
            ("Human CSV SHA256", hashlib.sha256(human_path.read_bytes()).hexdigest()),
            ("Generated CSV SHA256", hashlib.sha256(generated_path.read_bytes()).hexdigest()),
        ])
        return summary, pd.json_normalize(report["results"]), [str(path) for path in sorted(output.iterdir())]
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return summary_panel_html("AI-authorship evaluation", [("Details", str(exc))], variant="error"), pd.DataFrame(), None


def build_research_benchmarks_tab():
    with gr.Tab("Research Benchmarks", id="research-benchmarks"):
        published, published_file = published_research_rows()
        gr.Dataframe(
            label="Published Benchmark Results", value=published, interactive=False,
            show_search="filter", wrap=True, max_height=320,
        )
        if published_file:
            gr.File(label="Release Result Index", value=published_file, interactive=False)

        with gr.Accordion("Inspect SimilCode score slices", open=False):
            scores = gr.File(label="SimilCode Scores CSV", file_types=[".csv"], type="filepath")
            slice_button = gr.Button("Summarize Score Slices", variant="primary")
            slice_status = gr.HTML()
            slice_table = gr.Dataframe(label="Scores by Language, Category, and Clone Type", interactive=False,
                                       max_height=360)
            slice_button.click(similcode_slices_gradio, inputs=scores, outputs=[slice_status, slice_table])

        with gr.Accordion("Evaluate clone discovery", open=False):
            gr.Markdown("Upload reference and detected span pairs as CSVs to measure recall by clone type.")
            with gr.Row():
                references = gr.File(label="Reference Clones CSV", file_types=[".csv"], type="filepath")
                findings = gr.File(label="Detector Findings CSV", file_types=[".csv"], type="filepath")
            coverage = gr.Slider(0.05, 1, value=0.70, step=0.05, label="Required Span Coverage")
            recall_button = gr.Button("Evaluate Clone Recall", variant="primary")
            recall_summary = gr.HTML()
            recall_table = gr.Dataframe(label="Recall by Type and Project Scope", interactive=False, max_height=360)
            recall_artifact = gr.File(label="Clone Recall JSON", interactive=False)
            recall_button.click(evaluate_clones_gradio, inputs=[references, findings, coverage],
                                outputs=[recall_summary, recall_table, recall_artifact], concurrency_limit=1)

        with gr.Accordion("Evaluate AI authorship", open=False):
            gr.Markdown("Use the two pinned AIGCodeSet source tables to reproduce the snippet-classification benchmark.")
            with gr.Row():
                human = gr.File(label="AIGCodeSet Human CSV", file_types=[".csv"], type="filepath")
                generated = gr.File(label="AIGCodeSet Generated CSV", file_types=[".csv"], type="filepath")
            target_fpr = gr.Slider(0, 0.2, value=0.05, step=0.01, label="Target Validation False-Positive Rate")
            authorship_button = gr.Button("Run AI-Authorship Benchmark", variant="primary")
            authorship_summary = gr.HTML()
            authorship_table = gr.Dataframe(label="Authorship Metrics", interactive=False, max_height=360)
            authorship_artifacts = gr.File(label="Authorship Artifacts", file_count="multiple", interactive=False)
            authorship_button.click(benchmark_authorship_gradio, inputs=[human, generated, target_fpr],
                                    outputs=[authorship_summary, authorship_table, authorship_artifacts], concurrency_limit=1)
