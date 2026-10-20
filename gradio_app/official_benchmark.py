"""Read-only views over validated, published benchmark artifacts."""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from matheel.benchmark_protocol import DEFAULT_BENCHMARK_ID, validate_benchmark_result
from matheel.benchmark_analysis import load_benchmark_analysis
from matheel.leaderboard import available_leaderboard_metrics

try:
    from gradio_app.presentation import summary_panel_html
except ModuleNotFoundError:
    from presentation import summary_panel_html


OFFICIAL_BENCHMARK_PATH = (
    Path(__file__).resolve().parent
    / "benchmark_data"
    / "plagiarism-v1"
    / "matheel_plagiarism_v1.json"
)
_BUNDLE_ROOT = Path(__file__).resolve().parent / "benchmark_data"
PUBLISHED_BENCHMARK_EDITIONS = {
    "v1": ("0.7.0 · original v1 · 1,800 s budget", _BUNDLE_ROOT / "plagiarism-v1" / "matheel_plagiarism_v1.json"),
    "071-all": ("0.7.1 · all methods · 180 s budget", _BUNDLE_ROOT / "plagiarism-071" / "all-methods" / "matheel_plagiarism_v1.json"),
    "071-java": ("0.7.1 · Java detector comparison", _BUNDLE_ROOT / "plagiarism-071" / "detectors-java" / "matheel_plagiarism_v1.json"),
    "071-cpp": ("0.7.1 · C++ detector comparison", _BUNDLE_ROOT / "plagiarism-071" / "detectors-cpp" / "matheel_plagiarism_v1.json"),
    "071-python": ("0.7.1 · Python detector comparison", _BUNDLE_ROOT / "plagiarism-071" / "detectors-python" / "matheel_plagiarism_v1.json"),
}
ALL_OFFICIAL_METHODS = "all"
TASK_LABELS = {"pair": "Pair classification", "retrieval": "Retrieval"}
DEFAULT_METRICS = {"pair": "f1", "retrieval": "mean_average_precision"}
SCOPE_COLUMNS = [
    "Dataset",
    "Category",
    "Evaluation Task",
    "Counts",
    "Selection",
    "Split",
    "Source Revision",
    "Source",
    "Content SHA256",
]


def published_benchmark_choices():
    """Expose only installed bundles; callers cannot select arbitrary paths."""
    return [(label, identity) for identity, (label, path) in PUBLISHED_BENCHMARK_EDITIONS.items()
            if path.is_file()]


def update_published_benchmark_gradio(edition, task_family, metric, method_id):
    if edition not in PUBLISHED_BENCHMARK_EDITIONS:
        raise ValueError("Select a published benchmark edition.")
    _, artifact = PUBLISHED_BENCHMARK_EDITIONS[edition]
    if edition == "v1":
        artifact = OFFICIAL_BENCHMARK_PATH
    return update_official_benchmark_gradio(task_family, metric, method_id, artifact)


def load_official_benchmark_payload(artifact_path=None):
    """Validate portable protocol/results without opening original datasets."""
    path = Path(artifact_path or OFFICIAL_BENCHMARK_PATH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Official benchmark artifact must contain a JSON object.")
    validation = validate_benchmark_result(payload)
    if validation["benchmark_id"] != DEFAULT_BENCHMARK_ID:
        raise ValueError(f"Expected the {DEFAULT_BENCHMARK_ID} benchmark artifact.")
    return payload


def _empty_view(message, *, error=False):
    return {
        "available": False,
        "summary": summary_panel_html(
            "Official Plagiarism Benchmark",
            [
                ("Benchmark", DEFAULT_BENCHMARK_ID),
                (
                    "Status",
                    "Artifact validation failed" if error else "Results pending publication",
                ),
                ("Details", message),
            ],
            variant="error" if error else "empty",
        ),
        "task": "pair",
        "tasks": [(TASK_LABELS["pair"], "pair")],
        "metric": "f1",
        "metrics": ["f1"],
        "method": ALL_OFFICIAL_METHODS,
        "methods": [("All methods", ALL_OFFICIAL_METHODS)],
        "aggregate": pd.DataFrame(),
        "per_dataset": pd.DataFrame(),
        "scope": pd.DataFrame(columns=SCOPE_COLUMNS),
        "protocol": {},
        "configurations": {},
        "artifact": None,
    }


def _dataset_scope(protocol):
    rows = []
    for entry in protocol["datasets"]:
        content = entry.get("content") or {}
        selection = entry.get("selection") or {}
        rows.append(
            {
                "Dataset": entry["name"],
                "Category": entry["domain"].replace("_", " ").title(),
                "Evaluation Task": TASK_LABELS[entry["task_family"]],
                "Counts": "; ".join(
                    f"{name}: {value}" for name, value in sorted(content.get("counts", {}).items())
                ),
                "Selection": selection.get("description")
                or selection.get("mode", "Frozen protocol selection"),
                "Split": entry.get("split", ""),
                "Source Revision": entry.get("source_revision", ""),
                "Source": entry.get("source_url", ""),
                "Content SHA256": content.get("sha256", ""),
            }
        )
    return pd.DataFrame(rows, columns=SCOPE_COLUMNS)


def _metric_choices(rows, task):
    present = set(rows.loc[rows["task_family"] == task, "metric"])
    ordered = [metric for metric in available_leaderboard_metrics(task) if metric in present]
    return ordered + sorted(present.difference(ordered))


def _add_diagnostics(aggregate, per_dataset, artifact_path):
    """Add presentation columns; leave canonical scores, ranks and flags intact."""
    for frame in (aggregate, per_dataset):
        if "status" not in frame:
            frame["status"] = frame["eligible"].map({True: "completed", False: "failed"})
        frame["availability"] = frame["eligible"].map({True: "Available", False: "N/A"})
        if "error" in frame:
            frame.loc[frame["error"].eq("TimeoutError"), "status"] = "timed_out"
    aggregate["pareto_efficient"] = False
    aggregate["failure_reason"] = ""
    for identity, rows in per_dataset.groupby("algorithm_id"):
        failed = rows[~rows["eligible"]]
        reasons = []
        for column in ("not_applicable_reason", "error"):
            if column in failed:
                reasons.extend(str(value) for value in failed[column].dropna().unique() if value)
        mask = aggregate["algorithm_id"].eq(identity)
        aggregate.loc[mask, "failure_reason"] = "; ".join(sorted(set(reasons)))
        if "timed_out" in set(failed["status"]):
            aggregate.loc[mask, "status"] = "timed_out"
    valid = aggregate[aggregate["eligible"] & aggregate["elapsed_seconds"].gt(0)]
    for index, row in valid.iterrows():
        dominated = ((valid["mean_score"] >= row["mean_score"]) &
                     (valid["elapsed_seconds"] <= row["elapsed_seconds"]) &
                     ((valid["mean_score"] > row["mean_score"]) |
                      (valid["elapsed_seconds"] < row["elapsed_seconds"]))).any()
        aggregate.loc[index, "pareto_efficient"] = not dominated
    per_dataset["ci_lower"] = None
    per_dataset["ci_upper"] = None
    per_dataset["uncertainty_status"] = "N/A: no matching analysis companion"
    try:
        companion = load_benchmark_analysis(artifact_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        per_dataset["uncertainty_status"] = f"N/A: analysis validation failed ({exc})"
        companion = None
    if companion:
        by_key = {(row["dataset_name"], row["algorithm_id"], row["metric"]): row
                  for row in companion["metrics"]}
        for index, row in per_dataset.iterrows():
            analysis = by_key.get((row["dataset_name"], row["algorithm_id"], row["metric"]))
            if analysis:
                for column in ("ci_lower", "ci_upper", "uncertainty_status"):
                    per_dataset.at[index, column] = analysis[column]
    return aggregate, per_dataset


def build_official_benchmark_view(
    task_family="pair",
    metric=None,
    method_id=ALL_OFFICIAL_METHODS,
    artifact_path=None,
):
    artifact_path = Path(artifact_path or OFFICIAL_BENCHMARK_PATH)
    try:
        payload = load_official_benchmark_payload(artifact_path)
    except FileNotFoundError:
        return _empty_view(
            "The completed, validated result bundle will appear here when published."
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _empty_view(str(exc), error=True)

    metadata = payload["metadata"]
    protocol = metadata["benchmark_protocol"]
    aggregate = pd.DataFrame(payload["aggregate"])
    per_dataset = pd.DataFrame(payload["per_dataset"])
    tasks = [task for task in TASK_LABELS if task in set(per_dataset["task_family"])]
    task = task_family if task_family in tasks else tasks[0]
    metrics = _metric_choices(per_dataset, task)
    selected_metric = metric if metric in metrics else DEFAULT_METRICS[task]
    if selected_metric not in metrics:
        selected_metric = metrics[0]

    names = per_dataset[["algorithm_id", "algorithm_name"]].drop_duplicates("algorithm_id")
    names = names.sort_values(["algorithm_name", "algorithm_id"])
    methods = [("All methods", ALL_OFFICIAL_METHODS)] + [
        (f"{row.algorithm_name} · {row.algorithm_id[:12]}", row.algorithm_id)
        for row in names.itertuples(index=False)
    ]
    selected_method = method_id if method_id in set(names["algorithm_id"]) else ALL_OFFICIAL_METHODS
    aggregate = aggregate[
        (aggregate["task_family"] == task) & (aggregate["metric"] == selected_metric)
    ].copy()
    per_dataset = per_dataset[
        (per_dataset["task_family"] == task) & (per_dataset["metric"] == selected_metric)
    ].copy()
    # Compute tradeoffs across the full selected task/metric before filtering a
    # displayed method. A filter must not change its Pareto membership or rank.
    aggregate, per_dataset = _add_diagnostics(aggregate, per_dataset, artifact_path)
    if selected_method != ALL_OFFICIAL_METHODS:
        aggregate = aggregate[aggregate["algorithm_id"] == selected_method]
        per_dataset = per_dataset[per_dataset["algorithm_id"] == selected_method]
    # Keep validated eligibility, complete coverage, nullable ranks, full method
    # identities, and original scores. View filters never calculate new ranks.
    aggregate = aggregate.sort_values(
        ["rank", "mean_score", "algorithm_name", "algorithm_id"],
        ascending=[True, False, True, True],
        na_position="last",
        ignore_index=True,
    )
    per_dataset = per_dataset.sort_values(
        ["dataset_name", "rank", "algorithm_name", "algorithm_id"],
        na_position="last",
        ignore_index=True,
    )
    # Put methods and their scores in the initial viewport. Portable JSON uses
    # sorted keys, which otherwise leaves long identity hashes before results.
    aggregate_first = ["rank", "algorithm_name", "mean_score", "coverage", "eligible", "elapsed_seconds", "status", "pareto_efficient"]
    dataset_first = ["dataset_name", "rank", "algorithm_name", "score", "eligible", "ci_lower", "ci_upper", "elapsed_seconds", "status"]
    aggregate = aggregate[
        aggregate_first + [column for column in aggregate if column not in aggregate_first]
    ]
    per_dataset = per_dataset[
        dataset_first + [column for column in per_dataset if column not in dataset_first]
    ]
    threshold_policy = protocol.get("threshold_policy") or {}
    threshold = (
        f"Fixed at {threshold_policy.get('value')}"
        if threshold_policy.get("mode") == "fixed"
        else str(threshold_policy.get("mode") or "Recorded in protocol")
    )
    selected_datasets = [entry for entry in protocol["datasets"] if entry["task_family"] == task]
    summary = summary_panel_html(
        "Official Plagiarism Benchmark",
        [
            ("Benchmark ID", protocol["benchmark_id"]),
            ("Protocol ID", protocol["protocol_id"]),
            ("Method Budget", f"{protocol['execution_policy']['max_method_seconds']:g} seconds"),
            ("Category", protocol["domain"].replace("_", " ").title()),
            ("Evaluation", TASK_LABELS[task]),
            ("Metric", selected_metric),
            ("Dataset Views", str(len(selected_datasets))),
            ("Weighting", "Equal weight per declared dataset within each task and metric"),
            ("Threshold Policy", threshold)
            if task == "pair"
            else (
                "Retrieval k",
                ", ".join(
                    str(value)
                    for value in sorted({entry.get("k", 10) for entry in selected_datasets})
                ),
            ),
            ("Eligible Methods", str(int(aggregate["eligible"].sum()))),
            ("Unranked Methods", str(int(aggregate["rank"].isna().sum()))),
            ("Uncertainty", "Per-view conditional grouped bootstrap where available; published ranks remain fixed"),
            ("Runtime", "Observed wall time including preparation; hardware and concurrent execution affect comparisons"),
            ("Status", "Validated published results"),
        ],
    )
    return {
        "available": True,
        "summary": summary,
        "task": task,
        "tasks": [(TASK_LABELS[value], value) for value in tasks],
        "metric": selected_metric,
        "metrics": metrics,
        "method": selected_method,
        "methods": methods,
        "aggregate": aggregate,
        "per_dataset": per_dataset,
        "scope": _dataset_scope(protocol),
        "protocol": protocol,
        "configurations": metadata["method_configurations"],
        "artifact": str(Path(artifact_path)),
    }


def update_official_benchmark_gradio(
    task_family,
    metric,
    method_id,
    artifact_path=None,
):
    view = build_official_benchmark_view(task_family, metric, method_id, artifact_path)
    return (
        view["summary"],
        gr.update(choices=view["tasks"], value=view["task"], interactive=view["available"]),
        gr.update(choices=view["metrics"], value=view["metric"], interactive=view["available"]),
        gr.update(choices=view["methods"], value=view["method"], interactive=view["available"]),
        view["aggregate"],
        view["per_dataset"],
        view["scope"],
        view["protocol"],
        view["configurations"],
        view["artifact"],
    )
