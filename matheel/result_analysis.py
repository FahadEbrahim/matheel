"""Source-free error inspection and paired uncertainty over existing scored rows.

This module never fits a threshold or changes a published ranking. Retrieval
resamples queries. Pair classification resamples connected components of source
IDs (or an explicitly supplied grouping column), keeping shared code together.
"""

import json
import math
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import precision_recall_curve, roc_curve
from .evaluation import pair_classification_metrics, retrieval_ranking_metrics


_DERIVED_COLUMNS = {"reference_score", "reference_rank", "score_delta", "decision_disagreement",
                    "threshold_distance", "outcome", "pair_occurrence", "score_row_index", "rank"}


def _enrich_from_dataset(frame, task, dataset_root):
    from .datasets import load_code_texts, load_pair_dataset, load_retrieval_dataset

    dataset = load_pair_dataset(dataset_root) if task == "pair" else load_retrieval_dataset(dataset_root)
    texts = load_code_texts(dataset)
    files = dataset.files.copy()
    files["file_id"] = files["file_id"].astype(str)
    file_rows = files.set_index("file_id").to_dict("index")
    suffixes = {".py": "python", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp",
                ".cc": "cpp", ".cs": "csharp", ".js": "javascript", ".ts": "typescript",
                ".go": "go", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".rs": "rust"}
    languages = dataset.metadata.get("languages") or []
    if isinstance(languages, str):
        languages = [languages]

    def language(file_id):
        row = file_rows[file_id]
        explicit = row.get("language")
        if explicit is not None and not pd.isna(explicit):
            return str(explicit)
        suffix = str(row.get("suffix") or Path(str(row.get("file_path", ""))).suffix).lower()
        return suffixes.get(suffix, str(languages[0]) if len(languages) == 1 else "unknown")

    if task == "pair":
        endpoints = list(zip(frame["left_id"], frame["right_id"]))
    else:
        queries = {str(row["query_id"]): str(row["file_id"])
                   for row in dataset.queries.to_dict("records")}
        endpoints = [(queries.get(query),) for query in frame["query_id"]]
    if any(file_id not in texts for pair in endpoints for file_id in pair):
        raise ValueError("Analysis rows reference source IDs outside the supplied normalized dataset.")
    if "language" not in frame:
        frame["language"] = [values[0] if len(set(values)) == 1 else "mixed"
                             for values in ([language(file_id) for file_id in pair] for pair in endpoints)]
    if "length_bucket" not in frame:
        lengths = [max(len(texts[file_id]) for file_id in pair) for pair in endpoints]
        frame["source_characters"] = lengths
        frame["length_bucket"] = ["short (<=1000 chars)" if count <= 1000 else
                                  "medium (<=10000 chars)" if count <= 10000 else "long (>10000 chars)"
                                  for count in lengths]
    return frame


def _frame(rows):
    if isinstance(rows, (str, Path)):
        return pd.read_csv(rows, dtype={key: str for key in (
            "left_id", "right_id", "query_id", "document_id", "query_file_id", "document_file_id",
        )})
    return rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)


def _prepare(rows, task, score_column):
    frame = _frame(rows)
    if score_column in _DERIVED_COLUMNS:
        raise ValueError("score_column must not use a reserved analysis-derived column name.")
    # Exported errors can be analyzed again. Derived fields always describe the
    # current comparison, never inherited values or merge-generated _x/_y names.
    frame = frame.drop(columns=list(_DERIVED_COLUMNS), errors="ignore")
    frame["score_row_index"] = np.arange(len(frame))
    task = ("retrieval" if "query_id" in frame else "pair") if task == "auto" else task
    if task not in {"pair", "retrieval"}:
        raise ValueError("task must be auto, pair, or retrieval.")
    keys = ["left_id", "right_id"] if task == "pair" else ["query_id", "document_id"]
    truth = "label" if task == "pair" else "relevance"
    missing = set(keys + [truth, score_column]).difference(frame.columns)
    if missing:
        raise ValueError(f"Scored rows are missing columns: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ValueError("Scored rows must not be empty.")
    for key in keys:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Scored rows contain missing {key} values.")
        frame[key] = frame[key].astype(str)
    if task == "retrieval" and frame.duplicated(keys).any():
        raise ValueError("Scored rows must contain one score per declared pair.")
    if task == "pair":
        # Supplied pair datasets may repeat observations. Preserve their weight
        # and distinguish occurrences when aligning another method's scores.
        frame["pair_occurrence"] = frame.groupby(keys, sort=False).cumcount()
        keys = keys + ["pair_occurrence"]
    for column in (truth, score_column):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} values must be finite.")
    if task == "pair" and not frame[truth].isin([0, 1]).all():
        raise ValueError("Pair labels must be binary 0/1 values.")
    if task == "retrieval" and (frame[truth] < 0).any():
        raise ValueError("Retrieval relevance must be nonnegative.")
    return frame.sort_values(keys, kind="stable").reset_index(drop=True), task, keys, truth


def _metrics(frame, task, threshold, k, score_column):
    if task == "retrieval":
        return {key: value for key, value in retrieval_ranking_metrics(
            frame, k=k, score_column=score_column,
        ).items() if key not in {"query_count", "result_count", "relevant_count", "k"}}
    result = {key: value for key, value in pair_classification_metrics(
        frame, threshold=threshold, score_column=score_column,
    ).items() if key in {"precision", "recall", "f1", "accuracy"}}
    if frame["label"].nunique() == 2:
        result["average_precision"] = precision_recall_curve(
            frame.to_dict("records"), score_key=score_column,
        ).attrs["average_precision"]
        result["auroc"] = roc_curve(
            frame.to_dict("records"), score_key=score_column,
        ).attrs["auroc"]
    return result


def _pair_groups(frame, group_column):
    if group_column:
        if group_column not in frame or frame[group_column].isna().any():
            raise ValueError("The grouping column must exist and contain no missing values.")
        return frame[group_column].astype(str).tolist(), f"column:{group_column}"
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in zip(frame["left_id"], frame["right_id"]):
        a, b = find(left), find(right)
        parent[b] = a
    return [find(value) for value in frame["left_id"]], "connected_source_components"


def _pair_bootstrap_context(frame, threshold, score_column):
    scores = frame[score_column].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=float)
    order = np.argsort(-scores, kind="stable")
    ends = np.flatnonzero(np.r_[scores[order][1:] != scores[order][:-1], True])
    return labels, scores >= threshold, order, ends


def _weighted_pair_metrics(context, weights):
    """Reuse fixed score ordering for bootstrap multiplicities, including ties."""
    labels, predicted, order, ends = context
    positives, negatives = weights * labels, weights * (1 - labels)
    tp, fp = float(positives[predicted].sum()), float(negatives[predicted].sum())
    total_p, total_n = float(positives.sum()), float(negatives.sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / total_p if total_p else 0.
    result = {"precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
              "accuracy": (tp + total_n - fp) / (total_p + total_n)}
    if total_p and total_n:
        cumulative_p, cumulative_n = np.cumsum(positives[order])[ends], np.cumsum(negatives[order])[ends]
        group_p, group_n = np.diff(np.r_[0, cumulative_p]), np.diff(np.r_[0, cumulative_n])
        denominator = cumulative_p + cumulative_n
        group_precision = np.divide(cumulative_p, denominator, out=np.zeros_like(cumulative_p),
                                    where=denominator > 0)
        result["average_precision"] = float(np.sum(group_p * group_precision) / total_p)
        result["auroc"] = float(np.sum(group_p * (total_n - cumulative_n + .5 * group_n)) / (total_p * total_n))
    return result


def _error_rows(frame, task, threshold, k, score_column, reference):
    rows = frame.copy()
    if task == "pair":
        predicted = rows[score_column] >= threshold
        positive = rows["label"] == 1
        rows["outcome"] = np.select(
            [predicted & positive, predicted & ~positive, ~predicted & positive],
            ["true_positive", "false_positive", "false_negative"], default="true_negative",
        )
        rows["threshold_distance"] = (rows[score_column] - threshold).abs()
    else:
        # Same deterministic score/ID ordering used by retrieval_ranking_metrics.
        rows = rows.sort_values(["query_id", score_column, "document_id"],
                                ascending=[True, False, True], kind="stable")
        rows["rank"] = rows.groupby("query_id", sort=False).cumcount() + 1
        positive, retrieved = rows["relevance"] > 0, rows["rank"] <= k
        rows["outcome"] = np.select(
            [positive & retrieved, positive & ~retrieved, ~positive & retrieved],
            ["relevant_in_top_k", "relevant_below_k", "nonrelevant_in_top_k"],
            default="nonrelevant_below_k",
        )
    if reference is not None:
        keys = ["left_id", "right_id", "pair_occurrence"] if task == "pair" else ["query_id", "document_id"]
        other = reference[keys + [score_column]].rename(columns={score_column: "reference_score"})
        rows = rows.merge(other, on=keys, validate="one_to_one")
        rows["score_delta"] = rows[score_column] - rows["reference_score"]
        if task == "pair":
            rows["decision_disagreement"] = (rows[score_column] >= threshold) != (
                rows["reference_score"] >= threshold)
        else:
            other = other.sort_values(["query_id", "reference_score", "document_id"],
                                      ascending=[True, False, True], kind="stable")
            other["reference_rank"] = other.groupby("query_id", sort=False).cumcount() + 1
            rows = rows.merge(other[keys + ["reference_rank"]], on=keys, validate="one_to_one")
            rows["decision_disagreement"] = (rows["rank"] <= k) != (rows["reference_rank"] <= k)
    return rows.reset_index(drop=True)


def analyze_scored_results(
    scored_rows, *, task="auto", reference_rows=None, threshold=0.5, k=10,
    score_column="similarity_score", group_column=None, rounds=500, seed=7, confidence=0.95,
    dataset_root=None,
):
    """Inspect one candidate universe, optionally paired with another method.

    Score CSVs must contain the complete evaluated candidate universe: this
    function cannot recover excluded candidates or unrecorded relevance labels.
    Confidence intervals are conditional on these supplied datasets and group
    assumptions, and are not a claim about unseen assignments or datasets.
    """
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 10000:
        raise ValueError("rounds must be an integer between 1 and 10000.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1.")
    frame, task, keys, truth = _prepare(scored_rows, task, score_column)
    if dataset_root is not None:
        frame = _enrich_from_dataset(frame, task, dataset_root)
    reference = None
    if reference_rows is not None:
        reference, _, _, _ = _prepare(reference_rows, task, score_column)
        if not frame[keys + [truth]].equals(reference[keys + [truth]]):
            raise ValueError("Paired methods must have identical candidates and ground truth.")
        if group_column and (group_column not in reference or not frame[group_column].equals(
            reference[group_column]
        )):
            raise ValueError("Paired methods must have identical grouping values.")
    point = _metrics(frame, task, threshold, k, score_column)
    reference_point = _metrics(reference, task, threshold, k, score_column) if reference is not None else {}
    if task == "pair":
        groups, grouping = _pair_groups(frame, group_column)
    else:
        if group_column:
            raise ValueError("Retrieval analysis resamples query IDs; do not supply group_column.")
        groups, grouping = frame["query_id"].tolist(), "query_id"
    grouped = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    indices = [np.asarray(value, dtype=int) for value in grouped.values()]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    differences = {key: [] for key in point} if reference is not None else {}
    per_query = []
    if task == "retrieval":
        per_query = [_metrics(frame.iloc[index], task, threshold, k, score_column) for index in indices]
        reference_queries = ([_metrics(reference.iloc[index], task, threshold, k, score_column)
                              for index in indices] if reference is not None else [])
    else:
        pair_context = _pair_bootstrap_context(frame, threshold, score_column)
        other_context = (_pair_bootstrap_context(reference, threshold, score_column)
                         if reference is not None else None)
    if len(indices) >= 2:
        for _ in range(rounds):
            selected = rng.integers(0, len(indices), len(indices))
            if task == "retrieval":
                current = {key: float(np.mean([per_query[i][key] for i in selected])) for key in point}
                other = ({key: float(np.mean([reference_queries[i][key] for i in selected]))
                          for key in point} if reference is not None else {})
            else:
                selected_rows = np.concatenate([indices[i] for i in selected])
                weights = np.bincount(selected_rows, minlength=len(frame))
                current = _weighted_pair_metrics(pair_context, weights)
                other = (_weighted_pair_metrics(other_context, weights)
                         if reference is not None else {})
            for key in samples:
                if key in current:
                    samples[key].append(current[key])
                if key in differences and key in current and key in other:
                    differences[key].append(current[key] - other[key])
    alpha = (1 - confidence) / 2
    metric_rows = []
    for key, value in point.items():
        values = samples[key]
        row = {"metric": key, "score": value, "ci_lower": None, "ci_upper": None,
               "valid_rounds": len(values), "uncertainty_status": "available"}
        if len(indices) < 2:
            row["uncertainty_status"] = "N/A: fewer than two independent groups"
        elif len(values) < max(2, math.ceil(rounds * 0.8)):
            row["uncertainty_status"] = "N/A: too few valid bootstrap rounds"
        else:
            row["ci_lower"], row["ci_upper"] = [float(v) for v in np.quantile(values, [alpha, 1 - alpha])]
        if reference is not None:
            row.update(reference_score=reference_point[key], delta=value - reference_point[key],
                       delta_ci_lower=None, delta_ci_upper=None)
            values = differences[key]
            if row["uncertainty_status"] == "available" and len(values) >= max(2, math.ceil(rounds * .8)):
                row["delta_ci_lower"], row["delta_ci_upper"] = [float(v) for v in np.quantile(values, [alpha, 1-alpha])]
        metric_rows.append(row)
    errors = _error_rows(frame, task, threshold, k, score_column, reference)
    errors = errors.astype(object).where(pd.notna(errors), None)
    slices, slice_notes = [], []
    for column in ("language", "assignment", "length_bucket"):
        if column in frame:
            if task == "retrieval" and frame.groupby("query_id")[column].nunique(dropna=False).gt(1).any():
                slice_notes.append(f"Skipped {column}: retrieval slices must keep every query's candidate rows together.")
                continue
            for value, subset in frame.groupby(column, dropna=False, sort=True):
                slices.append({"slice_column": column, "slice_value": str(value), "rows": len(subset),
                               **_metrics(subset, task, threshold, k, score_column)})
    queries = [{"query_id": group, **metric} for group, metric in zip(grouped, per_query)]
    return {
        "schema_version": 1, "task_family": task, "score_column": score_column,
        "threshold": float(threshold), "k": k,
        "rows": len(frame), "confidence": confidence, "rounds": rounds, "seed": seed,
        "grouping": grouping, "group_count": len(indices), "paired": reference is not None,
        "interpretation": "Conditional grouped percentile bootstrap on supplied scored rows; no threshold fitting, candidate recovery, or official rank changes.",
        "metrics": metric_rows, "errors": errors.to_dict("records"), "slices": slices,
        "per_query": queries, "slice_notes": slice_notes,
    }


def write_result_analysis(report, output_dir, basename="result_analysis"):
    """Write escaped HTML, JSON and source-free CSVs for local inspection."""
    if not basename or Path(basename).name != basename or basename in {".", ".."}:
        raise ValueError("basename must be a single filename stem.")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts = {"json": target / f"{basename}.json", "html": target / f"{basename}.html"}
    # CSV metadata may be user-supplied. Do not emit source text or local paths.
    safe_report = dict(report)
    public_columns = {"left_id", "right_id", "pair_occurrence", "query_id", "document_id", "label", "relevance",
                      "similarity_score", "rank", "outcome", "threshold_distance", "reference_score",
                      "reference_rank", "score_delta", "decision_disagreement", "language", "assignment",
                      "length_bucket", "query_file_id", "document_file_id", "score_row_index", "source_characters"}
    public_columns.add(report.get("score_column", "similarity_score"))
    safe_report["errors"] = [{key: value for key, value in row.items() if key in public_columns}
                             for row in report["errors"]]
    artifacts["json"].write_text(json.dumps(safe_report, indent=2, allow_nan=False), encoding="utf-8")
    tables = []
    for name in ("metrics", "errors", "slices", "per_query"):
        frame = pd.DataFrame(safe_report[name])
        artifacts[name] = target / f"{basename}_{name}.csv"
        frame.to_csv(artifacts[name], index=False)
        tables.append(f"<h2>{escape(name.replace('_', ' ').title())}</h2>" +
                      frame.head(200).to_html(index=False, escape=True))
    artifacts["html"].write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Result analysis</title>'
        '<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #ddd}</style>'
        '<h1>Result analysis</h1><p>Each table previews up to 200 rows; CSVs contain all rows.</p><p>' + escape(report["interpretation"]) + '</p><p>Grouping: '
        + escape(report["grouping"]) + f'; groups: {report["group_count"]}</p>' + ''.join(tables) + '</html>',
        encoding="utf-8",
    )
    return artifacts
