import json
import math
from html import escape
from pathlib import Path

import pandas as pd

from .feature_weights import available_default_features
from .vectors import similarity_function_score_range


_BOUNDED_FEATURES = {
    "levenshtein",
    "jaro_winkler",
    "winnowing",
    "gst",
    "code_metric",
}


def _finite_score(value):
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("Scores must be finite numbers.")
    return score


def _coerce_label(value, positive_label=True):
    if positive_label is True:
        return bool(value)
    return value == positive_label


def _coerce_score_label_pairs(
    scores,
    labels=None,
    score_key="score",
    label_key="label",
    positive_label=True,
):
    if labels is not None:
        score_values = list(scores)
        label_values = list(labels)
        if len(score_values) != len(label_values):
            raise ValueError("scores and labels must have the same length.")
        pairs = zip(score_values, label_values)
    elif isinstance(scores, pd.DataFrame):
        if score_key not in scores.columns or label_key not in scores.columns:
            raise ValueError(f"scored rows must contain {score_key!r} and {label_key!r}.")
        pairs = ((row[score_key], row[label_key]) for row in scores.to_dict(orient="records"))
    else:
        pairs = []
        for item in scores:
            if isinstance(item, dict):
                if score_key not in item or label_key not in item:
                    raise ValueError(f"Each labeled score mapping must contain {score_key!r} and {label_key!r}.")
                pairs.append((item[score_key], item[label_key]))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError("Labeled scores must be mappings or (score, label) pairs.")

    parsed = [(_finite_score(score), _coerce_label(label, positive_label=positive_label)) for score, label in pairs]
    if not parsed:
        raise ValueError("At least one labeled score is required.")
    return parsed


def _class_counts(pairs):
    positive_count = sum(1 for _, label in pairs if label)
    negative_count = len(pairs) - positive_count
    return positive_count, negative_count


def _require_two_classes(pairs):
    positive_count, negative_count = _class_counts(pairs)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC and precision-recall reports require at least one positive and one negative label.")
    return positive_count, negative_count


def _threshold_candidates(scores):
    unique_scores = sorted(set(float(score) for score in scores))
    if len(unique_scores) == 1:
        margin = max(abs(unique_scores[0]) * 1e-12, 1e-12)
        return (unique_scores[0] - margin, unique_scores[0], unique_scores[0] + margin)

    span = unique_scores[-1] - unique_scores[0]
    margin = max(span * 1e-12, 1e-12)
    candidates = [unique_scores[0] - margin, unique_scores[-1] + margin]
    candidates.extend(unique_scores)
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:])
    )
    return tuple(sorted(set(candidates)))


def _ordered_curve_thresholds(scores, greater_is_match=True):
    unique_scores = sorted(set(float(score) for score in scores))
    if len(unique_scores) == 1:
        margin = max(abs(unique_scores[0]) * 1e-12, 1e-12)
    else:
        margin = max((unique_scores[-1] - unique_scores[0]) * 1e-12, 1e-12)
    lower = unique_scores[0] - margin
    upper = unique_scores[-1] + margin
    if greater_is_match:
        return tuple([upper, *reversed(unique_scores), lower])
    return tuple([lower, *unique_scores, upper])


def _trapezoid_auc(x_values, y_values):
    points = sorted((float(x), float(y)) for x, y in zip(x_values, y_values))
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def _average_precision_from_rows(rows):
    previous_recall = 0.0
    average_precision = 0.0
    for row in rows:
        recall = float(row["recall"])
        precision = float(row["precision"])
        delta = max(0.0, recall - previous_recall)
        average_precision += delta * precision
        previous_recall = max(previous_recall, recall)
    return average_precision


def feature_score_range(
    feature_name,
    similarity_function="cosine",
    normalize_semantic_scores=False,
):
    key = str(feature_name or "").strip()
    if key == "semantic":
        return similarity_function_score_range(
            similarity_function,
            normalize_score=normalize_semantic_scores,
        )
    if key in _BOUNDED_FEATURES:
        return (0.0, 1.0)
    supported = ", ".join(available_default_features())
    raise ValueError(f"Unsupported feature name: {feature_name}. Supported features: {supported}.")


def evaluate_threshold(
    scores,
    labels=None,
    threshold=0.5,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    threshold = _finite_score(threshold)
    true_positive = false_positive = true_negative = false_negative = 0
    for score, is_positive in pairs:
        predicted_positive = score >= threshold if greater_is_match else score <= threshold
        if predicted_positive and is_positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif is_positive:
            false_negative += 1
        else:
            true_negative += 1

    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative
    precision = true_positive / predicted_positive_count if predicted_positive_count else 0.0
    recall = true_positive / actual_positive_count if actual_positive_count else 0.0
    f1_denom = precision + recall
    f1 = (2.0 * precision * recall / f1_denom) if f1_denom else 0.0
    total = len(pairs)
    accuracy = (true_positive + true_negative) / float(total)

    return {
        "threshold": threshold,
        "positive_count": actual_positive_count,
        "negative_count": true_negative + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def calibration_curve(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    selected_thresholds = thresholds
    if selected_thresholds is None:
        selected_thresholds = _threshold_candidates([score for score, _ in pairs])
    return [
        evaluate_threshold(
            pairs,
            threshold=threshold,
            positive_label=True,
            greater_is_match=greater_is_match,
        )
        for threshold in selected_thresholds
    ]


def threshold_sweep(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
):
    rows = calibration_curve(
        scores,
        labels=labels,
        thresholds=thresholds,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
        greater_is_match=greater_is_match,
    )
    frame = pd.DataFrame(rows)
    frame.attrs["curve_type"] = "threshold_sweep"
    frame.attrs["score_key"] = score_key
    frame.attrs["label_key"] = label_key
    frame.attrs["greater_is_match"] = bool(greater_is_match)
    return frame


def roc_curve(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    positive_count, negative_count = _require_two_classes(pairs)
    selected_thresholds = thresholds
    if selected_thresholds is None:
        selected_thresholds = _ordered_curve_thresholds(
            [score for score, _ in pairs],
            greater_is_match=greater_is_match,
        )

    rows = []
    for threshold in selected_thresholds:
        row = evaluate_threshold(
            pairs,
            threshold=threshold,
            positive_label=True,
            greater_is_match=greater_is_match,
        )
        row["true_positive_rate"] = row["recall"]
        row["false_positive_rate"] = row["false_positive"] / float(negative_count)
        row["positive_count"] = positive_count
        row["negative_count"] = negative_count
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["curve_type"] = "roc"
    frame.attrs["auroc"] = _trapezoid_auc(frame["false_positive_rate"], frame["true_positive_rate"])
    frame.attrs["score_key"] = score_key
    frame.attrs["label_key"] = label_key
    frame.attrs["greater_is_match"] = bool(greater_is_match)
    return frame


def precision_recall_curve(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    positive_count, negative_count = _require_two_classes(pairs)
    selected_thresholds = thresholds
    if selected_thresholds is None:
        selected_thresholds = _ordered_curve_thresholds(
            [score for score, _ in pairs],
            greater_is_match=greater_is_match,
        )

    rows = []
    for threshold in selected_thresholds:
        row = evaluate_threshold(
            pairs,
            threshold=threshold,
            positive_label=True,
            greater_is_match=greater_is_match,
        )
        row["predicted_positive"] = row["true_positive"] + row["false_positive"]
        row["positive_count"] = positive_count
        row["negative_count"] = negative_count
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["curve_type"] = "precision_recall"
    frame.attrs["average_precision"] = _average_precision_from_rows(rows)
    frame.attrs["score_key"] = score_key
    frame.attrs["label_key"] = label_key
    frame.attrs["greater_is_match"] = bool(greater_is_match)
    return frame


def calibrate_threshold(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
    optimize="f1",
):
    metric_name = str(optimize or "f1").strip().lower()
    if metric_name not in {"f1", "accuracy", "precision", "recall"}:
        raise ValueError("optimize must be one of: f1, accuracy, precision, recall.")

    curve = calibration_curve(
        scores,
        labels=labels,
        thresholds=thresholds,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
        greater_is_match=greater_is_match,
    )
    threshold_direction = 1.0 if greater_is_match else -1.0
    best = max(
        curve,
        key=lambda item: (
            item[metric_name],
            item["precision"],
            item["recall"],
            item["accuracy"],
            threshold_direction * item["threshold"],
        ),
    )
    result = dict(best)
    result["optimized_metric"] = metric_name
    result["candidate_count"] = len(curve)
    return result


def calibration_report(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
    optimize="f1",
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    positive_count, negative_count = _require_two_classes(pairs)
    sweep = threshold_sweep(
        pairs,
        thresholds=thresholds,
        positive_label=True,
        greater_is_match=greater_is_match,
    )
    roc = roc_curve(
        pairs,
        positive_label=True,
        greater_is_match=greater_is_match,
    )
    pr = precision_recall_curve(
        pairs,
        positive_label=True,
        greater_is_match=greater_is_match,
    )
    best = calibrate_threshold(
        pairs,
        thresholds=thresholds,
        positive_label=True,
        greater_is_match=greater_is_match,
        optimize=optimize,
    )
    summary = {
        "pair_count": int(len(pairs)),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "score_key": score_key,
        "label_key": label_key,
        "greater_is_match": bool(greater_is_match),
        "auroc": float(roc.attrs["auroc"]),
        "average_precision": float(pr.attrs["average_precision"]),
        "optimized_threshold": best,
    }
    return {
        "summary": summary,
        "threshold_sweep": sweep,
        "roc": roc,
        "precision_recall": pr,
    }


def calibration_report_payload(report):
    return {
        "schema_version": 1,
        "summary": _json_safe_mapping(report["summary"]),
        "threshold_sweep": _frame_records(report["threshold_sweep"]),
        "roc": _frame_records(report["roc"]),
        "precision_recall": _frame_records(report["precision_recall"]),
    }


def threshold_tuning_report(
    scores,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
    optimize="f1",
):
    pairs = _coerce_score_label_pairs(
        scores,
        labels=labels,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
    )
    positive_count, negative_count = _class_counts(pairs)
    sweep = threshold_sweep(
        pairs,
        thresholds=thresholds,
        positive_label=True,
        greater_is_match=greater_is_match,
    )
    best = calibrate_threshold(
        pairs,
        thresholds=thresholds,
        positive_label=True,
        greater_is_match=greater_is_match,
        optimize=optimize,
    )
    warnings = []
    roc = pd.DataFrame()
    pr = pd.DataFrame()
    auroc = None
    average_precision = None
    if positive_count and negative_count:
        roc = roc_curve(pairs, positive_label=True, greater_is_match=greater_is_match)
        pr = precision_recall_curve(pairs, positive_label=True, greater_is_match=greater_is_match)
        auroc = float(roc.attrs["auroc"])
        average_precision = float(pr.attrs["average_precision"])
    else:
        warnings.append(
            "ROC and precision-recall summaries require at least one positive and one negative label."
        )

    summary = {
        "pair_count": int(len(pairs)),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "score_key": score_key,
        "label_key": label_key,
        "greater_is_match": bool(greater_is_match),
        "optimized_metric": str(optimize or "f1").strip().lower(),
        "optimized_threshold": best,
        "candidate_count": int(len(sweep)),
        "auroc": auroc,
        "average_precision": average_precision,
        "warnings": warnings,
    }
    return {
        "summary": summary,
        "threshold_sweep": sweep,
        "roc": roc,
        "precision_recall": pr,
    }


def threshold_tuning_report_payload(report):
    payload = {
        "schema_version": 1,
        "summary": _json_safe_mapping(report["summary"]),
        "threshold_sweep": _frame_records(report["threshold_sweep"]),
    }
    if report["roc"] is not None and not report["roc"].empty:
        payload["roc"] = _frame_records(report["roc"])
    if report["precision_recall"] is not None and not report["precision_recall"].empty:
        payload["precision_recall"] = _frame_records(report["precision_recall"])
    return payload


def threshold_tuning_report_html(report):
    payload = threshold_tuning_report_payload(report)
    summary = payload["summary"]
    optimized = summary["optimized_threshold"]
    warning_rows = "".join(
        f"<li>{escape(str(warning))}</li>" for warning in summary.get("warnings", [])
    )
    if not warning_rows:
        warning_rows = "<li>None</li>"
    preview_rows = []
    for row in payload.get("threshold_sweep", [])[:50]:
        preview_rows.append(
            "<tr><td>{threshold:.6g}</td><td>{precision:.4f}</td><td>{recall:.4f}</td>"
            "<td>{f1:.4f}</td><td>{accuracy:.4f}</td></tr>".format(**row)
        )
    if not preview_rows:
        preview_rows.append('<tr><td colspan="5">No threshold rows</td></tr>')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Matheel Threshold Tuning</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2328; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 0.4rem; }}
    h2 {{ font-size: 1.05rem; margin-top: 1.4rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 0.6rem; }}
    th, td {{ border: 1px solid #d6d9df; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Threshold Tuning</h1>
  <p>Pairs: {summary['pair_count']}. Positives: {summary['positive_count']}. Negatives: {summary['negative_count']}.</p>
  <p>Best threshold by {escape(str(summary['optimized_metric']))}: {float(optimized['threshold']):.6g}
     with F1 {float(optimized['f1']):.4f}, precision {float(optimized['precision']):.4f},
     recall {float(optimized['recall']):.4f}, and accuracy {float(optimized['accuracy']):.4f}.</p>
  <h2>Warnings</h2>
  <ul>{warning_rows}</ul>
  <h2>Threshold Preview</h2>
  <table>
    <thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Accuracy</th></tr></thead>
    <tbody>{''.join(preview_rows)}</tbody>
  </table>
</body>
</html>
"""


def write_threshold_tuning_report_artifacts(
    scores,
    output_dir,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
    optimize="f1",
    basename="threshold_tuning",
):
    report = threshold_tuning_report(
        scores,
        labels=labels,
        thresholds=thresholds,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
        greater_is_match=greater_is_match,
        optimize=optimize,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_basename(basename or "threshold_tuning")
    artifacts = {
        "threshold_sweep_csv": target / f"{stem}_threshold_sweep.csv",
        "summary_json": target / f"{stem}_summary.json",
        "report_json": target / f"{stem}_report.json",
        "report_html": target / f"{stem}_report.html",
    }
    report["threshold_sweep"].to_csv(artifacts["threshold_sweep_csv"], index=False)
    if report["roc"] is not None and not report["roc"].empty:
        artifacts["roc_csv"] = target / f"{stem}_roc.csv"
        report["roc"].to_csv(artifacts["roc_csv"], index=False)
    if report["precision_recall"] is not None and not report["precision_recall"].empty:
        artifacts["precision_recall_csv"] = target / f"{stem}_precision_recall.csv"
        report["precision_recall"].to_csv(artifacts["precision_recall_csv"], index=False)
    artifacts["summary_json"].write_text(
        json.dumps(_json_safe_mapping(report["summary"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts["report_json"].write_text(
        json.dumps(threshold_tuning_report_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts["report_html"].write_text(threshold_tuning_report_html(report), encoding="utf-8")
    return report, artifacts


def write_calibration_report_artifacts(
    scores,
    output_dir,
    labels=None,
    thresholds=None,
    score_key="score",
    label_key="label",
    positive_label=True,
    greater_is_match=True,
    optimize="f1",
    basename="calibration",
):
    report = calibration_report(
        scores,
        labels=labels,
        thresholds=thresholds,
        score_key=score_key,
        label_key=label_key,
        positive_label=positive_label,
        greater_is_match=greater_is_match,
        optimize=optimize,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_basename(basename or "calibration")
    artifacts = {
        "threshold_sweep_csv": target / f"{stem}_threshold_sweep.csv",
        "roc_csv": target / f"{stem}_roc.csv",
        "precision_recall_csv": target / f"{stem}_precision_recall.csv",
        "summary_json": target / f"{stem}_summary.json",
        "report_json": target / f"{stem}_report.json",
    }
    report["threshold_sweep"].to_csv(artifacts["threshold_sweep_csv"], index=False)
    report["roc"].to_csv(artifacts["roc_csv"], index=False)
    report["precision_recall"].to_csv(artifacts["precision_recall_csv"], index=False)
    artifacts["summary_json"].write_text(
        json.dumps(_json_safe_mapping(report["summary"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts["report_json"].write_text(
        json.dumps(calibration_report_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report, artifacts


def _frame_records(frame):
    return [_json_safe_mapping(row) for row in frame.to_dict(orient="records")]


def _json_safe_mapping(values):
    payload = {}
    for key, value in dict(values).items():
        if isinstance(value, dict):
            payload[str(key)] = _json_safe_mapping(value)
        elif isinstance(value, (list, tuple)):
            payload[str(key)] = [
                _json_safe_mapping(item) if isinstance(item, dict) else item for item in value
            ]
        elif pd.isna(value):
            payload[str(key)] = None
        elif isinstance(value, (int, float, str, bool)) or value is None:
            payload[str(key)] = value
        else:
            payload[str(key)] = value.item() if hasattr(value, "item") else value
    return payload


def _safe_artifact_basename(value):
    text = str(value or "").strip()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    return safe.strip("._") or "calibration"
