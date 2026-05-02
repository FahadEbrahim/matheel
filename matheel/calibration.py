import math

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
