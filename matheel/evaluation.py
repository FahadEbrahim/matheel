import math

import pandas as pd

from .calibration import evaluate_threshold
from .datasets import PairDataset, load_code_texts, load_pair_dataset
from .similarity import calculate_similarity


def score_pair_dataset(
    dataset,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    if not isinstance(dataset, PairDataset):
        dataset = load_pair_dataset(dataset)
    texts = load_code_texts(dataset)
    options = dict(similarity_options or {})
    rows = []
    for pair in dataset.pairs.to_dict(orient="records"):
        left_id = str(pair["left_id"])
        right_id = str(pair["right_id"])
        left_text = texts[left_id]
        right_text = texts[right_id]
        if scorer is None:
            score = calculate_similarity(left_text, right_text, **options)
        else:
            score = scorer(left_text, right_text, pair)
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"Pair score must be finite. Got: {score}")
        row = dict(pair)
        row[score_column] = numeric_score
        rows.append(row)
    return pd.DataFrame(rows)


def pair_classification_metrics(
    scored_pairs,
    threshold=0.5,
    score_column="similarity_score",
    label_column="label",
):
    frame = scored_pairs.copy() if isinstance(scored_pairs, pd.DataFrame) else pd.DataFrame(scored_pairs)
    for column in (score_column, label_column):
        if column not in frame.columns:
            raise ValueError(f"scored_pairs is missing required column: {column}")
    metrics = evaluate_threshold(
        frame.to_dict(orient="records"),
        threshold=threshold,
        score_key=score_column,
        label_key=label_column,
    )
    metrics["pair_count"] = int(len(frame))
    return metrics


def evaluate_pair_dataset(
    dataset,
    threshold=0.5,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    scored_pairs = score_pair_dataset(
        dataset,
        scorer=scorer,
        similarity_options=similarity_options,
        score_column=score_column,
    )
    metrics = pair_classification_metrics(
        scored_pairs,
        threshold=threshold,
        score_column=score_column,
    )
    return scored_pairs, metrics
