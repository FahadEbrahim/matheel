import pandas as pd
import pytest

from matheel.datasets import write_pair_dataset
from matheel.evaluation import (
    evaluate_pair_dataset,
    pair_classification_metrics,
    score_pair_dataset,
)


def _write_tiny_pair_dataset(path):
    return write_pair_dataset(
        path,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
                {"file_id": "c", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
            ]
        ),
    )


def test_score_pair_dataset_uses_custom_scorer(tmp_path):
    dataset = _write_tiny_pair_dataset(tmp_path / "pairs")

    scored = score_pair_dataset(
        dataset,
        scorer=lambda left, right, pair: 1.0 if left == right else 0.0,
    )

    assert scored["similarity_score"].tolist() == [1.0, 0.0]
    assert scored["label"].tolist() == [1, 0]


def test_evaluate_pair_dataset_returns_scored_pairs_and_metrics(tmp_path):
    dataset = _write_tiny_pair_dataset(tmp_path / "pairs")

    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=0.5,
        scorer=lambda left, right, pair: 1.0 if left == right else 0.0,
    )

    assert len(scored) == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_pair_classification_metrics_requires_score_column():
    with pytest.raises(ValueError, match="similarity_score"):
        pair_classification_metrics([{"label": 1}], threshold=0.5)
