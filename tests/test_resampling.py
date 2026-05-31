import numpy as np
import pandas as pd
import pytest

import matheel
from matheel.resampling import (
    DataSplit,
    available_resampling_methods,
    bootstrap_resamples,
    compare_metric_samples,
    kfold_splits,
    metric_summary,
    repeated_kfold_splits,
    single_split,
    summarize_metric_samples,
)


def test_resampling_helpers_are_exported_from_package_root():
    assert matheel.DataSplit is DataSplit
    assert matheel.available_resampling_methods is available_resampling_methods
    assert matheel.single_split is single_split
    assert matheel.kfold_splits is kfold_splits
    assert matheel.repeated_kfold_splits is repeated_kfold_splits
    assert matheel.bootstrap_resamples is bootstrap_resamples
    assert matheel.metric_summary is metric_summary
    assert matheel.summarize_metric_samples is summarize_metric_samples
    assert matheel.compare_metric_samples is compare_metric_samples


def test_single_split_partition_sizes_sum_to_total():
    split = single_split(10, train_size=0.6, validation_size=0.2, seed=123)

    assert len(split.train_indices) + len(split.validation_indices) + len(split.test_indices) == 10
    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert set(split.train_indices).isdisjoint(split.test_indices)
    assert split.method == "single"


def test_kfold_splits_cover_all_items_once_in_test_sets():
    splits = kfold_splits(12, n_splits=3, shuffle=False)

    test_indices = []
    for split in splits:
        test_indices.extend(split.test_indices)
    assert len(splits) == 3
    assert sorted(test_indices) == list(range(12))


def test_kfold_splits_accepts_numpy_integer_count():
    splits = kfold_splits(np.int64(10), n_splits=5, shuffle=False)

    assert len(splits) == 5
    assert sum(len(split.test_indices) for split in splits) == 10


def test_kfold_splits_can_stratify_labels():
    splits = kfold_splits(6, n_splits=3, shuffle=False, labels=[1, 1, 1, 0, 0, 0])

    for split in splits:
        labels = [1, 1, 1, 0, 0, 0]
        split_labels = [labels[index] for index in split.test_indices]
        assert sorted(split_labels) == [0, 1]


def test_kfold_splits_rejects_label_folds_that_would_be_empty():
    with pytest.raises(ValueError, match="smallest label count"):
        kfold_splits(2, n_splits=2, labels=[1, 0])


def test_kfold_splits_rejects_group_folds_that_would_be_empty():
    with pytest.raises(ValueError, match="unique groups"):
        kfold_splits(4, n_splits=3, groups=["a", "a", "b", "b"])


def test_repeated_kfold_splits_marks_method_and_count():
    splits = repeated_kfold_splits(20, n_splits=4, n_repeats=2, seed=42)

    assert len(splits) == 8
    assert all(split.method == "repeated_kfold" for split in splits)


def test_bootstrap_resamples_returns_requested_rounds():
    splits = bootstrap_resamples(range(15), n_rounds=5, sample_size=10, seed=9)

    assert len(splits) == 5
    assert all(split.method == "bootstrap" for split in splits)
    assert all(len(split.train_indices) == 10 for split in splits)
    assert all(set(split.test_indices).isdisjoint(split.train_indices) for split in splits)
    assert all(split.test_indices for split in splits)


def test_bootstrap_resamples_rejects_inputs_without_oob_tests():
    with pytest.raises(ValueError, match="out-of-bag"):
        bootstrap_resamples(1, n_rounds=1)


def test_stratified_single_split_rejects_class_missing_from_requested_partition():
    with pytest.raises(ValueError, match="Stratified single split"):
        single_split(4, train_size=0.5, test_size=0.5, labels=[0, 0, 0, 1], seed=1)


def test_metric_summary_reports_uncertainty_fields():
    summary = metric_summary([0.2, 0.4, 0.8], confidence=0.5)

    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(0.4666666667)
    assert summary["std"] > 0
    assert summary["ci_lower"] == pytest.approx(0.3)
    assert summary["ci_upper"] == pytest.approx(0.6)
    assert summary["confidence"] == 0.5


def test_summarize_metric_samples_excludes_count_columns_by_default():
    summary = summarize_metric_samples(
        pd.DataFrame(
            [
                {"split": "fold_1", "accuracy": 1.0, "f1": 0.8, "pair_count": 2},
                {"split": "fold_2", "accuracy": 0.5, "f1": 0.4, "pair_count": 2},
            ]
        ),
        confidence=0.5,
    )

    assert summary["metric"].tolist() == ["accuracy", "f1"]
    assert summary.loc[summary["metric"] == "accuracy", "mean"].item() == pytest.approx(0.75)


def test_compare_metric_samples_reports_paired_improvement():
    comparison = compare_metric_samples(
        [0.5, 0.6, 0.7],
        [0.6, 0.8, 0.7],
        metric_name="accuracy",
    )

    assert comparison["metric"] == "accuracy"
    assert comparison["count"] == 3
    assert comparison["candidate_mean"] > comparison["baseline_mean"]
    assert comparison["win_count"] == 2
    assert comparison["loss_count"] == 0
    assert comparison["tie_count"] == 1
    assert comparison["sign_test_p_value"] == pytest.approx(0.5)
    assert comparison["conclusion"] == "candidate_better"


def test_resampling_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="n_splits"):
        kfold_splits(3, n_splits=4)

    with pytest.raises(ValueError, match="confidence"):
        metric_summary([0.1], confidence=1.0)

    with pytest.raises(ValueError, match="same length"):
        compare_metric_samples([0.1], [0.1, 0.2])
