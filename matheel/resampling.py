import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    name: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    method: str


def available_resampling_methods():
    return ("single", "kfold", "repeated_kfold", "bootstrap")


def single_split(
    items_or_count,
    train_size=0.7,
    validation_size=0.0,
    test_size=None,
    shuffle=True,
    seed=None,
    labels=None,
    groups=None,
    name="split_1",
):
    count = _normalize_count(items_or_count)
    labels = _normalize_optional_array(labels, count, "labels")
    groups = _normalize_optional_array(groups, count, "groups")
    if labels is not None and groups is not None:
        raise ValueError("single_split supports labels or groups, but not both at once.")

    if groups is not None:
        train_indices, validation_indices, test_indices = _group_single_split(
            count,
            groups=groups,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            shuffle=shuffle,
            seed=seed,
        )
    elif labels is not None:
        train_indices, validation_indices, test_indices = _stratified_single_split(
            count,
            labels=labels,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        indices = _ordered_indices(count, shuffle=shuffle, seed=seed)
        train_count, validation_count, _ = _partition_counts(
            count,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
        train_end = train_count
        validation_end = train_end + validation_count
        train_indices = np.sort(indices[:train_end])
        validation_indices = np.sort(indices[train_end:validation_end])
        test_indices = np.sort(indices[validation_end:])

    return DataSplit(
        name=str(name),
        train_indices=_to_index_tuple(train_indices),
        validation_indices=_to_index_tuple(validation_indices),
        test_indices=_to_index_tuple(test_indices),
        method="single",
    )


def kfold_splits(items_or_count, n_splits=5, shuffle=True, seed=None, labels=None, groups=None, prefix="fold"):
    count = _normalize_count(items_or_count)
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > count:
        raise ValueError("n_splits must not exceed the number of items.")

    fold_tests = _folds_from_labels_or_groups(
        count,
        groups=groups,
        labels=labels,
        n_splits=n_splits,
        shuffle=shuffle,
        seed=seed,
    )

    all_indices = np.arange(count, dtype=int)
    splits = []
    for fold_index, test_indices in enumerate(fold_tests, start=1):
        train_mask = np.ones(count, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        splits.append(
            DataSplit(
                name=f"{prefix}_{fold_index}",
                train_indices=_to_index_tuple(np.sort(train_indices)),
                validation_indices=tuple(),
                test_indices=_to_index_tuple(np.sort(test_indices)),
                method="kfold",
            )
        )
    return splits


def repeated_kfold_splits(
    items_or_count,
    n_splits=5,
    n_repeats=2,
    shuffle=True,
    seed=None,
    labels=None,
    groups=None,
):
    repeats = int(n_repeats)
    if repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    generator = _rng(seed)
    seeds = generator.integers(0, np.iinfo(np.int32).max, size=repeats)
    splits = []
    for repeat_index, repeat_seed in enumerate(seeds.tolist(), start=1):
        repeated = kfold_splits(
            items_or_count,
            n_splits=n_splits,
            shuffle=shuffle,
            seed=int(repeat_seed),
            labels=labels,
            groups=groups,
            prefix=f"repeat_{repeat_index}_fold",
        )
        for split in repeated:
            splits.append(
                DataSplit(
                    name=split.name,
                    train_indices=split.train_indices,
                    validation_indices=split.validation_indices,
                    test_indices=split.test_indices,
                    method="repeated_kfold",
                )
            )
    return splits


def bootstrap_resamples(items_or_count, n_rounds=100, sample_size=None, seed=None, prefix="bootstrap"):
    count = _normalize_count(items_or_count)
    rounds = int(n_rounds)
    if rounds < 1:
        raise ValueError("n_rounds must be at least 1.")

    sample_count = count if sample_size is None else int(sample_size)
    if sample_count <= 0:
        raise ValueError("sample_size must be positive.")

    generator = _rng(seed)
    all_indices = np.arange(count, dtype=int)
    splits = []
    for round_index in range(1, rounds + 1):
        train_indices = generator.choice(all_indices, size=sample_count, replace=True)
        unique_train = set(int(value) for value in train_indices.tolist())
        test_indices = np.asarray(
            [index for index in all_indices.tolist() if index not in unique_train],
            dtype=int,
        )
        splits.append(
            DataSplit(
                name=f"{prefix}_{round_index}",
                train_indices=_to_index_tuple(train_indices),
                validation_indices=tuple(),
                test_indices=_to_index_tuple(test_indices),
                method="bootstrap",
            )
        )
    return splits


def metric_summary(values, confidence=0.95):
    array = _finite_values(values, "values")
    confidence = _normalize_confidence(confidence)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "standard_error": float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "ci_lower": _quantile(array, (1.0 - confidence) / 2.0),
        "ci_upper": _quantile(array, 1.0 - ((1.0 - confidence) / 2.0)),
        "confidence": confidence,
    }


def summarize_metric_samples(metric_rows, metric_columns=None, confidence=0.95):
    frame = metric_rows.copy() if isinstance(metric_rows, pd.DataFrame) else pd.DataFrame(metric_rows)
    if frame.empty:
        raise ValueError("metric_rows must contain at least one row.")

    selected_columns = metric_columns
    if selected_columns is None:
        selected_columns = _default_metric_columns(frame)
    selected_columns = tuple(selected_columns)
    if not selected_columns:
        raise ValueError("No metric columns were selected for summarization.")

    rows = []
    for column in selected_columns:
        if column not in frame.columns:
            raise ValueError(f"metric_rows is missing metric column: {column}")
        row = {"metric": column}
        row.update(metric_summary(frame[column].tolist(), confidence=confidence))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_metric_samples(
    baseline_values,
    candidate_values,
    metric_name="metric",
    confidence=0.95,
    higher_is_better=True,
):
    baseline = _finite_values(baseline_values, "baseline_values")
    candidate = _finite_values(candidate_values, "candidate_values")
    if baseline.size != candidate.size:
        raise ValueError("baseline_values and candidate_values must have the same length.")

    confidence = _normalize_confidence(confidence)
    differences = candidate - baseline
    if higher_is_better:
        wins = int(np.sum(differences > 0))
        losses = int(np.sum(differences < 0))
    else:
        wins = int(np.sum(differences < 0))
        losses = int(np.sum(differences > 0))
    ties = int(np.sum(differences == 0))
    comparison_count = wins + losses

    if higher_is_better:
        conclusion = _comparison_conclusion(
            _quantile(differences, (1.0 - confidence) / 2.0),
            _quantile(differences, 1.0 - ((1.0 - confidence) / 2.0)),
        )
    else:
        conclusion = _comparison_conclusion(
            -_quantile(differences, 1.0 - ((1.0 - confidence) / 2.0)),
            -_quantile(differences, (1.0 - confidence) / 2.0),
        )

    return {
        "metric": str(metric_name),
        "count": int(baseline.size),
        "baseline_mean": float(np.mean(baseline)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_lower": _quantile(differences, (1.0 - confidence) / 2.0),
        "ci_upper": _quantile(differences, 1.0 - ((1.0 - confidence) / 2.0)),
        "confidence": confidence,
        "higher_is_better": bool(higher_is_better),
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "improvement_rate": wins / comparison_count if comparison_count else 0.0,
        "sign_test_p_value": _two_sided_sign_test_p_value(wins, losses),
        "conclusion": conclusion,
    }


def _normalize_count(items_or_count):
    count = int(items_or_count) if isinstance(items_or_count, int) else len(items_or_count)
    if count <= 0:
        raise ValueError("items_or_count must describe at least one item.")
    return count


def _rng(seed=None):
    return np.random.default_rng(seed)


def _normalize_optional_array(values, expected_length, name):
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape[0] != expected_length:
        raise ValueError(f"{name} must have length {expected_length}.")
    return array


def _partition_counts(total_count, train_size, validation_size, test_size):
    train_ratio = float(train_size)
    validation_ratio = float(validation_size)
    if test_size is None:
        test_ratio = 1.0 - train_ratio - validation_ratio
    else:
        test_ratio = float(test_size)

    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_size, validation_size, and test_size must sum to 1.0.")

    train_count = int(round(total_count * train_ratio))
    validation_count = int(round(total_count * validation_ratio))
    test_count = total_count - train_count - validation_count
    if test_count < 0:
        raise ValueError("Invalid split sizes for the requested item count.")
    return train_count, validation_count, test_count


def _ordered_indices(count, shuffle=True, seed=None):
    indices = np.arange(count, dtype=int)
    if shuffle:
        generator = _rng(seed)
        generator.shuffle(indices)
    return indices


def _stratified_single_split(count, labels, train_size, validation_size, test_size, shuffle=True, seed=None):
    generator = _rng(seed)
    train_parts = []
    validation_parts = []
    test_parts = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        if shuffle:
            generator.shuffle(label_indices)
        train_count, validation_count, _ = _partition_counts(
            len(label_indices),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
        train_end = train_count
        validation_end = train_end + validation_count
        train_parts.append(label_indices[:train_end])
        validation_parts.append(label_indices[train_end:validation_end])
        test_parts.append(label_indices[validation_end:])

    return (
        _concat_sorted(train_parts),
        _concat_sorted(validation_parts),
        _concat_sorted(test_parts),
    )


def _group_single_split(count, groups, train_size, validation_size, test_size, shuffle=True, seed=None):
    generator = _rng(seed)
    unique_groups = np.unique(groups)
    if shuffle:
        generator.shuffle(unique_groups)

    train_target, validation_target, _ = _partition_counts(
        count,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )

    train_indices = []
    validation_indices = []
    test_indices = []
    train_count = 0
    validation_count = 0

    for group in unique_groups:
        group_indices = np.where(groups == group)[0]
        if train_count < train_target:
            train_indices.extend(group_indices.tolist())
            train_count += len(group_indices)
            continue
        if validation_count < validation_target:
            validation_indices.extend(group_indices.tolist())
            validation_count += len(group_indices)
            continue
        test_indices.extend(group_indices.tolist())

    return (
        np.asarray(sorted(train_indices), dtype=int),
        np.asarray(sorted(validation_indices), dtype=int),
        np.asarray(sorted(test_indices), dtype=int),
    )


def _folds_from_labels_or_groups(count, groups=None, labels=None, n_splits=5, shuffle=True, seed=None):
    groups = _normalize_optional_array(groups, count, "groups")
    labels = _normalize_optional_array(labels, count, "labels")
    if groups is not None and labels is not None:
        raise ValueError("kfold_splits supports labels or groups, but not both at once.")

    generator = _rng(seed)
    if groups is not None:
        unique_groups = np.unique(groups)
        if shuffle:
            generator.shuffle(unique_groups)
        group_sizes = {group: int(np.sum(groups == group)) for group in unique_groups}
        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        for group in sorted(unique_groups, key=lambda item: (-group_sizes[item], str(item))):
            target = int(np.argmin(fold_sizes))
            fold_groups[target].append(group)
            fold_sizes[target] += group_sizes[group]
        return [np.where(np.isin(groups, fold_groups[index]))[0] for index in range(n_splits)]

    if labels is not None:
        folds = [[] for _ in range(n_splits)]
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            if shuffle:
                generator.shuffle(label_indices)
            for index, value in enumerate(label_indices.tolist()):
                folds[index % n_splits].append(value)
        return [np.asarray(sorted(values), dtype=int) for values in folds]

    indices = _ordered_indices(count, shuffle=shuffle, seed=seed)
    return [np.asarray(values, dtype=int) for values in np.array_split(indices, n_splits)]


def _concat_sorted(parts):
    if not parts:
        return np.asarray([], dtype=int)
    return np.sort(np.concatenate(parts))


def _to_index_tuple(indices):
    return tuple(int(value) for value in np.asarray(indices, dtype=int).tolist())


def _finite_values(values, name):
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _normalize_confidence(confidence):
    value = float(confidence)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be between 0 and 1.")
    return value


def _quantile(values, q):
    return float(np.quantile(values, q))


def _default_metric_columns(frame):
    excluded = {
        "split",
        "split_method",
        "threshold",
        "k",
        "pair_count",
        "query_count",
        "result_count",
        "relevant_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "train_count",
        "validation_count",
        "test_count",
    }
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return tuple(columns)


def _comparison_conclusion(lower_improvement, upper_improvement):
    if lower_improvement > 0:
        return "candidate_better"
    if upper_improvement < 0:
        return "baseline_better"
    return "inconclusive"


def _two_sided_sign_test_p_value(wins, losses):
    trials = int(wins) + int(losses)
    if trials == 0:
        return 1.0
    successes = min(int(wins), int(losses))
    tail = sum(math.comb(trials, k) for k in range(successes + 1)) / (2**trials)
    return float(min(1.0, 2.0 * tail))
