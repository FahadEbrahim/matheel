import pytest

import matheel
from matheel.calibration import (
    calibrate_threshold,
    calibration_curve,
    evaluate_threshold,
    feature_score_range,
)


def test_evaluate_threshold_reports_confusion_counts_and_metrics():
    report = evaluate_threshold(
        [0.9, 0.8, 0.4, 0.2],
        [True, False, True, False],
        threshold=0.5,
    )

    assert report["threshold"] == 0.5
    assert report["true_positive"] == 1
    assert report["false_positive"] == 1
    assert report["true_negative"] == 1
    assert report["false_negative"] == 1
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)
    assert report["accuracy"] == pytest.approx(0.5)


def test_calibration_helpers_are_exported_from_package_root():
    assert matheel.calibrate_threshold is calibrate_threshold
    assert matheel.calibration_curve is calibration_curve
    assert matheel.evaluate_threshold is evaluate_threshold
    assert matheel.feature_score_range is feature_score_range


def test_calibrate_threshold_finds_best_cutoff_for_tiny_labeled_dataset():
    report = calibrate_threshold(
        [
            (0.95, True),
            (0.82, True),
            (0.61, False),
            (0.40, False),
        ]
    )

    assert report["threshold"] == pytest.approx(0.82)
    assert report["true_positive"] == 2
    assert report["false_positive"] == 0
    assert report["true_negative"] == 2
    assert report["false_negative"] == 0
    assert report["f1"] == pytest.approx(1.0)
    assert report["optimized_metric"] == "f1"


def test_calibrate_threshold_accepts_mapping_rows_and_custom_keys():
    report = calibrate_threshold(
        [
            {"similarity_score": 0.3, "label": "different"},
            {"similarity_score": 0.7, "label": "same"},
            {"similarity_score": 0.8, "label": "same"},
        ],
        score_key="similarity_score",
        label_key="label",
        positive_label="same",
    )

    assert report["threshold"] == pytest.approx(0.7)
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(1.0)


def test_calibration_curve_accepts_explicit_thresholds():
    curve = calibration_curve(
        [0.9, 0.6, 0.2],
        [True, False, False],
        thresholds=[0.5, 0.8],
    )

    assert [item["threshold"] for item in curve] == [0.5, 0.8]
    assert curve[0]["false_positive"] == 1
    assert curve[1]["false_positive"] == 0


def test_calibrate_threshold_supports_lower_scores_as_matches():
    report = calibrate_threshold(
        [0.1, 0.2, 0.8, 0.9],
        [True, True, False, False],
        greater_is_match=False,
    )

    assert report["threshold"] == pytest.approx(0.2)
    assert report["f1"] == pytest.approx(1.0)


def test_feature_score_range_documents_raw_and_normalized_scales():
    assert feature_score_range("semantic", similarity_function="cosine") == (-1.0, 1.0)
    assert feature_score_range("semantic", similarity_function="dot") == (
        float("-inf"),
        float("inf"),
    )
    assert feature_score_range("semantic", similarity_function="euclidean") == (
        float("-inf"),
        0.0,
    )
    assert feature_score_range(
        "semantic",
        similarity_function="euclidean",
        normalize_semantic_scores=True,
    ) == (0.0, 1.0)
    assert feature_score_range("levenshtein") == (0.0, 1.0)
    assert feature_score_range("jaro_winkler") == (0.0, 1.0)
    assert feature_score_range("winnowing") == (0.0, 1.0)
    assert feature_score_range("gst") == (0.0, 1.0)
    assert feature_score_range("code_metric") == (0.0, 1.0)


def test_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same length"):
        evaluate_threshold([0.1], [True, False])

    with pytest.raises(ValueError, match="finite"):
        evaluate_threshold([float("nan")], [True])

    with pytest.raises(ValueError, match="At least one"):
        calibrate_threshold([])

    with pytest.raises(ValueError, match="Unsupported feature"):
        feature_score_range("unknown")
