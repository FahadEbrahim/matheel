import pytest

from matheel.feature_weights import (
    available_default_features,
    default_feature_weights,
    normalize_feature_weights,
    resolve_feature_weights,
)


def test_normalize_feature_weights_scales_to_one():
    normalized = normalize_feature_weights({"semantic": 3, "code_metric": 1})

    assert normalized == {"semantic": 0.75, "code_metric": 0.25}


def test_resolve_feature_weights_accepts_string_overrides():
    resolved = resolve_feature_weights(feature_weights="semantic=2,code_metric=2")

    assert resolved == {"semantic": 0.5, "code_metric": 0.5}


@pytest.mark.parametrize(
    "feature_weights",
    [
        {"semantic": float("nan")},
        {"semantic": float("inf")},
        "semantic=nan",
        "semantic=inf",
        ["semantic=nan"],
    ],
)
def test_resolve_feature_weights_rejects_non_finite_values(feature_weights):
    with pytest.raises(ValueError, match="finite"):
        resolve_feature_weights(feature_weights=feature_weights)


@pytest.mark.parametrize("code_metric_weight", [float("nan"), float("inf")])
def test_default_feature_weights_rejects_non_finite_code_metric_weight(code_metric_weight):
    with pytest.raises(ValueError, match="finite"):
        default_feature_weights(code_metric_weight=code_metric_weight)


@pytest.mark.parametrize(
    "feature_weights",
    [
        {"levenshten": 1.0},
        "levenshten=1.0",
        ["semantic=0.5", "levenshten=0.5"],
        [("semantic", 0.5), ("levenshten", 0.5)],
    ],
)
def test_resolve_feature_weights_rejects_unknown_feature_names(feature_weights):
    with pytest.raises(ValueError, match="Unsupported feature weight.*levenshten.*levenshtein"):
        resolve_feature_weights(feature_weights=feature_weights)


def test_normalize_feature_weights_rejects_unknown_feature_names():
    with pytest.raises(ValueError, match="Unsupported feature weight.*unknown"):
        normalize_feature_weights({"semantic": 0.5, "unknown": 0.5})


def test_resolve_feature_weights_uses_defaults_without_legacy_weights():
    resolved = resolve_feature_weights(feature_weights=None)

    assert resolved == default_feature_weights()


def test_available_default_features_includes_new_lexical_baselines():
    assert available_default_features() == (
        "semantic",
        "levenshtein",
        "jaro_winkler",
        "winnowing",
        "gst",
        "code_metric",
    )
