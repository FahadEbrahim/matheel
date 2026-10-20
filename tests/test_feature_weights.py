import pytest

from matheel.feature_weights import (
    available_default_features,
    default_feature_weights,
    normalize_feature_weights,
    resolve_feature_weights,
)
from matheel.lexical import available_lexical_metrics


def test_normalize_feature_weights_scales_to_one():
    normalized = normalize_feature_weights({"semantic": 3, "code_metric": 1})

    assert normalized == {"semantic": 0.75, "code_metric": 0.25}


def test_normalize_feature_weights_preserves_ratios_when_sum_overflows():
    normalized = normalize_feature_weights(
        {"semantic": 1e308, "levenshtein": 1e308, "code_metric": 5e307, "gst": 0.0}
    )

    assert normalized == pytest.approx(
        {"semantic": 0.4, "levenshtein": 0.4, "code_metric": 0.2, "gst": 0.0}
    )
    assert sum(normalized.values()) == pytest.approx(1.0)


def test_similarity_with_large_finite_weights_matches_equivalent_small_weights():
    from matheel.similarity import calculate_similarity

    large = calculate_similarity(
        "a = 1", "a = 1", feature_weights={"levenshtein": 1e308, "jaro_winkler": 1e308}
    )
    small = calculate_similarity(
        "a = 1", "a = 1", feature_weights={"levenshtein": 1.0, "jaro_winkler": 1.0}
    )

    assert large == pytest.approx(small)
    assert large == pytest.approx(1.0)


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
    assert available_default_features()[:6] == (
        "semantic",
        "levenshtein",
        "jaro_winkler",
        "winnowing",
        "gst",
        "code_metric",
    )
    assert set(available_default_features()) == {
        "semantic", "winnowing", "gst", "code_metric", *available_lexical_metrics()
    }


@pytest.mark.parametrize("name", available_lexical_metrics())
def test_feature_weights_accept_each_lexical_metric(name):
    assert resolve_feature_weights(f"{name}=3,semantic=1") == {name: 0.75, "semantic": 0.25}


def test_new_lexical_metrics_do_not_change_default_blend():
    assert default_feature_weights() == {
        "semantic": 0.7, "levenshtein": 0.3, "jaro_winkler": 0.0
    }
