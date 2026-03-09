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
