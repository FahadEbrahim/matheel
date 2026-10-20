import pytest
from rapidfuzz import distance, fuzz

import matheel
from matheel.lexical import available_lexical_metrics, lexical_metric_metadata, score_lexical_metric


DISTANCE_SCORERS = {
    "levenshtein": distance.Levenshtein,
    "jaro_winkler": distance.JaroWinkler,
    "damerau_levenshtein": distance.DamerauLevenshtein,
    "osa": distance.OSA,
    "hamming": distance.Hamming,
    "indel": distance.Indel,
    "lcs": distance.LCSseq,
    "jaro": distance.Jaro,
    "prefix": distance.Prefix,
    "postfix": distance.Postfix,
}
FUZZ_SCORERS = {
    "rapidfuzz_ratio": fuzz.ratio,
    "rapidfuzz_partial_ratio": fuzz.partial_ratio,
    "rapidfuzz_token_sort_ratio": fuzz.token_sort_ratio,
    "rapidfuzz_token_set_ratio": fuzz.token_set_ratio,
    "rapidfuzz_token_ratio": fuzz.token_ratio,
    "rapidfuzz_partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
    "rapidfuzz_partial_token_set_ratio": fuzz.partial_token_set_ratio,
    "rapidfuzz_partial_token_ratio": fuzz.partial_token_ratio,
    "rapidfuzz_wratio": fuzz.WRatio,
    "rapidfuzz_qratio": fuzz.QRatio,
}


def test_registry_covers_all_rapidfuzz_distance_families_and_scalar_fuzz_scorers():
    assert set(available_lexical_metrics()) == DISTANCE_SCORERS.keys() | FUZZ_SCORERS.keys()
    assert len(available_lexical_metrics()) == 20
    assert "rapidfuzz_partial_ratio_alignment" not in available_lexical_metrics()


@pytest.mark.parametrize("name", available_lexical_metrics())
@pytest.mark.parametrize(
    "left,right",
    [
        ("", ""),
        ("", "abc"),
        ("identical", "identical"),
        ("abc", "xyz"),
        ("CA", "ABC"),
        ("fuzzy fuzzy was a bear", "was a fuzzy bear"),
        ("def sum(α, β):\n    return α + β", "def sum(α, β):\n    return β + α"),
        ("CaseSensitive!", "casesensitive"),
    ],
)
def test_scores_match_rapidfuzz_on_normalized_scale(name, left, right):
    expected = (
        DISTANCE_SCORERS[name].normalized_similarity(left, right)
        if name in DISTANCE_SCORERS
        else FUZZ_SCORERS[name](left, right) / 100.0
    )

    actual = score_lexical_metric(name, left, right)

    assert actual == pytest.approx(expected)
    assert 0.0 <= actual <= 1.0


@pytest.mark.parametrize("name", available_lexical_metrics())
def test_metric_metadata_has_normalized_scale_and_returns_independent_values(name):
    metadata = lexical_metric_metadata(name)

    assert metadata["name"] == name
    assert metadata["label"]
    assert metadata["score_range"] == (0.0, 1.0)
    assert metadata["family"] == ("distance" if name in DISTANCE_SCORERS else "fuzz")
    metadata["label"] = "overwritten"
    assert lexical_metric_metadata(name)["label"] != "overwritten"


@pytest.mark.parametrize("name", ["unknown", "", None, "rapidfuzz_partial_ratio_alignment"])
def test_unknown_metric_fails_with_available_choices(name):
    with pytest.raises(ValueError, match="Unsupported lexical metric.*Supported lexical metrics"):
        score_lexical_metric(name, "left", "right")
    with pytest.raises(ValueError, match="Unsupported lexical metric"):
        lexical_metric_metadata(name)


def test_lcs_uses_subsequence_length_normalized_by_longer_input():
    assert score_lexical_metric("lcs", "abcdef", "ace") == pytest.approx(3 / 6)


def test_transposition_metrics_preserve_damerau_and_osa_difference():
    assert score_lexical_metric("damerau_levenshtein", "CA", "ABC") == pytest.approx(1 / 3)
    assert score_lexical_metric("osa", "CA", "ABC") == 0.0


def test_hamming_accepts_unequal_lengths_with_padding():
    assert score_lexical_metric("hamming", "abc", "ab") == pytest.approx(2 / 3)


def test_existing_lexical_options_are_forwarded():
    assert score_lexical_metric(
        "levenshtein", "abcd", "abxcd", levenshtein_weights=(3, 1, 1)
    ) == pytest.approx(distance.Levenshtein.normalized_similarity("abcd", "abxcd", weights=(3, 1, 1)))
    assert score_lexical_metric(
        "jaro_winkler", "prefixAlpha", "prefixBeta", jaro_winkler_prefix_weight=0.25
    ) == pytest.approx(
        distance.JaroWinkler.normalized_similarity("prefixAlpha", "prefixBeta", prefix_weight=0.25)
    )


def test_registry_functions_are_exported_from_package_root():
    assert matheel.available_lexical_metrics is available_lexical_metrics
    assert matheel.lexical_metric_metadata is lexical_metric_metadata
    assert matheel.score_lexical_metric is score_lexical_metric
