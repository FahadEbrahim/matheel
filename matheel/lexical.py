"""RapidFuzz's scalar similarities on a common zero-to-one score scale.

Distance families use ``normalized_similarity``; ``fuzz`` scorers are divided
by 100. Strings are compared as supplied, without RapidFuzz preprocessing.
Empty-input behavior is preserved: for example, ``rapidfuzz_ratio('', '')`` is
1, while ``rapidfuzz_qratio('', '')`` and ``rapidfuzz_wratio('', '')`` are 0.
Hamming pads unequal inputs. ``lcs`` means longest common subsequence, not
longest common substring. Alignment/edit-operation APIs are not scalar metrics.
"""

from rapidfuzz import fuzz
from rapidfuzz.distance import (
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    OSA,
    Postfix,
    Prefix,
)


# name: (display label, family, scorer, divisor)
_LEXICAL_METRICS = {
    "levenshtein": ("Levenshtein", "distance", Levenshtein.normalized_similarity, 1.0),
    "jaro_winkler": ("Jaro-Winkler", "distance", JaroWinkler.normalized_similarity, 1.0),
    "damerau_levenshtein": (
        "Damerau-Levenshtein", "distance", DamerauLevenshtein.normalized_similarity, 1.0
    ),
    "osa": ("Optimal String Alignment (OSA)", "distance", OSA.normalized_similarity, 1.0),
    "hamming": ("Hamming", "distance", Hamming.normalized_similarity, 1.0),
    "indel": ("Indel", "distance", Indel.normalized_similarity, 1.0),
    "lcs": ("Longest Common Subsequence (LCS)", "distance", LCSseq.normalized_similarity, 1.0),
    "jaro": ("Jaro", "distance", Jaro.normalized_similarity, 1.0),
    "prefix": ("Common Prefix", "distance", Prefix.normalized_similarity, 1.0),
    "postfix": ("Common Postfix", "distance", Postfix.normalized_similarity, 1.0),
    "rapidfuzz_ratio": ("RapidFuzz Ratio", "fuzz", fuzz.ratio, 100.0),
    "rapidfuzz_partial_ratio": ("RapidFuzz Partial Ratio", "fuzz", fuzz.partial_ratio, 100.0),
    "rapidfuzz_token_sort_ratio": (
        "RapidFuzz Token Sort Ratio", "fuzz", fuzz.token_sort_ratio, 100.0
    ),
    "rapidfuzz_token_set_ratio": (
        "RapidFuzz Token Set Ratio", "fuzz", fuzz.token_set_ratio, 100.0
    ),
    "rapidfuzz_token_ratio": ("RapidFuzz Token Ratio", "fuzz", fuzz.token_ratio, 100.0),
    "rapidfuzz_partial_token_sort_ratio": (
        "RapidFuzz Partial Token Sort Ratio", "fuzz", fuzz.partial_token_sort_ratio, 100.0
    ),
    "rapidfuzz_partial_token_set_ratio": (
        "RapidFuzz Partial Token Set Ratio", "fuzz", fuzz.partial_token_set_ratio, 100.0
    ),
    "rapidfuzz_partial_token_ratio": (
        "RapidFuzz Partial Token Ratio", "fuzz", fuzz.partial_token_ratio, 100.0
    ),
    "rapidfuzz_wratio": ("RapidFuzz WRatio", "fuzz", fuzz.WRatio, 100.0),
    "rapidfuzz_qratio": ("RapidFuzz QRatio", "fuzz", fuzz.QRatio, 100.0),
}


def available_lexical_metrics():
    """Return the canonical feature-weight names of all scalar lexical metrics."""
    return tuple(_LEXICAL_METRICS)


def _resolve_lexical_metric(name):
    key = str(name or "").strip().lower()
    if key not in _LEXICAL_METRICS:
        supported = ", ".join(available_lexical_metrics())
        raise ValueError(f"Unsupported lexical metric: {name}. Supported lexical metrics: {supported}.")
    return key, _LEXICAL_METRICS[key]


def lexical_metric_metadata(name):
    """Describe a metric without exposing or modifying its registered scorer."""
    key, (label, family, _, _) = _resolve_lexical_metric(name)
    return {"name": key, "label": label, "family": family, "score_range": (0.0, 1.0)}


def score_lexical_metric(
    name,
    left,
    right,
    *,
    levenshtein_weights=(1, 1, 1),
    jaro_winkler_prefix_weight=0.1,
):
    """Score strings using a registered metric, returning similarity in [0, 1].

    Levenshtein's weights are insertion, deletion, and substitution costs;
    Jaro-Winkler accepts its prefix weight. Other metrics use RapidFuzz defaults.
    ``rapidfuzz_*token*`` scorers split on whitespace rather than Matheel's
    code-aware lexical tokenizer, and may ignore token order or repetitions.
    """
    key, (_, _, scorer, divisor) = _resolve_lexical_metric(name)
    options = {}
    if key == "levenshtein":
        options["weights"] = levenshtein_weights
    elif key == "jaro_winkler":
        options["prefix_weight"] = jaro_winkler_prefix_weight
    elif key == "hamming":
        options["pad"] = True
    return float(scorer(left, right, **options) / divisor)
