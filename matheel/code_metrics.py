import math
import re
from collections import Counter
from pathlib import Path


try:
    from codebleu import bleu as package_codebleu_bleu
    from codebleu.codebleu import PACKAGE_DIR as codebleu_package_dir
    from codebleu import weighted_ngram_match as package_weighted_ngram_match
except ImportError:  # pragma: no cover - optional dependency
    package_codebleu_bleu = None
    codebleu_package_dir = None
    package_weighted_ngram_match = None


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_CODE_LANGUAGES = ("java", "python", "c", "cpp")
_CODEBLEU_KEYWORD_CACHE = {}
_KEYWORDS = {
    "c": {
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
        "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
        "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
        "union", "unsigned", "void", "volatile", "while",
    },
    "java": {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
        "const", "continue", "default", "do", "double", "else", "enum", "extends", "false",
        "final", "finally", "float", "for", "if", "implements", "import", "instanceof", "int",
        "interface", "long", "native", "new", "null", "package", "private", "protected",
        "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized",
        "this", "throw", "throws", "transient", "true", "try", "void", "volatile", "while",
    },
    "python": {
        "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
        "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
        "while", "with", "yield",
    },
    "cpp": {
        "auto", "bool", "break", "case", "catch", "char", "class", "const", "continue",
        "default", "delete", "do", "double", "else", "enum", "false", "float", "for", "if",
        "include", "int", "long", "namespace", "new", "nullptr", "private", "protected",
        "public", "return", "short", "signed", "static", "struct", "switch", "template",
        "this", "throw", "true", "try", "typedef", "typename", "using", "void", "while",
    },
}


def available_code_metrics():
    return (
        "none",
        "codebleu",
        "codebleu_ngram",
        "codebleu_weighted_ngram",
        "codebleu_syntax",
        "codebleu_dataflow",
        "crystalbleu",
    )


def available_code_metric_languages():
    return _SUPPORTED_CODE_LANGUAGES


def normalize_code_language(language):
    key = (language or "java").strip().lower()
    if key in ("py", "python3"):
        return "python"
    if key in ("c++", "cc", "cxx", "hpp", "h++", "cpp"):
        return "cpp"
    if key in ("c", "h"):
        return "c"
    if key == "java":
        return "java"
    return key


def _load_package_keywords(language):
    if codebleu_package_dir is None:
        return None
    package_path = Path(codebleu_package_dir) / "keywords" / f"{language}.txt"
    if not package_path.exists():
        return None
    keywords = {
        line.strip()
        for line in package_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return keywords or None


def keyword_set_for_language(language):
    key = normalize_code_language(language)
    if key not in _SUPPORTED_CODE_LANGUAGES:
        supported = ", ".join(_SUPPORTED_CODE_LANGUAGES)
        raise ValueError(f"CodeBLEU-style metrics currently support: {supported}. Got: {language}")
    if key in _CODEBLEU_KEYWORD_CACHE:
        return _CODEBLEU_KEYWORD_CACHE[key]
    package_keywords = _load_package_keywords(key)
    if package_keywords is not None:
        _CODEBLEU_KEYWORD_CACHE[key] = package_keywords
        return package_keywords
    if key in ("c", "cpp") and "c" in _KEYWORDS and "cpp" in _KEYWORDS:
        merged = set(_KEYWORDS["c"]) | set(_KEYWORDS["cpp"])
        _CODEBLEU_KEYWORD_CACHE[key] = merged if key == "cpp" else set(_KEYWORDS["c"])
        if key == "cpp":
            return _CODEBLEU_KEYWORD_CACHE[key]
        return _CODEBLEU_KEYWORD_CACHE[key]
    _CODEBLEU_KEYWORD_CACHE[key] = set(_KEYWORDS[key])
    return _CODEBLEU_KEYWORD_CACHE[key]


def code_metric_language_supported(language):
    return normalize_code_language(language) in _SUPPORTED_CODE_LANGUAGES


def tokenize_for_code_metrics(text):
    return _TOKEN_RE.findall(text or "")


def ngram_tuples(tokens, n):
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def _brevity_penalty(reference_length, hypothesis_length):
    if hypothesis_length <= 0:
        return 0.0
    if hypothesis_length > reference_length:
        return 1.0
    return math.exp(1.0 - (reference_length / float(hypothesis_length)))


def _weighted_match_score(reference_tokens, hypothesis_tokens, weights, ignored_by_order=None, epsilon=1e-12):
    if not hypothesis_tokens:
        return 0.0

    precisions = []
    max_order = len(weights)
    for n in range(1, max_order + 1):
        hyp_counts = Counter(ngram_tuples(hypothesis_tokens, n))
        ref_counts = Counter(ngram_tuples(reference_tokens, n))
        ignored = set() if ignored_by_order is None else set(ignored_by_order.get(n, set()))
        if ignored:
            hyp_counts = Counter({ngram: count for ngram, count in hyp_counts.items() if ngram not in ignored})
            ref_counts = Counter({ngram: count for ngram, count in ref_counts.items() if ngram not in ignored})

        denom = sum(hyp_counts.values())
        if denom == 0:
            precisions.append(epsilon)
            continue

        numer = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in hyp_counts.items())
        if numer == 0:
            return 0.0
        precisions.append(max(epsilon, numer / float(denom)))

    brevity_penalty = _brevity_penalty(len(reference_tokens), len(hypothesis_tokens))
    return float(
        brevity_penalty * math.exp(sum(weight * math.log(precision) for weight, precision in zip(weights, precisions)))
    )


def _keyword_weighted_match_score(reference_tokens, hypothesis_tokens, keywords, weights, epsilon=1e-12):
    if not hypothesis_tokens:
        return 0.0

    if package_weighted_ngram_match is not None:
        references = [[reference_tokens]]
        hypotheses = [hypothesis_tokens]

        def token_weights(tokens):
            return {token: 1.0 if token in keywords else 0.2 for token in tokens}

        refs_with_weights = [
            [[tokens, token_weights(tokens)] for tokens in reference]
            for reference in references
        ]
        return float(package_weighted_ngram_match.corpus_bleu(refs_with_weights, hypotheses))

    precisions = []
    max_order = len(weights)
    for n in range(1, max_order + 1):
        hyp_counts = Counter(ngram_tuples(hypothesis_tokens, n))
        ref_counts = Counter(ngram_tuples(reference_tokens, n))
        if not hyp_counts:
            precisions.append(epsilon)
            continue

        def ngram_weight(ngram):
            token_weights = [1.0 if token in keywords else 0.2 for token in ngram]
            return sum(token_weights) / float(len(token_weights))

        denom = sum(count * ngram_weight(ngram) for ngram, count in hyp_counts.items())
        if denom <= 0:
            precisions.append(epsilon)
            continue

        numer = 0.0
        for ngram, count in hyp_counts.items():
            numer += min(count, ref_counts.get(ngram, 0)) * ngram_weight(ngram)
        if numer <= 0:
            return 0.0
        precisions.append(max(epsilon, numer / float(denom)))

    brevity_penalty = _brevity_penalty(len(reference_tokens), len(hypothesis_tokens))
    return float(
        brevity_penalty * math.exp(sum(weight * math.log(precision) for weight, precision in zip(weights, precisions)))
    )


def _syntax_shape_tokens(tokens, keywords):
    shapes = []
    for token in tokens:
        if token in keywords:
            shapes.append(f"kw:{token}")
        elif token.isdigit():
            shapes.append("num")
        elif _IDENTIFIER_RE.match(token):
            shapes.append("id")
        else:
            shapes.append(f"sym:{token}")
    return shapes


def _identifier_overlap_score(reference_tokens, hypothesis_tokens, keywords):
    ref_ids = [token for token in reference_tokens if _IDENTIFIER_RE.match(token) and token not in keywords]
    hyp_ids = [token for token in hypothesis_tokens if _IDENTIFIER_RE.match(token) and token not in keywords]
    if not ref_ids and not hyp_ids:
        return 1.0
    if not ref_ids or not hyp_ids:
        return 0.0

    ref_counts = Counter(ref_ids)
    hyp_counts = Counter(hyp_ids)
    overlap = sum(min(count, hyp_counts.get(token, 0)) for token, count in ref_counts.items())
    if overlap <= 0:
        return 0.0

    precision = overlap / float(sum(hyp_counts.values()))
    recall = overlap / float(sum(ref_counts.values()))
    if precision + recall <= 0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def parse_component_weights(raw_weights=None):
    if raw_weights in (None, ""):
        return (0.25, 0.25, 0.25, 0.25)
    if isinstance(raw_weights, (tuple, list)):
        values = [float(value) for value in raw_weights]
    else:
        values = [float(value.strip()) for value in str(raw_weights).split(",") if value.strip()]
    if len(values) != 4:
        raise ValueError("CodeBLEU component weights must contain exactly 4 values.")
    total = sum(values)
    if total <= 0:
        raise ValueError("CodeBLEU component weights must sum to a positive value.")
    return tuple(value / total for value in values)


def codebleu_components(reference, prediction, language="java", component_weights=None):
    reference_tokens = tokenize_for_code_metrics(reference)
    hypothesis_tokens = tokenize_for_code_metrics(prediction)
    if not reference_tokens and not hypothesis_tokens:
        return {
            "codebleu": 1.0,
            "codebleu_ngram": 1.0,
            "codebleu_weighted_ngram": 1.0,
            "codebleu_syntax": 1.0,
            "codebleu_dataflow": 1.0,
        }

    weights = parse_component_weights(component_weights)
    bleu_weights = (0.25, 0.25, 0.25, 0.25)
    keywords = keyword_set_for_language(language)

    if package_codebleu_bleu is not None:
        ngram_score = float(package_codebleu_bleu.corpus_bleu([[reference_tokens]], [hypothesis_tokens]))
    else:
        ngram_score = _weighted_match_score(reference_tokens, hypothesis_tokens, bleu_weights)

    weighted_ngram_score = _keyword_weighted_match_score(reference_tokens, hypothesis_tokens, keywords, bleu_weights)
    syntax_score = _weighted_match_score(
        _syntax_shape_tokens(reference_tokens, keywords),
        _syntax_shape_tokens(hypothesis_tokens, keywords),
        bleu_weights,
    )
    dataflow_score = _identifier_overlap_score(reference_tokens, hypothesis_tokens, keywords)

    alpha, beta, gamma, theta = weights
    combined = float(
        (alpha * ngram_score)
        + (beta * weighted_ngram_score)
        + (gamma * syntax_score)
        + (theta * dataflow_score)
    )

    return {
        "codebleu": combined,
        "codebleu_ngram": float(ngram_score),
        "codebleu_weighted_ngram": float(weighted_ngram_score),
        "codebleu_syntax": float(syntax_score),
        "codebleu_dataflow": float(dataflow_score),
    }


def prepare_crystalbleu_context(codes, max_order=4):
    order_limit = max(1, min(int(max_order), 8))
    tokens_by_doc = [tokenize_for_code_metrics(code) for code in codes]
    sorted_ngrams_by_n = {}

    for n in range(1, order_limit + 1):
        counts = Counter()
        for tokens in tokens_by_doc:
            counts.update(ngram_tuples(tokens, n))
        sorted_ngrams_by_n[n] = [ngram for ngram, _ in counts.most_common()]

    return {
        "tokens_by_doc": tokens_by_doc,
        "sorted_ngrams_by_n": sorted_ngrams_by_n,
        "max_order": order_limit,
    }


def _pair_crystalbleu_score(
    reference_tokens,
    hypothesis_tokens,
    sorted_ngrams_by_n,
    max_order=4,
    trivial_ngram_count=500,
):
    order_limit = max(1, min(int(max_order), 8))
    ignored_by_order = {}
    for n in range(1, order_limit + 1):
        ignored_by_order[n] = set(sorted_ngrams_by_n.get(n, [])[: max(0, int(trivial_ngram_count))])
    weights = tuple([1.0 / float(order_limit)] * order_limit)
    return _weighted_match_score(
        reference_tokens,
        hypothesis_tokens,
        weights=weights,
        ignored_by_order=ignored_by_order,
    )


def score_code_metric_pair(
    reference,
    prediction,
    metric_name="none",
    language="java",
    bidirectional=False,
    component_weights=None,
    crystalbleu_context=None,
    reference_index=None,
    prediction_index=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=50,
):
    metric_key = (metric_name or "none").strip().lower()
    if metric_key in ("none", ""):
        return 0.0

    if metric_key.startswith("codebleu"):
        language = normalize_code_language(language)
        forward = codebleu_components(
            reference,
            prediction,
            language=language,
            component_weights=component_weights,
        )[metric_key]
        if not bidirectional:
            return float(forward)
        reverse = codebleu_components(
            prediction,
            reference,
            language=language,
            component_weights=component_weights,
        )[metric_key]
        return float((forward + reverse) * 0.5)

    if metric_key == "crystalbleu":
        if crystalbleu_context is None:
            crystalbleu_context = prepare_crystalbleu_context(
                [reference, prediction],
                max_order=crystalbleu_max_order,
            )
            left_index = 0
            right_index = 1
        else:
            left_index = int(reference_index)
            right_index = int(prediction_index)

        tokens_by_doc = crystalbleu_context["tokens_by_doc"]
        sorted_ngrams_by_n = crystalbleu_context["sorted_ngrams_by_n"]
        forward = _pair_crystalbleu_score(
            tokens_by_doc[left_index],
            tokens_by_doc[right_index],
            sorted_ngrams_by_n=sorted_ngrams_by_n,
            max_order=crystalbleu_max_order,
            trivial_ngram_count=crystalbleu_trivial_ngram_count,
        )
        if not bidirectional:
            return float(forward)
        reverse = _pair_crystalbleu_score(
            tokens_by_doc[right_index],
            tokens_by_doc[left_index],
            sorted_ngrams_by_n=sorted_ngrams_by_n,
            max_order=crystalbleu_max_order,
            trivial_ngram_count=crystalbleu_trivial_ngram_count,
        )
        return float((forward + reverse) * 0.5)

    raise ValueError(f"Unsupported code metric: {metric_name}")
