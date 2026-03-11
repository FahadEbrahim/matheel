import math
import re
import time
from collections import Counter
from pathlib import Path


try:
    from codebleu import bleu as package_codebleu_bleu
    from codebleu.codebleu import PACKAGE_DIR as codebleu_package_dir
    from codebleu import weighted_ngram_match as package_weighted_ngram_match
    from codebleu.utils import get_tree_sitter_language as codebleu_get_tree_sitter_language
except ImportError:  # pragma: no cover - optional dependency
    package_codebleu_bleu = None
    codebleu_package_dir = None
    package_weighted_ngram_match = None
    codebleu_get_tree_sitter_language = None

try:
    from bert_score import BERTScorer
except ImportError:  # pragma: no cover - optional dependency
    BERTScorer = None

try:
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - optional dependency
    AutoConfig = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from apted import APTED
except ImportError:  # pragma: no cover - optional dependency
    APTED = None

try:
    from apted import Config as APTEDConfig
except ImportError:  # pragma: no cover - optional dependency
    APTEDConfig = None

try:
    from apted import PerEditOperationConfig
except ImportError:  # pragma: no cover - optional dependency
    PerEditOperationConfig = None

try:
    from tree_sitter import Parser as TreeSitterParser
except ImportError:  # pragma: no cover - optional dependency
    TreeSitterParser = None

try:
    from tree_sitter_language_pack import get_parser as get_tree_sitter_parser
except ImportError:  # pragma: no cover - optional dependency
    get_tree_sitter_parser = None

try:
    import networkx as nx
    from networkx.algorithms.similarity import optimize_graph_edit_distance
except ImportError:  # pragma: no cover - optional dependency
    nx = None
    optimize_graph_edit_distance = None

try:
    from func_timeout import FunctionTimedOut, func_timeout
except ImportError:  # pragma: no cover - optional dependency
    FunctionTimedOut = None
    func_timeout = None


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUBY_TRANX_SPLIT_RE = re.compile(r"([^A-Za-z0-9_])")
_RUBY_TRANX_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_SUPPORTED_CODE_LANGUAGES = ("java", "python", "c", "cpp")
_CODEBLEU_KEYWORD_CACHE = {}
_TREE_SITTER_PARSER_CACHE = {}
_CODEBERTSCORE_SCORER_CACHE = {}
_CODEBERTSCORE_LAYER_CACHE = {}
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


class _TSEDNode:
    def __init__(self, name, path):
        self.id = path
        self.name = name
        self.children = []
        self.len_sexp = 0

    def add_child(self, node):
        self.children.append(node)


def available_code_metrics():
    return (
        "none",
        "codebleu",
        "codebleu_ngram",
        "codebleu_weighted_ngram",
        "codebleu_syntax",
        "codebleu_dataflow",
        "crystalbleu",
        "ruby",
        "tsed",
        "codebertscore",
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
        raise ValueError(f"Code metrics currently support: {supported}. Got: {language}")
    if key in _CODEBLEU_KEYWORD_CACHE:
        return _CODEBLEU_KEYWORD_CACHE[key]
    package_keywords = _load_package_keywords(key)
    if package_keywords is not None:
        _CODEBLEU_KEYWORD_CACHE[key] = package_keywords
        return package_keywords
    if key in ("c", "cpp"):
        _CODEBLEU_KEYWORD_CACHE["c"] = set(_KEYWORDS["c"])
        _CODEBLEU_KEYWORD_CACHE["cpp"] = set(_KEYWORDS["c"]) | set(_KEYWORDS["cpp"])
        return _CODEBLEU_KEYWORD_CACHE[key]
    _CODEBLEU_KEYWORD_CACHE[key] = set(_KEYWORDS[key])
    return _CODEBLEU_KEYWORD_CACHE[key]


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


def _ruby_ngram_f1(reference_counts, hypothesis_counts, epsilon=1e-12):
    ref_total = int(sum(reference_counts.values()))
    hyp_total = int(sum(hypothesis_counts.values()))
    if ref_total <= 0 and hyp_total <= 0:
        return 1.0
    if ref_total <= 0 or hyp_total <= 0:
        return 0.0
    overlap = 0
    for ngram, count in reference_counts.items():
        overlap += min(count, int(hypothesis_counts.get(ngram, 0)))
    if overlap <= 0:
        return 0.0
    precision = float(overlap) / float(hyp_total)
    recall = float(overlap) / float(ref_total)
    denom = precision + recall
    if denom <= epsilon:
        return 0.0
    return float((2.0 * precision * recall) / denom)


def _ruby_normalize_mode(mode):
    normalized = (mode or "auto").strip().lower()
    if normalized in {"auto", "graph", "tree", "string", "ngram"}:
        return normalized
    raise ValueError(f"Unsupported ruby_mode: {mode}")


def _ruby_normalize_tokenizer(tokenizer):
    normalized = (tokenizer or "tranx").strip().lower()
    if normalized in {"tranx", "regex"}:
        return normalized
    raise ValueError(f"Unsupported ruby_tokenizer: {tokenizer}")


def _ruby_normalize_denominator(denominator):
    normalized = (denominator or "max").strip().lower()
    if normalized in {"max", "mean"}:
        return normalized
    raise ValueError(f"Unsupported ruby_denominator: {denominator}")


def _tokenize_tranx(text):
    value = _RUBY_TRANX_SPLIT_RE.sub(r" \1 ", text or "")
    value = _RUBY_TRANX_CAMEL_RE.sub(r"\1 \2", value)
    value = re.sub(r"\s+", " ", value)
    value = value.replace('"', "`").replace("'", "`")
    return [token for token in value.split(" ") if token]


def _tokenize_for_ruby(text, tokenizer="tranx"):
    if _ruby_normalize_tokenizer(tokenizer) == "regex":
        return tokenize_for_code_metrics(text)
    return _tokenize_tranx(text)


def _token_levenshtein_distance(left_tokens, right_tokens):
    if left_tokens == right_tokens:
        return 0
    if not left_tokens:
        return len(right_tokens)
    if not right_tokens:
        return len(left_tokens)
    previous = list(range(len(right_tokens) + 1))
    for row, left_token in enumerate(left_tokens, start=1):
        current = [row]
        for column, right_token in enumerate(right_tokens, start=1):
            insert_cost = current[column - 1] + 1
            delete_cost = previous[column] + 1
            replace_cost = previous[column - 1] + (0 if left_token == right_token else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return int(previous[-1])


def _ruby_string_similarity_from_tokens(left_tokens, right_tokens, denominator="max"):
    if not left_tokens and not right_tokens:
        return 1.0
    distance = float(_token_levenshtein_distance(left_tokens, right_tokens))
    denominator_mode = _ruby_normalize_denominator(denominator)
    if denominator_mode == "mean":
        denom = max((len(left_tokens) + len(right_tokens)) / 2.0, 1.0)
    else:
        denom = max(float(len(left_tokens)), float(len(right_tokens)), 1.0)
    score = 1.0 - (distance / float(denom))
    return float(max(0.0, min(1.0, score)))


def _ruby_flatten_tree_labels(root):
    if root is None:
        return []
    labels = []
    stack = [root]
    while stack:
        current = stack.pop()
        labels.append(str(getattr(current, "name", "unknown")))
        children = list(getattr(current, "children", []))
        for child in reversed(children):
            stack.append(child)
    return labels


def _ruby_tree_similarity(reference_tree, prediction_tree):
    if reference_tree is None or prediction_tree is None:
        return None
    if APTED is not None:
        return _tsed_pair_score(
            reference_tree,
            prediction_tree,
            delete_cost=1.0,
            insert_cost=1.0,
            rename_cost=1.0,
        )
    left_labels = _ruby_flatten_tree_labels(reference_tree)
    right_labels = _ruby_flatten_tree_labels(prediction_tree)
    return _ruby_string_similarity_from_tokens(left_labels, right_labels, denominator="max")


def _ruby_tree_to_graph(root, include_leaf_edges=True):
    if root is None or nx is None:
        return None
    graph = nx.DiGraph()
    node_index = 0
    leaves = []

    def add_node(label):
        nonlocal node_index
        current = node_index
        graph.add_node(current, label=str(label))
        node_index += 1
        return current

    def traverse(node, parent=None):
        current_idx = add_node(getattr(node, "name", "unknown"))
        children = list(getattr(node, "children", []))
        if parent is not None:
            graph.add_edge(parent, current_idx, label="AST")
        if not children:
            leaves.append(current_idx)
            return
        for child in children:
            traverse(child, parent=current_idx)

    traverse(root, parent=None)
    if include_leaf_edges:
        for left, right in zip(leaves, leaves[1:]):
            graph.add_edge(left, right, label="NEXT")
    return graph


def _ruby_graph_similarity(
    reference_tree,
    prediction_tree,
    timeout_seconds=1.0,
    use_edge_cost=True,
    include_leaf_edges=True,
):
    if (
        reference_tree is None
        or prediction_tree is None
        or nx is None
        or optimize_graph_edit_distance is None
    ):
        return None

    left_graph = _ruby_tree_to_graph(reference_tree, include_leaf_edges=include_leaf_edges)
    right_graph = _ruby_tree_to_graph(prediction_tree, include_leaf_edges=include_leaf_edges)
    if left_graph is None or right_graph is None:
        return None

    total_size = left_graph.number_of_nodes() + right_graph.number_of_nodes()
    if use_edge_cost:
        total_size += left_graph.number_of_edges() + right_graph.number_of_edges()
    if total_size <= 0:
        return 1.0

    try:
        ged_generator = optimize_graph_edit_distance(
            left_graph,
            right_graph,
            node_match=lambda left, right: left.get("label") == right.get("label"),
            edge_match=lambda left, right: left.get("label") == right.get("label"),
            edge_ins_cost=(lambda _edge: 1 if use_edge_cost else 0),
            edge_del_cost=(lambda _edge: 1 if use_edge_cost else 0),
        )
    except Exception:
        return None

    best_distance = float(total_size + 1)
    parsed_timeout = max(0.0, float(timeout_seconds))
    if func_timeout is not None and FunctionTimedOut is not None and parsed_timeout > 0:
        while True:
            try:
                candidate = func_timeout(parsed_timeout, next, args=(ged_generator,))
            except (FunctionTimedOut, StopIteration):
                break
            except Exception:
                break
            best_distance = min(best_distance, float(candidate))
    else:
        start = time.monotonic()
        while True:
            if parsed_timeout > 0 and (time.monotonic() - start) > parsed_timeout:
                break
            try:
                candidate = next(ged_generator)
            except StopIteration:
                break
            except Exception:
                break
            best_distance = min(best_distance, float(candidate))

    if best_distance > float(total_size):
        return None
    score = 1.0 - (best_distance / float(total_size))
    return float(max(0.0, min(1.0, score)))


def _ruby_pair_score(
    reference_tokens,
    hypothesis_tokens,
    max_order=4,
    epsilon=1e-12,
):
    if not reference_tokens and not hypothesis_tokens:
        return 1.0
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    order_limit = max(1, min(int(max_order), 8))
    order_scores = []
    for n in range(1, order_limit + 1):
        ref_counts = Counter(ngram_tuples(reference_tokens, n))
        hyp_counts = Counter(ngram_tuples(hypothesis_tokens, n))
        if (not ref_counts) and (not hyp_counts):
            continue
        order_scores.append(_ruby_ngram_f1(ref_counts, hyp_counts, epsilon=epsilon))

    if not order_scores:
        return 0.0
    return float(sum(order_scores) / float(len(order_scores)))


def prepare_ruby_context(
    codes,
    language="java",
    max_order=4,
    tokenizer="tranx",
    tree_max_nodes=180,
    tree_max_depth=10,
    tree_max_children=8,
    graph_include_leaf_edges=True,
):
    order_limit = max(1, min(int(max_order), 8))
    tokenizer_name = _ruby_normalize_tokenizer(tokenizer)
    normalized_language = normalize_code_language(language)
    tokens_by_doc = [tokenize_for_code_metrics(code) for code in codes]
    tranx_tokens_by_doc = [_tokenize_tranx(code) for code in codes]
    selected_tokens_by_doc = (
        tranx_tokens_by_doc if tokenizer_name == "tranx" else [list(tokens) for tokens in tokens_by_doc]
    )
    ngram_counts_by_order = {}
    for n in range(1, order_limit + 1):
        ngram_counts_by_order[n] = [Counter(ngram_tuples(tokens, n)) for tokens in tokens_by_doc]
    trees = [
        _tsed_get_tree(
            normalized_language,
            code,
            max_nodes=tree_max_nodes,
            max_depth=tree_max_depth,
            max_children=tree_max_children,
        )
        for code in codes
    ]
    return {
        "language": normalized_language,
        "tokens_by_doc": tokens_by_doc,
        "tranx_tokens_by_doc": tranx_tokens_by_doc,
        "selected_tokens_by_doc": selected_tokens_by_doc,
        "ngram_counts_by_order": ngram_counts_by_order,
        "max_order": order_limit,
        "trees": trees,
        "tree_max_nodes": max(1, int(tree_max_nodes)),
        "tree_max_depth": max(1, int(tree_max_depth)),
        "tree_max_children": max(1, int(tree_max_children)),
        "graph_include_leaf_edges": bool(graph_include_leaf_edges),
        "pair_cache": {},
    }


def _tsed_runtime_available():
    has_parser = (
        get_tree_sitter_parser is not None
        or (TreeSitterParser is not None and codebleu_get_tree_sitter_language is not None)
    )
    return APTED is not None and has_parser


def _resolve_tsed_parser(language):
    normalized_language = normalize_code_language(language)
    if normalized_language in _TREE_SITTER_PARSER_CACHE:
        return _TREE_SITTER_PARSER_CACHE[normalized_language]

    parser = None
    if get_tree_sitter_parser is not None:
        try:
            parser = get_tree_sitter_parser(normalized_language)
        except Exception:
            parser = None

    if parser is None and TreeSitterParser is not None and codebleu_get_tree_sitter_language is not None:
        try:
            ts_language = codebleu_get_tree_sitter_language(normalized_language)
            try:
                parser = TreeSitterParser(ts_language)
            except TypeError:
                parser = TreeSitterParser()
                parser.language = ts_language
        except Exception:
            parser = None

    _TREE_SITTER_PARSER_CACHE[normalized_language] = parser
    return parser


def _tsed_from_tree_sitter_node(ts_node, depth, path, remaining_budget, max_depth, max_children):
    out = _TSEDNode(getattr(ts_node, "type", "unknown"), path)
    total = 1
    if depth >= max_depth or remaining_budget[0] <= 0:
        return out, total

    children = list(getattr(ts_node, "children", []))
    if len(children) > max_children:
        step = max(1, len(children) // max_children)
        children = children[::step][:max_children]

    for index, child in enumerate(children):
        if remaining_budget[0] <= 0:
            break
        remaining_budget[0] -= 1
        child_node, child_count = _tsed_from_tree_sitter_node(
            child,
            depth=depth + 1,
            path=f"{path}.{index}",
            remaining_budget=remaining_budget,
            max_depth=max_depth,
            max_children=max_children,
        )
        out.add_child(child_node)
        total += child_count
    return out, total


def _tsed_get_tree(language, code, max_nodes=180, max_depth=10, max_children=8):
    parser = _resolve_tsed_parser(language)
    if parser is None:
        return None
    try:
        tree = parser.parse(bytes(code or "", encoding="utf-8"))
    except Exception:
        return None
    root = getattr(tree, "root_node", None)
    if root is None:
        return None

    remaining_budget = [max(int(max_nodes) - 1, 0)]
    tree_node, node_count = _tsed_from_tree_sitter_node(
        root,
        depth=0,
        path="0",
        remaining_budget=remaining_budget,
        max_depth=max(1, int(max_depth)),
        max_children=max(1, int(max_children)),
    )
    tree_node.len_sexp = node_count
    return tree_node


def _build_apted_config(delete_cost, insert_cost, rename_cost):
    if PerEditOperationConfig is not None:
        return PerEditOperationConfig(delete_cost, insert_cost, rename_cost)
    if APTEDConfig is None:
        return None

    class _WeightedAPTEDConfig(APTEDConfig):
        def delete(self, _node):
            return float(delete_cost)

        def insert(self, _node):
            return float(insert_cost)

        def rename(self, left, right):
            return 0.0 if getattr(left, "name", "") == getattr(right, "name", "") else float(rename_cost)

        def children(self, node):
            return getattr(node, "children", [])

    return _WeightedAPTEDConfig()


def prepare_tsed_context(codes, language="java", max_nodes=180, max_depth=10, max_children=8):
    if not _tsed_runtime_available():
        raise ImportError(
            "TSED requires optional dependencies. Install with: pip install apted tree-sitter-language-pack"
        )
    normalized_language = normalize_code_language(language)
    if normalized_language not in _SUPPORTED_CODE_LANGUAGES:
        supported = ", ".join(_SUPPORTED_CODE_LANGUAGES)
        raise ValueError(f"TSED currently supports: {supported}. Got: {language}")
    trees = [
        _tsed_get_tree(
            normalized_language,
            code,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_children=max_children,
        )
        for code in codes
    ]
    return {
        "language": normalized_language,
        "trees": trees,
        "max_nodes": max(1, int(max_nodes)),
        "max_depth": max(1, int(max_depth)),
        "max_children": max(1, int(max_children)),
    }


def _tsed_pair_score(reference_tree, prediction_tree, delete_cost=1.0, insert_cost=1.0, rename_cost=1.0):
    if reference_tree is None or prediction_tree is None:
        return 0.0
    max_len = max(int(getattr(reference_tree, "len_sexp", 0)), int(getattr(prediction_tree, "len_sexp", 0)))
    if max_len <= 0:
        return 1.0
    config = _build_apted_config(float(delete_cost), float(insert_cost), float(rename_cost))
    if config is None:
        return 0.0
    try:
        apted = APTED(reference_tree, prediction_tree, config)
        edit_distance = float(apted.compute_edit_distance())
    except Exception:
        return 0.0
    if edit_distance > max_len:
        return 0.0
    return float(max(0.0, min(1.0, (max_len - edit_distance) / float(max_len))))


def _resolve_codebertscore_device(requested):
    token = (requested or "auto").strip().lower()
    if token in {"cpu", "cuda", "mps"}:
        if token == "cuda":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if token == "mps":
            if (
                torch is not None
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
            return "cpu"
        return token

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if (
        torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _infer_codebertscore_num_layers(model_type):
    if model_type in _CODEBERTSCORE_LAYER_CACHE:
        return _CODEBERTSCORE_LAYER_CACHE[model_type]
    if AutoConfig is None:
        _CODEBERTSCORE_LAYER_CACHE[model_type] = None
        return None
    try:
        config = AutoConfig.from_pretrained(model_type)
    except Exception:
        _CODEBERTSCORE_LAYER_CACHE[model_type] = None
        return None
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            _CODEBERTSCORE_LAYER_CACHE[model_type] = int(value)
            return int(value)
    _CODEBERTSCORE_LAYER_CACHE[model_type] = None
    return None


def _infer_codebertscore_position_limit(scorer):
    model = getattr(scorer, "_model", None)
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "max_sequence_length"):
        value = getattr(config, attr, None)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _configure_codebertscore_tokenizer_max_length(scorer, requested_max_length=0):
    tokenizer = getattr(scorer, "_tokenizer", None)
    if tokenizer is None:
        return None

    requested = None
    try:
        requested = int(requested_max_length)
    except (TypeError, ValueError):
        requested = None
    if requested is not None and requested <= 0:
        requested = None

    model_limit = _infer_codebertscore_position_limit(scorer)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer_limit = int(tokenizer_limit)
    except (TypeError, ValueError):
        tokenizer_limit = None
    if tokenizer_limit is not None and tokenizer_limit <= 0:
        tokenizer_limit = None

    if model_limit is not None and requested is not None:
        effective = min(model_limit, requested)
    elif model_limit is not None:
        effective = model_limit
    elif requested is not None:
        effective = requested
    elif tokenizer_limit is not None:
        effective = tokenizer_limit
    else:
        effective = None

    if effective is not None:
        tokenizer.model_max_length = int(effective)
    return effective


def _resolve_codebertscore_scorer(
    model_type="microsoft/codebert-base",
    num_layers=None,
    device="auto",
    lang=None,
    idf=False,
    rescale_with_baseline=False,
    use_fast_tokenizer=False,
    nthreads=4,
):
    if BERTScorer is None:
        raise ImportError("CodeBERTScore requires bert-score. Install with: pip install bert-score")

    resolved_device = _resolve_codebertscore_device(device)
    parsed_layers = None if num_layers in (None, "", 0, "0") else int(num_layers)
    if parsed_layers is not None and parsed_layers <= 0:
        parsed_layers = None
    if parsed_layers is None:
        parsed_layers = _infer_codebertscore_num_layers(model_type)

    normalized_lang = None
    if isinstance(lang, str):
        normalized_lang = lang.strip() or None
    elif lang is not None:
        normalized_lang = str(lang).strip() or None

    cache_key = (
        str(model_type),
        parsed_layers,
        resolved_device,
        normalized_lang,
        bool(idf),
        bool(rescale_with_baseline),
        bool(use_fast_tokenizer),
        int(nthreads),
    )
    if cache_key in _CODEBERTSCORE_SCORER_CACHE:
        return _CODEBERTSCORE_SCORER_CACHE[cache_key]

    scorer = BERTScorer(
        model_type=str(model_type),
        num_layers=parsed_layers,
        lang=normalized_lang,
        idf=bool(idf),
        rescale_with_baseline=bool(rescale_with_baseline),
        use_fast_tokenizer=bool(use_fast_tokenizer),
        nthreads=max(1, int(nthreads)),
        device=resolved_device,
    )
    _CODEBERTSCORE_SCORER_CACHE[cache_key] = scorer
    return scorer


def prepare_codebertscore_context(codes):
    return {"texts": list(codes), "pair_cache": {}}


def _tensor_values_to_float_list(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(item) for item in values]


def _score_codebertscore_pair(
    reference,
    prediction,
    bidirectional=False,
    model_type="microsoft/codebert-base",
    num_layers=None,
    batch_size=16,
    max_length=0,
    device="auto",
    lang=None,
    idf=False,
    rescale_with_baseline=False,
    use_fast_tokenizer=False,
    nthreads=4,
    verbose=False,
):
    left = reference or ""
    right = prediction or ""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    scorer = _resolve_codebertscore_scorer(
        model_type=model_type,
        num_layers=num_layers,
        device=device,
        lang=lang,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
        use_fast_tokenizer=use_fast_tokenizer,
        nthreads=nthreads,
    )
    _configure_codebertscore_tokenizer_max_length(scorer=scorer, requested_max_length=max_length)
    batch = max(1, int(batch_size))

    _, _, f1_forward = scorer.score(cands=[right], refs=[left], batch_size=batch, verbose=bool(verbose))
    score_forward = _tensor_values_to_float_list(f1_forward)[0]
    if not bidirectional:
        return float(score_forward)

    _, _, f1_reverse = scorer.score(cands=[left], refs=[right], batch_size=batch, verbose=bool(verbose))
    score_reverse = _tensor_values_to_float_list(f1_reverse)[0]
    return float((score_forward + score_reverse) * 0.5)


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
    ruby_context=None,
    ruby_max_order=4,
    ruby_epsilon=1e-12,
    ruby_mode="auto",
    ruby_tokenizer="tranx",
    ruby_denominator="max",
    ruby_graph_timeout_seconds=1.0,
    ruby_graph_use_edge_cost=True,
    ruby_graph_include_leaf_edges=True,
    ruby_tree_max_nodes=180,
    ruby_tree_max_depth=10,
    ruby_tree_max_children=8,
    tsed_context=None,
    tsed_delete_cost=1.0,
    tsed_insert_cost=1.0,
    tsed_rename_cost=1.0,
    tsed_max_nodes=180,
    tsed_max_depth=10,
    tsed_max_children=8,
    codebertscore_context=None,
    codebertscore_model="microsoft/codebert-base",
    codebertscore_num_layers=None,
    codebertscore_batch_size=16,
    codebertscore_max_length=0,
    codebertscore_device="auto",
    codebertscore_lang=None,
    codebertscore_idf=False,
    codebertscore_rescale_with_baseline=False,
    codebertscore_use_fast_tokenizer=False,
    codebertscore_nthreads=4,
    codebertscore_verbose=False,
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

    if metric_key == "ruby":
        resolved_mode = _ruby_normalize_mode(ruby_mode)
        resolved_tokenizer = _ruby_normalize_tokenizer(ruby_tokenizer)
        resolved_denominator = _ruby_normalize_denominator(ruby_denominator)
        order_limit = max(1, min(int(ruby_max_order), 8))
        epsilon = float(ruby_epsilon)

        if resolved_mode == "ngram":
            if ruby_context is None:
                forward = _ruby_pair_score(
                    tokenize_for_code_metrics(reference),
                    tokenize_for_code_metrics(prediction),
                    max_order=order_limit,
                    epsilon=epsilon,
                )
            else:
                left_index = int(reference_index)
                right_index = int(prediction_index)
                order_scores = []
                counts_by_order = ruby_context["ngram_counts_by_order"]
                for n in range(1, order_limit + 1):
                    order_counts = counts_by_order.get(n, [])
                    if left_index >= len(order_counts) or right_index >= len(order_counts):
                        continue
                    ref_counts = order_counts[left_index]
                    hyp_counts = order_counts[right_index]
                    if (not ref_counts) and (not hyp_counts):
                        continue
                    order_scores.append(_ruby_ngram_f1(ref_counts, hyp_counts, epsilon=epsilon))
                forward = 0.0 if not order_scores else float(sum(order_scores) / float(len(order_scores)))
            return float(forward)

        normalized_language = normalize_code_language(language)
        if normalized_language not in _SUPPORTED_CODE_LANGUAGES:
            supported = ", ".join(_SUPPORTED_CODE_LANGUAGES)
            raise ValueError(f"RUBY currently supports: {supported}. Got: {language}")

        left_tokens = _tokenize_for_ruby(reference, tokenizer=resolved_tokenizer)
        right_tokens = _tokenize_for_ruby(prediction, tokenizer=resolved_tokenizer)
        left_tree = None
        right_tree = None
        ruby_cache = {}
        cache_key = None

        if (
            ruby_context is not None
            and reference_index is not None
            and prediction_index is not None
        ):
            left_index = int(reference_index)
            right_index = int(prediction_index)
            ruby_cache = ruby_context.get("pair_cache", {})
            if bidirectional:
                left_index, right_index = min(left_index, right_index), max(left_index, right_index)
                direction_key = "sym"
            else:
                direction_key = "dir"
            cache_key = (
                direction_key,
                left_index,
                right_index,
                resolved_mode,
                resolved_tokenizer,
                resolved_denominator,
                float(ruby_graph_timeout_seconds),
                bool(ruby_graph_use_edge_cost),
                bool(ruby_graph_include_leaf_edges),
            )
            if cache_key in ruby_cache:
                return float(ruby_cache[cache_key])

            selected_tokens = ruby_context.get("selected_tokens_by_doc", [])
            tranx_tokens = ruby_context.get("tranx_tokens_by_doc", [])
            regex_tokens = ruby_context.get("tokens_by_doc", [])
            token_source = tranx_tokens if resolved_tokenizer == "tranx" else regex_tokens
            if not token_source:
                token_source = selected_tokens
            if left_index < len(token_source):
                left_tokens = token_source[left_index]
            if right_index < len(token_source):
                right_tokens = token_source[right_index]

            trees = ruby_context.get("trees", [])
            if left_index < len(trees):
                left_tree = trees[left_index]
            if right_index < len(trees):
                right_tree = trees[right_index]

        if left_tree is None or right_tree is None:
            left_tree = _tsed_get_tree(
                normalized_language,
                reference,
                max_nodes=ruby_tree_max_nodes,
                max_depth=ruby_tree_max_depth,
                max_children=ruby_tree_max_children,
            )
            right_tree = _tsed_get_tree(
                normalized_language,
                prediction,
                max_nodes=ruby_tree_max_nodes,
                max_depth=ruby_tree_max_depth,
                max_children=ruby_tree_max_children,
            )

        resolved_score = None
        if resolved_mode in {"auto", "graph"}:
            resolved_score = _ruby_graph_similarity(
                left_tree,
                right_tree,
                timeout_seconds=ruby_graph_timeout_seconds,
                use_edge_cost=bool(ruby_graph_use_edge_cost),
                include_leaf_edges=bool(ruby_graph_include_leaf_edges),
            )
        if resolved_score is None and resolved_mode in {"auto", "tree", "graph"}:
            resolved_score = _ruby_tree_similarity(left_tree, right_tree)
        if resolved_score is None:
            resolved_score = _ruby_string_similarity_from_tokens(
                left_tokens,
                right_tokens,
                denominator=resolved_denominator,
            )

        if cache_key is not None:
            ruby_cache[cache_key] = float(resolved_score)
            ruby_context["pair_cache"] = ruby_cache
        return float(resolved_score)

    if metric_key == "tsed":
        normalized_language = normalize_code_language(language)
        if normalized_language not in _SUPPORTED_CODE_LANGUAGES:
            supported = ", ".join(_SUPPORTED_CODE_LANGUAGES)
            raise ValueError(f"TSED currently supports: {supported}. Got: {language}")
        if not _tsed_runtime_available():
            raise ImportError(
                "TSED requires optional dependencies. Install with: pip install apted tree-sitter-language-pack"
            )

        if tsed_context is None:
            tsed_context = prepare_tsed_context(
                [reference, prediction],
                language=normalized_language,
                max_nodes=tsed_max_nodes,
                max_depth=tsed_max_depth,
                max_children=tsed_max_children,
            )
            left_tree = tsed_context["trees"][0]
            right_tree = tsed_context["trees"][1]
        else:
            left_index = int(reference_index)
            right_index = int(prediction_index)
            left_tree = tsed_context["trees"][left_index]
            right_tree = tsed_context["trees"][right_index]

        return _tsed_pair_score(
            left_tree,
            right_tree,
            delete_cost=tsed_delete_cost,
            insert_cost=tsed_insert_cost,
            rename_cost=tsed_rename_cost,
        )

    if metric_key == "codebertscore":
        cache = {}
        cache_key = None
        if (
            codebertscore_context is not None
            and reference_index is not None
            and prediction_index is not None
        ):
            cache = codebertscore_context.get("pair_cache", {})
            if bidirectional:
                left_index = min(int(reference_index), int(prediction_index))
                right_index = max(int(reference_index), int(prediction_index))
                cache_key = (
                    "sym",
                    left_index,
                    right_index,
                    str(codebertscore_model),
                    str(codebertscore_num_layers),
                    int(codebertscore_batch_size),
                    int(codebertscore_max_length),
                    str(codebertscore_device),
                )
            else:
                cache_key = (
                    "dir",
                    int(reference_index),
                    int(prediction_index),
                    str(codebertscore_model),
                    str(codebertscore_num_layers),
                    int(codebertscore_batch_size),
                    int(codebertscore_max_length),
                    str(codebertscore_device),
                )
            if cache_key in cache:
                return float(cache[cache_key])

        score = _score_codebertscore_pair(
            reference,
            prediction,
            bidirectional=bidirectional,
            model_type=codebertscore_model,
            num_layers=codebertscore_num_layers,
            batch_size=codebertscore_batch_size,
            max_length=codebertscore_max_length,
            device=codebertscore_device,
            lang=codebertscore_lang,
            idf=codebertscore_idf,
            rescale_with_baseline=codebertscore_rescale_with_baseline,
            use_fast_tokenizer=codebertscore_use_fast_tokenizer,
            nthreads=codebertscore_nthreads,
            verbose=codebertscore_verbose,
        )
        if cache_key is not None:
            cache[cache_key] = float(score)
            codebertscore_context["pair_cache"] = cache
        return float(score)

    raise ValueError(f"Unsupported code metric: {metric_name}")
