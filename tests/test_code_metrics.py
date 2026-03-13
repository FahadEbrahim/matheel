import pytest

import matheel.code_metrics as code_metrics_module
from matheel.native_codebleu import calc_codebleu as native_calc_codebleu
from matheel.code_metrics import (
    available_ast_metric_languages,
    available_codebleu_languages,
    available_code_metric_languages,
    available_code_metrics,
    codebleu_runtime_available,
    codebleu_components,
    normalize_code_language,
    prepare_crystalbleu_context,
    prepare_ruby_context,
    score_code_metric_pair,
)

try:
    from codebleu import calc_codebleu as pip_calc_codebleu
except ImportError:  # pragma: no cover - optional dependency
    pip_calc_codebleu = None


SUPPORTED_LANGUAGE_SNIPPETS = {
    "java": "class Demo { int add(int left, int right) { return left + right; } }",
    "python": "def add(left, right):\n    return left + right\n",
    "c": "int add(int left, int right) { return left + right; }",
    "cpp": "int add(int left, int right) { return left + right; }",
    "go": "package main\nfunc add(left int, right int) int { return left + right }\n",
    "javascript": "function add(left, right) { return left + right; }",
    "typescript": "function add(left: number, right: number): number { return left + right; }",
    "kotlin": "fun add(left: Int, right: Int): Int { return left + right }",
    "scala": "def add(left: Int, right: Int): Int = { return left + right }",
    "swift": "func add(_ left: Int, _ right: Int) -> Int { return left + right }",
    "solidity": "pragma solidity ^0.8.0; contract Demo { function add(uint256 left, uint256 right) public pure returns (uint256) { return left + right; } }",
    "dart": "int add(int left, int right) { return left + right; }",
    "php": "<?php function add($left, $right) { return $left + $right; }",
    "ruby": "def add(left, right)\n  return left + right\nend\n",
    "rust": "fn add(left: i32, right: i32) -> i32 { return left + right; }",
    "csharp": "class Demo { static int Add(int left, int right) { return left + right; } }",
    "lua": "local function add(left, right) local total = left + right return total end",
    "julia": "function add(left, right)\n total = left + right\n return total\n end",
    "r": "add <- function(left, right) { total <- left + right; return(total) }",
    "objc": "@implementation Demo - (int)add:(int)left right:(int)right { int total = left + right; return total; } @end",
}

STRICT_CODEBLEU_LANGUAGE_SNIPPETS = {
    language: SUPPORTED_LANGUAGE_SNIPPETS[language]
    for language in available_codebleu_languages()
}
REQUIRES_CODEBLEU = pytest.mark.skipif(
    not codebleu_runtime_available(),
    reason="Native CodeBLEU syntax/dataflow requires the tree-sitter runtime.",
)
REQUIRES_PIP_CODEBLEU = pytest.mark.skipif(
    pip_calc_codebleu is None,
    reason="The optional pip `codebleu` package is not installed.",
)

PIP_CODEBLEU_EXACT_MATCH_EXAMPLES = (
    (
        "java",
        "class Demo { int add ( int left , int right ) { int total = left + right ; return total ; } }",
        "class Demo { int add ( int left , int right ) { int total = left + right ; return total ; } }",
    ),
    (
        "c",
        "int add ( int left , int right ) { int total = left + right ; return total ; }",
        "int add ( int a , int b ) { int total = a + b ; return total ; }",
    ),
    (
        "php",
        "<?php function add ( $left , $right ) { $total = $left + $right ; return $total ; }",
        "<?php function add ( $left , $right ) { $total = $left + $right ; return $total ; }",
    ),
    (
        "csharp",
        "class Demo { static int Add ( int left , int right ) { int total = left + right ; return total ; } }",
        "class Demo { static int Add ( int a , int b ) { int total = a + b ; return total ; } }",
    ),
)

NORMALIZE_ALIAS_CASES = (
    ("c++", "cpp"),
    ("golang", "go"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("kt", "kotlin"),
    ("kts", "kotlin"),
    ("sol", "solidity"),
    ("rb", "ruby"),
    ("rs", "rust"),
    ("c#", "csharp"),
    ("c_sharp", "csharp"),
    ("cs", "csharp"),
    ("jl", "julia"),
    ("objective-c", "objc"),
    ("objectivec", "objc"),
    ("obj-c", "objc"),
)

STRICT_CODEBLEU_ALIAS_CASES = (
    ("c++", "cpp"),
    ("golang", "go"),
    ("js", "javascript"),
    ("rb", "ruby"),
    ("rs", "rust"),
    ("c#", "csharp"),
    ("c_sharp", "csharp"),
    ("cs", "csharp"),
    ("jl", "julia"),
    ("objective-c", "objc"),
    ("objectivec", "objc"),
    ("obj-c", "objc"),
)

NEW_TREE_SITTER_LANGUAGE_SNIPPETS = {
    "go": SUPPORTED_LANGUAGE_SNIPPETS["go"],
    "javascript": SUPPORTED_LANGUAGE_SNIPPETS["javascript"],
    "typescript": SUPPORTED_LANGUAGE_SNIPPETS["typescript"],
    "kotlin": SUPPORTED_LANGUAGE_SNIPPETS["kotlin"],
    "scala": SUPPORTED_LANGUAGE_SNIPPETS["scala"],
    "swift": SUPPORTED_LANGUAGE_SNIPPETS["swift"],
    "solidity": SUPPORTED_LANGUAGE_SNIPPETS["solidity"],
    "dart": SUPPORTED_LANGUAGE_SNIPPETS["dart"],
    "php": SUPPORTED_LANGUAGE_SNIPPETS["php"],
    "ruby": SUPPORTED_LANGUAGE_SNIPPETS["ruby"],
    "rust": SUPPORTED_LANGUAGE_SNIPPETS["rust"],
    "csharp": SUPPORTED_LANGUAGE_SNIPPETS["csharp"],
    "lua": SUPPORTED_LANGUAGE_SNIPPETS["lua"],
    "julia": SUPPORTED_LANGUAGE_SNIPPETS["julia"],
    "r": SUPPORTED_LANGUAGE_SNIPPETS["r"],
    "objc": SUPPORTED_LANGUAGE_SNIPPETS["objc"],
}


@REQUIRES_CODEBLEU
def test_codebleu_components_score_identical_code_as_one():
    components = codebleu_components("int value = 1;", "int value = 1;", language="java")

    assert components["codebleu"] == 1.0
    assert components["codebleu_ngram"] == 1.0
    assert components["codebleu_weighted_ngram"] == 1.0
    assert components["codebleu_syntax"] == 1.0
    assert components["codebleu_dataflow"] == 1.0


def test_available_code_metrics_includes_new_metrics():
    metrics = available_code_metrics()

    assert "ruby" in metrics
    assert "tsed" in metrics
    assert "codebertscore" in metrics


def test_available_code_metric_languages_expose_expanded_scope():
    assert available_code_metric_languages() == tuple(SUPPORTED_LANGUAGE_SNIPPETS.keys())


def test_available_ast_metric_languages_match_structural_scope():
    assert available_ast_metric_languages() == tuple(SUPPORTED_LANGUAGE_SNIPPETS.keys())


@REQUIRES_CODEBLEU
def test_available_codebleu_languages_expose_real_dfg_scope():
    assert available_codebleu_languages() == (
        "java",
        "python",
        "c",
        "cpp",
        "go",
        "javascript",
        "typescript",
        "kotlin",
        "scala",
        "swift",
        "solidity",
        "dart",
        "php",
        "ruby",
        "rust",
        "csharp",
        "lua",
        "julia",
        "r",
        "objc",
    )


@REQUIRES_CODEBLEU
@pytest.mark.parametrize("language,snippet", STRICT_CODEBLEU_LANGUAGE_SNIPPETS.items())
def test_codebleu_supports_real_dfg_languages(language, snippet):
    components = codebleu_components(snippet, snippet, language=language)

    assert components["codebleu"] == pytest.approx(1.0)


def test_crystalbleu_context_can_be_reused_across_pairs():
    codes = [
        "int value = 1;",
        "int value = 1;",
        "return value + 2;",
    ]
    context = prepare_crystalbleu_context(codes, max_order=4)

    identical_score = score_code_metric_pair(
        codes[0],
        codes[1],
        metric_name="crystalbleu",
        crystalbleu_context=context,
        reference_index=0,
        prediction_index=1,
        crystalbleu_trivial_ngram_count=0,
    )
    different_score = score_code_metric_pair(
        codes[0],
        codes[2],
        metric_name="crystalbleu",
        crystalbleu_context=context,
        reference_index=0,
        prediction_index=2,
        crystalbleu_trivial_ngram_count=0,
    )

    assert identical_score == 1.0
    assert different_score < identical_score


@pytest.mark.parametrize("alias,normalized_language", NORMALIZE_ALIAS_CASES)
def test_normalize_code_language_accepts_aliases(alias, normalized_language):
    assert normalize_code_language(alias) == normalized_language


@REQUIRES_CODEBLEU
@pytest.mark.parametrize("alias,normalized_language", STRICT_CODEBLEU_ALIAS_CASES)
def test_codebleu_accepts_real_dfg_language_aliases(alias, normalized_language):
    snippet = SUPPORTED_LANGUAGE_SNIPPETS[normalized_language]
    components = codebleu_components(snippet, snippet, language=alias)

    assert components["codebleu"] == 1.0


@REQUIRES_CODEBLEU
@pytest.mark.parametrize("language", ("haskell", "sql", "bash", "elixir"))
def test_codebleu_rejects_languages_without_real_dataflow(language):
    with pytest.raises(ValueError):
        codebleu_components("print('x')", "print('x')", language=language)


@REQUIRES_CODEBLEU
@REQUIRES_PIP_CODEBLEU
@pytest.mark.parametrize("language,reference,prediction", PIP_CODEBLEU_EXACT_MATCH_EXAMPLES)
def test_native_codebleu_matches_pip_selected_exact_examples(language, reference, prediction):
    native_scores = native_calc_codebleu([reference], [prediction], lang=language)
    pip_scores = pip_calc_codebleu([reference], [prediction], lang="c_sharp" if language == "csharp" else language)

    assert native_scores["codebleu"] == pytest.approx(pip_scores["codebleu"])
    assert native_scores["ngram_match_score"] == pytest.approx(pip_scores["ngram_match_score"])
    assert native_scores["weighted_ngram_match_score"] == pytest.approx(pip_scores["weighted_ngram_match_score"])
    assert native_scores["syntax_match_score"] == pytest.approx(pip_scores["syntax_match_score"])
    assert native_scores["dataflow_match_score"] == pytest.approx(pip_scores["dataflow_match_score"])


def test_ruby_graph_mode_is_strict_when_graph_score_is_unavailable(monkeypatch):
    monkeypatch.setattr(code_metrics_module, "_tsed_get_tree", lambda *args, **kwargs: object())
    monkeypatch.setattr(code_metrics_module, "_ruby_graph_similarity", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_metrics_module, "_ruby_tree_similarity", lambda *args, **kwargs: 0.73)

    with pytest.raises(RuntimeError):
        score_code_metric_pair(
            "def add(a, b): return a + b",
            "def sum_two(x, y): return x + y",
            metric_name="ruby",
            language="python",
            ruby_mode="graph",
        )


def test_ruby_tree_mode_is_strict_when_tree_score_is_unavailable(monkeypatch):
    monkeypatch.setattr(code_metrics_module, "_tsed_get_tree", lambda *args, **kwargs: object())
    monkeypatch.setattr(code_metrics_module, "_ruby_tree_similarity", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError):
        score_code_metric_pair(
            "def add(a, b): return a + b",
            "def sum_two(x, y): return x + y",
            metric_name="ruby",
            language="python",
            ruby_mode="tree",
        )


@REQUIRES_CODEBLEU
def test_codebleu_dataflow_tracks_program_structure_under_identifier_renaming():
    reference = "class Demo { int add(int left, int right) { int total = left + right; return total; } }"
    prediction = "class Demo { int add(int a, int b) { int result = a + b; return result; } }"

    components = codebleu_components(reference, prediction, language="java")

    assert components["codebleu_dataflow"] > 0.8
    assert components["codebleu_weighted_ngram"] < 1.0
    assert components["codebleu_dataflow"] > components["codebleu_weighted_ngram"]


def test_ruby_context_can_be_reused_across_pairs():
    codes = [
        "def add(a, b): return a + b",
        "def sum_two(x, y): return x + y",
        "def is_even(x): return x % 2 == 0",
    ]
    context = prepare_ruby_context(codes, max_order=4)

    similar_score = score_code_metric_pair(
        codes[0],
        codes[1],
        metric_name="ruby",
        ruby_context=context,
        reference_index=0,
        prediction_index=1,
        ruby_max_order=4,
    )
    different_score = score_code_metric_pair(
        codes[0],
        codes[2],
        metric_name="ruby",
        ruby_context=context,
        reference_index=0,
        prediction_index=2,
        ruby_max_order=4,
    )

    assert similar_score > different_score


def test_ruby_string_mode_uses_tranx_token_edit_similarity():
    identical = score_code_metric_pair(
        "int sum = a + b;",
        "int sum = a + b;",
        metric_name="ruby",
        language="java",
        ruby_mode="string",
        ruby_tokenizer="tranx",
        ruby_denominator="max",
    )
    near = score_code_metric_pair(
        "int sum = a + b;",
        "int sum = a + c;",
        metric_name="ruby",
        language="java",
        ruby_mode="string",
        ruby_tokenizer="tranx",
        ruby_denominator="max",
    )

    assert identical == pytest.approx(1.0)
    assert 0.0 <= near < identical


def test_ruby_string_mode_supports_cpp_language_scope():
    score = score_code_metric_pair(
        "int main(){ return 0; }",
        "int main(){ return 1; }",
        metric_name="ruby",
        language="cpp",
        ruby_mode="string",
    )

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("language,snippet", SUPPORTED_LANGUAGE_SNIPPETS.items())
def test_ruby_string_mode_supports_all_scoped_languages(language, snippet):
    score = score_code_metric_pair(
        snippet,
        snippet,
        metric_name="ruby",
        language=language,
        ruby_mode="string",
    )

    assert score == pytest.approx(1.0)


def test_ruby_auto_mode_falls_back_from_graph_to_tree(monkeypatch):
    monkeypatch.setattr(code_metrics_module, "_tsed_get_tree", lambda *args, **kwargs: object())
    monkeypatch.setattr(code_metrics_module, "_ruby_graph_similarity", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_metrics_module, "_ruby_tree_similarity", lambda *args, **kwargs: 0.73)
    monkeypatch.setattr(code_metrics_module, "_ruby_string_similarity_from_tokens", lambda *args, **kwargs: 0.25)

    score = score_code_metric_pair(
        "def add(a, b): return a + b",
        "def sum_two(x, y): return x + y",
        metric_name="ruby",
        language="python",
        ruby_mode="auto",
    )

    assert score == pytest.approx(0.73)


def test_ruby_auto_mode_falls_back_to_string_when_graph_and_tree_unavailable(monkeypatch):
    monkeypatch.setattr(code_metrics_module, "_tsed_get_tree", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_metrics_module, "_ruby_graph_similarity", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_metrics_module, "_ruby_tree_similarity", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_metrics_module, "_ruby_string_similarity_from_tokens", lambda *args, **kwargs: 0.41)

    score = score_code_metric_pair(
        "def add(a, b): return a + b",
        "def sum_two(x, y): return x + y",
        metric_name="ruby",
        language="python",
        ruby_mode="auto",
    )

    assert score == pytest.approx(0.41)


def test_codebertscore_reuses_pair_cache(monkeypatch):
    calls = {"count": 0}

    def fake_score(reference, prediction, **kwargs):
        calls["count"] += 1
        return 0.4321

    monkeypatch.setattr(code_metrics_module, "_score_codebertscore_pair", fake_score)
    context = {"pair_cache": {}}

    first = score_code_metric_pair(
        "left",
        "right",
        metric_name="codebertscore",
        codebertscore_context=context,
        reference_index=0,
        prediction_index=1,
    )
    second = score_code_metric_pair(
        "left",
        "right",
        metric_name="codebertscore",
        codebertscore_context=context,
        reference_index=0,
        prediction_index=1,
    )

    assert first == pytest.approx(0.4321)
    assert second == pytest.approx(0.4321)
    assert calls["count"] == 1


def test_codebertscore_bidirectional_cache_is_symmetric(monkeypatch):
    calls = {"count": 0}

    def fake_score(reference, prediction, **kwargs):
        calls["count"] += 1
        return 0.6123

    monkeypatch.setattr(code_metrics_module, "_score_codebertscore_pair", fake_score)
    context = {"pair_cache": {}}

    first = score_code_metric_pair(
        "left",
        "right",
        metric_name="codebertscore",
        bidirectional=True,
        codebertscore_context=context,
        reference_index=0,
        prediction_index=1,
    )
    second = score_code_metric_pair(
        "right",
        "left",
        metric_name="codebertscore",
        bidirectional=True,
        codebertscore_context=context,
        reference_index=1,
        prediction_index=0,
    )

    assert first == pytest.approx(0.6123)
    assert second == pytest.approx(0.6123)
    assert calls["count"] == 1


def test_tsed_metric_scores_identical_higher_than_different():
    if not code_metrics_module._tsed_runtime_available():
        pytest.skip("TSED optional runtime is not available.")

    identical = score_code_metric_pair(
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b\n",
        metric_name="tsed",
        language="python",
    )
    different = score_code_metric_pair(
        "def add(a, b):\n    return a + b\n",
        "def is_even(x):\n    return x % 2 == 0\n",
        metric_name="tsed",
        language="python",
    )

    assert 0.0 <= identical <= 1.0
    assert 0.0 <= different <= 1.0
    assert identical >= different


@pytest.mark.parametrize("language,snippet", NEW_TREE_SITTER_LANGUAGE_SNIPPETS.items())
def test_tsed_supports_new_scoped_languages(language, snippet):
    if not code_metrics_module._tsed_runtime_available():
        pytest.skip("TSED optional runtime is not available.")

    score = score_code_metric_pair(
        snippet,
        snippet,
        metric_name="tsed",
        language=language,
    )

    assert score == pytest.approx(1.0)
