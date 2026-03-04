import pytest

import matheel.code_metrics as code_metrics_module
from matheel.code_metrics import (
    available_code_metrics,
    codebleu_components,
    prepare_ruby_context,
    prepare_crystalbleu_context,
    score_code_metric_pair,
)


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


def test_codebleu_accepts_cpp_alias_and_rejects_unsupported_language():
    components = codebleu_components("int main() { return 0; }", "int main() { return 0; }", language="c++")

    assert components["codebleu"] == 1.0

    with pytest.raises(ValueError):
        codebleu_components("print('x')", "print('x')", language="ruby")


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
