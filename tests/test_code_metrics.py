import pytest

from matheel.code_metrics import (
    codebleu_components,
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
