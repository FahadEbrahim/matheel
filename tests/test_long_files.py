import json
import re
from types import SimpleNamespace

import pytest

from matheel import long_files
from matheel.long_files import compare_long_files, preflight_long_file


class OffsetTokenizer:
    """Small deterministic tokenizer with two retained boundary tokens."""

    model_max_length = 8

    def __init__(self, side="right"):
        self.side = side

    def __call__(self, text, *, truncation=False, max_length=None, **_kwargs):
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        if truncation:
            amount = max_length - 2
            offsets = offsets[:amount] if self.side == "right" else offsets[-amount:]
        return {"input_ids": [101, *range(len(offsets)), 102],
                "offset_mapping": [(0, 0), *offsets, (0, 0)],
                "special_tokens_mask": [1, *([0] * len(offsets)), 1]}


@pytest.mark.parametrize("side,visible", [("right", 2), ("left", 3)])
def test_tokenizer_offsets_measure_prompt_special_tokens_and_truncation_side(side, visible):
    report = preflight_long_file(
        "alpha beta gamma delta", options={"vector_backend": "sentence_transformers", "max_token_length": 5},
        tokenizer=OffsetTokenizer(side), prefix="query: ",
    )
    chunk = report["chunks"][0]
    assert chunk["input_tokens"] == 7
    assert chunk["special_tokens"] == 2
    assert chunk["source_tokens"] == 4
    assert chunk["retained_input_tokens"] == 5
    assert chunk["retained_source_tokens"] == visible
    assert chunk["truncated"] is True
    assert report["effective_model_input_coverage"]["byte_fraction"] < 1


def test_detected_model_limit_caps_requested_limit_and_default_prompt_is_used():
    model = SimpleNamespace(max_seq_length=5, tokenizer=OffsetTokenizer(),
                            default_prompt_name="document", prompts={"document": "query: "})
    report = preflight_long_file(
        "alpha beta gamma delta", model=model,
        options={"vector_backend": "sentence_transformers", "max_token_length": 100},
    )
    assert report["effective_max_input_tokens"] == 5
    assert report["prefix"] == "query: "
    assert report["chunks"][0]["retained_source_tokens"] == 2


def test_prefix_can_consume_all_content_budget_without_falsely_claiming_source_coverage():
    report = preflight_long_file(
        "source tokens", options={"vector_backend": "sentence_transformers", "max_token_length": 3},
        tokenizer=OffsetTokenizer(), prefix="query prompt: ",
    )
    assert report["chunks"][0]["retained_input_tokens"] == 3
    assert report["chunks"][0]["retained_source_tokens"] == 0
    assert report["effective_model_input_coverage"]["byte_fraction"] == 0


def test_preconfigured_rust_tokenizer_records_unknown_counts_without_mutation_or_failure():
    class TruncatingRustTokenizer:
        truncation = {"max_length": 4}

        def encode(self, *_args, **_kwargs):
            raise AssertionError("Already truncated encode must not be mistaken for a full token count")

    tokenizer = TruncatingRustTokenizer()
    report = preflight_long_file(
        "source tokens that may be longer than its input window",
        options={"vector_backend": "model2vec"}, tokenizer=tokenizer,
    )
    assert tokenizer.truncation == {"max_length": 4}
    assert report["effective_max_input_tokens"] == 4
    assert report["unknown_truncation_chunks"] == 1
    assert report["chunks"][0]["input_tokens"] is None
    assert "truncation configured" in report["chunks"][0]["tokenization_note"]


def test_count_only_tokenizer_reports_limit_overrun_but_not_invented_source_visibility():
    class CountOnlyTokenizer:
        model_max_length = 5

        def encode(self, text, *, add_special_tokens, truncation=False):
            assert truncation is False
            return [1] * (len(text.split()) + (2 if add_special_tokens else 0))

    report = preflight_long_file(
        "one two three four five", options={"vector_backend": "sentence_transformers"},
        tokenizer=CountOnlyTokenizer(),
    )
    assert report["chunks"][0]["input_tokens"] == 7
    assert report["chunks"][0]["special_tokens"] == 2
    assert report["chunks"][0]["truncated"] is True
    assert report["chunks"][0]["retained_input_tokens"] is None
    assert report["effective_model_input_coverage"]["byte_fraction"] is None


def test_multivector_query_and_document_limits_are_distinct():
    model = SimpleNamespace(query_length=4, document_length=8, tokenizer=OffsetTokenizer())
    options = {"vector_backend": "multivector"}
    query = preflight_long_file("one two three four five", options=options, model=model, is_query=True)
    document = preflight_long_file("one two three four five", options=options, model=model)
    assert query["effective_max_input_tokens"] == 4
    assert document["effective_max_input_tokens"] == 8
    assert query["truncated_chunks"] == 1 and document["truncated_chunks"] == 0


def test_unknown_tokenizer_has_real_utf8_line_counts_but_never_guesses_tokens():
    text = "écho\n日本語\n"
    report = preflight_long_file(text, options={"vector_backend": "sentence_transformers",
                                               "max_token_length": 2})
    assert report["source_utf8_bytes"] == len(text.encode("utf-8"))
    assert report["source_lines"] == 2
    assert report["selected_chunk_coverage"]["utf8_bytes"] == report["prepared_utf8_bytes"]
    assert report["prepared_utf8_bytes"] == len(text.strip().encode("utf-8"))
    assert report["chunks"][0]["input_tokens"] is None
    assert report["effective_model_input_coverage"]["byte_fraction"] is None
    assert report["unknown_truncation_chunks"] == 1


def test_static_hash_ignores_transformer_limit_and_preserves_all_input():
    report = preflight_long_file("one two three four", options={"vector_backend": "static_hash",
                                                                "max_token_length": 1})
    assert report["effective_max_input_tokens"] is None
    assert report["chunks"][0]["input_tokens"] == 4
    assert report["truncated_chunks"] == 0
    assert report["effective_model_input_coverage"]["byte_fraction"] == 1


def test_chunk_overlap_is_not_double_counted_and_first_aggregation_exposes_lost_input(monkeypatch):
    monkeypatch.setattr(long_files, "chunk_text", lambda *args, **kwargs: ["abcde", "defgh", "ghij"])
    report = preflight_long_file("abcdefghij", options={"max_chunks": 2, "chunk_aggregation": "first"})
    assert report["generated_chunks"] == 3
    assert report["selected_chunks"] == 2 and report["omitted_chunks"] == 1
    assert report["selected_chunk_coverage"]["byte_fraction"] == 0.8
    assert report["effective_model_input_coverage"]["byte_fraction"] == 0.5
    assert report["effective_chunks"] == 1


def test_generated_or_rewritten_chunk_does_not_get_false_complete_source_coverage(monkeypatch):
    monkeypatch.setattr(long_files, "chunk_text", lambda *args, **kwargs: ["rewritten source"])
    report = preflight_long_file("original source")
    assert report["alignment_ambiguous"]
    assert report["selected_chunk_coverage"]["byte_fraction"] is None
    assert report["effective_model_input_coverage"]["byte_fraction"] is None


def test_limit_too_small_for_required_special_tokens_is_rejected():
    with pytest.raises(ValueError, match="special tokens"):
        preflight_long_file("alpha", options={"vector_backend": "sentence_transformers", "max_token_length": 1},
                            tokenizer=OffsetTokenizer())


@pytest.mark.parametrize("maximum", [-1, 1.5, True])
def test_max_chunks_requires_a_nonnegative_integer(maximum):
    with pytest.raises(ValueError, match="max_chunks"):
        preflight_long_file("text", options={"max_chunks": maximum})


def test_real_offline_chunk_comparison_reveals_tail_blind_first_and_capped_strategies(tmp_path):
    pytest.importorskip("chonkie")
    shared = "".join(f"prefix_{index} = {index}\n" for index in range(50))
    left = shared + "".join(f"left_branch_{index} = {index * 7}\n" for index in range(100))
    right = shared + "".join(f"different_tail_{index} = '{index}'\n" for index in range(100))
    report, paths = compare_long_files(left, right, output_dir=tmp_path)
    scores = {row["run_name"]: row["similarity_score"] for row in report["scores"]}
    assert all(row["status"] == "ok" for row in report["scores"])
    assert scores["chunked_first"] == pytest.approx(1)
    assert scores["chunked_mean_cap_2"] == pytest.approx(1)
    assert scores["raw_static_hash"] < 0.99
    assert scores["chunked_mean"] < 0.99
    preflight = {row["run_name"]: row["files"]["left"] for row in report["preflight"]}
    assert preflight["chunked_first"]["effective_model_input_coverage"]["byte_fraction"] < 0.1
    assert preflight["chunked_mean_cap_2"]["omitted_chunks"] > 0
    saved = json.loads((tmp_path / "long_files.json").read_text())
    assert saved["inputs"]["left"]["text"] == left
    assert paths["scores_csv"].endswith("long_files_scores.csv")
