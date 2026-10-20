"""Score parity and operation-count checks for exact lexical fast paths."""

import random

import pandas as pd
import pytest

from matheel import evaluation, similarity
from matheel.datasets import write_retrieval_dataset


def reference_gst(left, right, minimum=5):
    """Deliberately exhaustive GST oracle with the established tile tie order."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    minimum = min(max(1, int(minimum)), len(left), len(right))
    used_left, used_right = set(), set()
    covered = 0
    while True:
        matches = []
        longest = minimum - 1
        for i in range(len(left)):
            for j in range(len(right)):
                length = 0
                while (i + length < len(left) and j + length < len(right)
                       and i + length not in used_left and j + length not in used_right
                       and left[i + length] == right[j + length]):
                    length += 1
                if length >= minimum:
                    if length > longest:
                        longest, matches = length, []
                    if length == longest:
                        matches.append((i, j, length))
        if not matches:
            return 2.0 * covered / float(len(left) + len(right))
        accepted = False
        for i, j, length in sorted(matches, key=lambda item: (-item[2], item[0], item[1])):
            if any(i + offset in used_left or j + offset in used_right for offset in range(length)):
                continue
            used_left.update(range(i, i + length))
            used_right.update(range(j, j + length))
            covered += length
            accepted = True
        assert accepted


def test_indexed_gst_is_exact_for_adversarial_and_random_token_streams():
    examples = [([], []), ([], ["x"]), (["x"], ["x"]),
                (list("abXcdabYcd"), list("abYcdabXcd")),
                (list("aaaaabaaaaa"), list("aaaaacaaaaa")),
                (list("ababababab"), list("bababababa"))]
    rng = random.Random(730)
    examples += [(rng.choices("abc();", k=rng.randrange(19)),
                  rng.choices("abc();", k=rng.randrange(19))) for _ in range(100)]
    for left, right in examples:
        for minimum in (1, 2, 5, 30):
            assert similarity.gst_similarity_from_tokens(left, right, minimum) == reference_gst(left, right, minimum)


def test_gst_window_index_prunes_punctuation_only_candidates_without_changing_score():
    class Token(str):
        comparisons = 0
        __hash__ = str.__hash__

        def __eq__(self, other):
            type(self).comparisons += 1
            return str.__eq__(self, other)

    left = [Token(token) for index in range(25) for token in ("(", f"left{index}", ")", ";", "}")]
    right = [Token(token) for index in range(25) for token in ("(", f"right{index}", ")", ";", "}")]
    expected = reference_gst(left, right)
    reference_comparisons = Token.comparisons
    Token.comparisons = 0
    assert similarity.gst_similarity_from_tokens(left, right) == expected
    assert Token.comparisons < reference_comparisons // 4


def test_gst_hash_collisions_are_verified_and_large_minimum_has_bounded_prefix_storage():
    class Collision(str):
        def __hash__(self):
            return 0
    left, right = list(map(Collision, "abababxyz")), list(map(Collision, "bababqxyz"))
    assert similarity.gst_similarity_from_tokens(left, right, 2) == reference_gst(left, right, 2)

    class CountedHash(str):
        calls = 0
        def __hash__(self):
            type(self).calls += 1
            return str.__hash__(self)
    tokens = [CountedHash(str(index)) for index in range(200)]
    assert similarity.gst_similarity_from_tokens(tokens, tokens, 100) == 1.0
    assert CountedHash.calls < 16 * len(tokens)


@pytest.mark.parametrize("weights", [
    {"levenshtein": 0.5, "winnowing": 0.25, "gst": 0.25},
    {"gst": 1}, {"winnowing": 1}, {"rapidfuzz_partial_ratio": 1},
    {"lcs": 0.3, "jaro_winkler": 0.2, "winnowing": 0.5},
])
def test_dataset_lexical_preparation_preserves_scalar_scores_and_runs_once_per_text(tmp_path, monkeypatch, weights):
    dataset = write_retrieval_dataset(tmp_path / "data", files=pd.DataFrame([
        {"file_id": "a", "text": "int total = value + 1;"},
        {"file_id": "b", "text": "int total = value + 2;"},
        {"file_id": "copy", "text": "int total = value + 1;"},
    ]), queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame([{"document_id": key, "file_id": key} for key in ("a", "b", "copy")]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "b", "relevance": 1}]))
    calls = []
    original = similarity.tokenize_for_lexical_matching
    def tokenize(text, **kwargs):
        calls.append(text)
        return original(text, **kwargs)
    options = {"feature_weights": weights, "gst_min_match_length": 2,
               "winnowing_kgram": 2, "winnowing_window": 2}
    texts = {"a": "int total = value + 1;", "b": "int total = value + 2;", "copy": "int total = value + 1;"}
    expected = [similarity.calculate_similarity(texts["a"], texts[key], **options) for key in ("a", "b", "copy")]
    monkeypatch.setattr(similarity, "tokenize_for_lexical_matching", tokenize)
    scored = evaluation.score_retrieval_dataset(dataset, similarity_options=options)
    assert scored["similarity_score"].tolist() == expected
    assert len(calls) == (2 if weights.get("winnowing") or weights.get("gst") else 0)
