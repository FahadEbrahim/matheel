import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import matheel.vectors as vectors_module
from matheel.vectors import (
    available_pooling_methods,
    available_similarity_functions,
    build_static_hash_vector,
    configure_model_max_token_length,
    configure_sentence_transformer_pooling,
    detect_model_max_token_length,
    multivector_similarity,
    resolve_max_token_length,
    similarity_function_score_range,
    single_vector_similarity,
)


def test_static_hash_vector_is_normalized_for_same_token_bag():
    left = build_static_hash_vector("def add(a, b):\n    return a + b\n", dim=64, lowercase=True)
    right = build_static_hash_vector("def add(b, a):\n    return b + a\n", dim=64, lowercase=True)

    assert round(float(left.dot(right)), 6) == 1.0


def test_multivector_similarity_is_one_for_identical_vectors():
    left = [[1.0, 0.0], [0.0, 1.0]]
    right = [[1.0, 0.0], [0.0, 1.0]]

    assert multivector_similarity(left, right, bidirectional=True) == 1.0


def test_single_vector_similarity_supports_all_supported_functions():
    left = [1.0, 0.0]
    right = [0.0, 1.0]

    assert available_similarity_functions() == ("cosine", "dot", "euclidean", "manhattan")
    assert similarity_function_score_range("cosine") == (-1.0, 1.0)
    assert similarity_function_score_range("dot") == (float("-inf"), float("inf"))
    assert similarity_function_score_range("euclidean") == (float("-inf"), 0.0)
    assert similarity_function_score_range("manhattan") == (float("-inf"), 0.0)
    for name in available_similarity_functions():
        assert similarity_function_score_range(name, normalize_score=True) == (0.0, 1.0)
    assert single_vector_similarity(left, left, similarity_function="cosine") == 1.0
    assert single_vector_similarity(left, left, similarity_function="dot") == 1.0
    assert single_vector_similarity(left, left, similarity_function="euclidean") == 0.0
    assert single_vector_similarity(left, left, similarity_function="manhattan") == 0.0
    assert single_vector_similarity(left, right, similarity_function="cosine") == 0.0
    assert single_vector_similarity(left, right, similarity_function="dot") == 0.0
    assert single_vector_similarity(left, right, similarity_function="euclidean") == pytest.approx(-(2.0 ** 0.5))
    assert single_vector_similarity(left, right, similarity_function="manhattan") == -2.0
    assert single_vector_similarity(left, right, similarity_function="euclidean", normalize_score=True) == pytest.approx(
        1.0 / (1.0 + (2.0 ** 0.5))
    )
    assert single_vector_similarity(left, right, similarity_function="manhattan", normalize_score=True) == pytest.approx(
        1.0 / 3.0
    )
    assert single_vector_similarity([2.0, 0.0], [2.0, 0.0], similarity_function="dot") == 4.0
    assert single_vector_similarity([2.0, 0.0], [2.0, 0.0], similarity_function="dot", normalize_score=True) == 1.0


def test_configure_sentence_transformer_pooling_replaces_single_pooling_mode():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    pooling_class = sentence_transformers.models.Pooling

    class DummyModel:
        def __init__(self):
            self._modules = {"1": pooling_class(8)}

    model = DummyModel()
    configured = configure_sentence_transformer_pooling(model, pooling_method="max")

    assert available_pooling_methods() == (
        "mean",
        "max",
        "cls",
        "lasttoken",
        "mean_sqrt_len_tokens",
        "weightedmean",
    )
    assert configured is model
    assert configured._modules["1"].pooling_mode_max_tokens is True
    assert configured._modules["1"].pooling_mode_mean_tokens is False


def test_configure_sentence_transformer_pooling_rejects_multi_mode_pooling():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    pooling_class = sentence_transformers.models.Pooling

    class DummyModel:
        def __init__(self):
            self._modules = {
                "1": pooling_class(
                    8,
                    pooling_mode_cls_token=True,
                    pooling_mode_mean_tokens=True,
                )
            }

    with pytest.raises(ValueError):
        configure_sentence_transformer_pooling(DummyModel(), pooling_method="max")


def test_detect_model_max_token_length_prefers_explicit_model_limit():
    class DummyTokenizer:
        model_max_length = 2048

    class DummyConfig:
        max_position_embeddings = 1024

    class DummyModel:
        max_seq_length = 384
        tokenizer = DummyTokenizer()
        config = DummyConfig()

    assert detect_model_max_token_length(model=DummyModel()) == 384


def test_detect_model_max_token_length_ignores_large_tokenizer_sentinel():
    class DummyTokenizer:
        model_max_length = 10**30

    class DummyConfig:
        max_position_embeddings = 1024

    class DummyModel:
        tokenizer = DummyTokenizer()
        config = DummyConfig()

    assert detect_model_max_token_length(model=DummyModel()) == 1024


def test_model_name_token_length_cache_is_shared_across_concurrent_loads(monkeypatch):
    calls = []

    class FakeTokenizer:
        model_max_length = 384

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            calls.append(model_name)
            time.sleep(0.01)
            return FakeTokenizer()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name):
            raise AssertionError(f"config lookup should not run after tokenizer hit: {model_name}")

    vectors_module._MODEL_NAME_TOKEN_LENGTH_CACHE.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=FakeAutoConfig, AutoTokenizer=FakeAutoTokenizer),
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: detect_model_max_token_length(model_name="demo-model"), range(8)))

    assert results == [384] * 8
    assert calls == ["demo-model"]


def test_configure_model_max_token_length_clamps_to_detected_limit():
    class DummyTokenizer:
        model_max_length = 512

    class DummyModel:
        max_seq_length = 512
        tokenizer = DummyTokenizer()

    model = DummyModel()
    configure_model_max_token_length(model, max_token_length=256)

    assert model.max_seq_length == 256
    assert model.tokenizer.model_max_length == 256
    assert resolve_max_token_length(1024, detected_max_token_length=512) == 512


def test_configure_model_max_token_length_updates_pylate_style_lengths():
    class DummyModel:
        document_length = 180
        query_length = 32
        max_seq_length = 179

    model = DummyModel()
    configure_model_max_token_length(model, max_token_length=96)

    assert detect_model_max_token_length(model=model) == 96
    assert model.document_length == 96
    assert model.query_length == 32
    assert model.max_seq_length == 96


def test_load_vector_model_requires_pylate_package(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pylate" or name.startswith("pylate."):
            raise ImportError("blocked pylate import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="pylate"):
        vectors_module.load_vector_model("demo/model", vector_backend="pylate")
