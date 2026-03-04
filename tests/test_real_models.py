import pytest

from matheel.similarity import calculate_similarity


pytestmark = pytest.mark.integration


_SIMILAR_CODE_LEFT = """
def add_numbers(values):
    total = 0
    for item in values:
        total += item
    return total
""".strip()

_SIMILAR_CODE_RIGHT = """
def sum_values(numbers):
    result = 0
    for value in numbers:
        result += value
    return result
""".strip()

_DIFFERENT_CODE = """
def is_prime(value):
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True
""".strip()

_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_STATIC_MODEL = "Jarbas/m2v-256-paraphrase-multilingual-MiniLM-L12-v2"
_MULTIVECTOR_MODEL = "NeuML/pylate-bert-tiny"
_SKIP_ERROR_MARKERS = (
    "couldn't connect",
    "failed to establish",
    "connection error",
    "read timed out",
    "not found in local cache",
    "offline mode",
    "401 client error",
    "403 client error",
    "404 client error",
)


def _is_environment_skip_error(exc):
    if isinstance(exc, (ImportError, FileNotFoundError, ConnectionError, OSError)):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _SKIP_ERROR_MARKERS)


def _score_or_skip(**kwargs):
    try:
        return calculate_similarity(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on network/cache/backend availability
        if _is_environment_skip_error(exc):
            pytest.skip(f"Real model test skipped because the backend or model was unavailable: {exc}")
        raise


def _assert_similar_code_scores_higher(model_name, vector_backend, **kwargs):
    similar_score = _score_or_skip(
        code1=_SIMILAR_CODE_LEFT,
        code2=_SIMILAR_CODE_RIGHT,
        Ws=1.0,
        Wl=0.0,
        Wj=0.0,
        model_name=model_name,
        vector_backend=vector_backend,
        device="cpu",
        **kwargs,
    )
    different_score = _score_or_skip(
        code1=_SIMILAR_CODE_LEFT,
        code2=_DIFFERENT_CODE,
        Ws=1.0,
        Wl=0.0,
        Wj=0.0,
        model_name=model_name,
        vector_backend=vector_backend,
        device="cpu",
        **kwargs,
    )

    assert similar_score > different_score


def test_real_sentence_transformer_dense_model_uses_code():
    pytest.importorskip("sentence_transformers")
    _assert_similar_code_scores_higher(
        _DENSE_MODEL,
        "sentence_transformers",
        similarity_function="dot",
        pooling_method="max",
    )


def test_real_model2vec_static_model_uses_code():
    pytest.importorskip("model2vec")
    _assert_similar_code_scores_higher(
        _STATIC_MODEL,
        "model2vec",
        static_vector_dim=256,
    )


def test_real_pylate_multivector_model_uses_code():
    pytest.importorskip("pylate")
    _assert_similar_code_scores_higher(
        _MULTIVECTOR_MODEL,
        "pylate",
        chunking_method="tokens",
        chunk_size=32,
        chunk_overlap=8,
    )
