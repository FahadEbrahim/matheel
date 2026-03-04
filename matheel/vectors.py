import re
import zlib

import numpy as np

from .chunking import chunk_text
from .model_routing import normalize_vector_backend_name


_STATIC_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]")
_DEFAULT_MAX_TOKEN_LENGTH = 512
_MAX_TOKEN_LENGTH_SENTINEL = 1_000_000
_SIMILARITY_FUNCTION_ALIASES = {
    "cosine": "cosine",
    "cos": "cosine",
    "cos_sim": "cosine",
    "dot": "dot",
    "dot_product": "dot",
    "dot_score": "dot",
    "euclidean": "euclidean",
    "euclidean_sim": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "manhattan_sim": "manhattan",
    "l1": "manhattan",
}
_POOLING_METHOD_ALIASES = {
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "max": "max",
    "cls": "cls",
    "cls_token": "cls",
    "lasttoken": "lasttoken",
    "last_token": "lasttoken",
    "mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "mean_sqrt_len": "mean_sqrt_len_tokens",
    "weightedmean": "weightedmean",
    "weighted_mean": "weightedmean",
}
_POOLING_MODE_FLAGS = {
    "cls": "pooling_mode_cls_token",
    "max": "pooling_mode_max_tokens",
    "mean": "pooling_mode_mean_tokens",
    "mean_sqrt_len_tokens": "pooling_mode_mean_sqrt_len_tokens",
    "weightedmean": "pooling_mode_weightedmean_tokens",
    "lasttoken": "pooling_mode_lasttoken",
}
_MODEL_NAME_TOKEN_LENGTH_CACHE = {}


def available_similarity_functions():
    return ("cosine", "dot", "euclidean", "manhattan")


def normalize_similarity_function_name(name):
    key = str(name or "cosine").strip().lower()
    normalized = _SIMILARITY_FUNCTION_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(available_similarity_functions())
        raise ValueError(
            f"Unsupported similarity function: {name}. Supported similarity functions: {supported}"
        )
    return normalized


def available_pooling_methods():
    return ("mean", "max", "cls", "lasttoken", "mean_sqrt_len_tokens", "weightedmean")


def normalize_pooling_method_name(name):
    key = str(name or "mean").strip().lower()
    normalized = _POOLING_METHOD_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(available_pooling_methods())
        raise ValueError(f"Unsupported pooling method: {name}. Supported pooling methods: {supported}")
    return normalized


def _coerce_token_length(value):
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    if numeric_value <= 0 or numeric_value >= _MAX_TOKEN_LENGTH_SENTINEL:
        return None
    return numeric_value


def resolve_max_token_length(max_token_length, detected_max_token_length=None):
    requested = _coerce_token_length(max_token_length)
    if requested is None:
        return None
    detected = _coerce_token_length(detected_max_token_length)
    if detected is None:
        return requested
    return min(requested, detected)


def _detect_token_length_from_tokenizer(tokenizer):
    if tokenizer is None:
        return None
    for attr_name in ("model_max_length", "max_seq_length"):
        detected = _coerce_token_length(getattr(tokenizer, attr_name, None))
        if detected is not None:
            return detected
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    return _coerce_token_length(init_kwargs.get("model_max_length"))


def _detect_token_length_from_config(config):
    if config is None:
        return None
    for attr_name in ("max_position_embeddings", "n_positions"):
        detected = _coerce_token_length(getattr(config, attr_name, None))
        if detected is not None:
            return detected
    return None


def _detect_token_length_from_model(model):
    if model is None:
        return None
    for attr_name in ("document_length", "query_length", "max_seq_length", "max_length"):
        detected = _coerce_token_length(getattr(model, attr_name, None))
        if detected is not None:
            return detected

    tokenizer = getattr(model, "tokenizer", None)
    detected = _detect_token_length_from_tokenizer(tokenizer)
    if detected is not None:
        return detected

    config = getattr(model, "config", None)
    detected = _detect_token_length_from_config(config)
    if detected is not None:
        return detected

    nested_model = getattr(model, "model", None)
    if nested_model is not None and nested_model is not model:
        return _detect_token_length_from_model(nested_model)
    return None


def _detect_token_length_from_model_name(model_name):
    model_key = str(model_name or "").strip()
    if not model_key:
        return None
    if model_key in _MODEL_NAME_TOKEN_LENGTH_CACHE:
        return _MODEL_NAME_TOKEN_LENGTH_CACHE[model_key]

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_key)
    except Exception:
        tokenizer = None
    detected = _detect_token_length_from_tokenizer(tokenizer)
    if detected is not None:
        _MODEL_NAME_TOKEN_LENGTH_CACHE[model_key] = detected
        return detected

    try:
        config = AutoConfig.from_pretrained(model_key)
    except Exception:
        config = None
    detected = _detect_token_length_from_config(config)
    _MODEL_NAME_TOKEN_LENGTH_CACHE[model_key] = detected
    return detected


def detect_model_max_token_length(model=None, model_name=None, default=_DEFAULT_MAX_TOKEN_LENGTH):
    detected = _detect_token_length_from_model(model)
    if detected is not None:
        return detected

    detected = _detect_token_length_from_model_name(model_name)
    if detected is not None:
        return detected

    fallback = _coerce_token_length(default)
    if fallback is not None:
        return fallback
    return _DEFAULT_MAX_TOKEN_LENGTH


def configure_model_max_token_length(model, max_token_length=None):
    if model is None:
        return model

    detected = _detect_token_length_from_model(model)
    selected = resolve_max_token_length(
        max_token_length,
        detected_max_token_length=detected,
    )
    if selected is None:
        return model

    has_document_length = hasattr(model, "document_length")
    has_query_length = hasattr(model, "query_length")

    if has_document_length:
        try:
            model.document_length = selected
        except Exception:
            pass
    elif has_query_length:
        try:
            model.query_length = selected
        except Exception:
            pass
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = selected
    if hasattr(model, "max_length"):
        try:
            model.max_length = selected
        except Exception:
            pass

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = selected
    return model


def tokenize_for_static_vectors(text, lowercase=True):
    tokens = _STATIC_TOKEN_RE.findall(text or "")
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def build_static_hash_vector(text, dim=256, lowercase=True):
    size = max(8, int(dim or 0))
    vector = np.zeros(size, dtype=float)
    tokens = tokenize_for_static_vectors(text, lowercase=lowercase)
    if not tokens:
        return vector

    for token in tokens:
        index = zlib.crc32(token.encode("utf-8")) % size
        vector[index] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def build_static_hash_vectors(codes, dim=256, lowercase=True):
    return [build_static_hash_vector(code, dim=dim, lowercase=lowercase) for code in codes]


def _find_sentence_transformer_pooling(model):
    try:
        from sentence_transformers.models import Pooling
    except ImportError:  # pragma: no cover - optional dependency during partial installs
        return None, None

    for module_name, module in reversed(list(getattr(model, "_modules", {}).items())):
        if isinstance(module, Pooling):
            return module_name, module
    return None, None


def _detect_current_pooling_method(pooling_module):
    for method_name, flag_name in _POOLING_MODE_FLAGS.items():
        if getattr(pooling_module, flag_name, False):
            return method_name
    return None


def configure_sentence_transformer_pooling(model, pooling_method="mean"):
    selected_method = normalize_pooling_method_name(pooling_method)
    module_name, pooling_module = _find_sentence_transformer_pooling(model)
    if pooling_module is None:
        if selected_method == "mean":
            return model
        raise ValueError("The selected sentence-transformers model does not expose a Pooling module.")

    current_method = _detect_current_pooling_method(pooling_module)
    if current_method == selected_method:
        return model

    word_dimension = int(getattr(pooling_module, "word_embedding_dimension", 0) or 0)
    output_dimension = None
    get_dimension = getattr(pooling_module, "get_sentence_embedding_dimension", None)
    if callable(get_dimension):
        output_dimension = int(get_dimension() or 0)
    if word_dimension <= 0:
        return model
    if output_dimension not in (None, 0, word_dimension):
        raise ValueError(
            "Custom pooling_method is only supported for sentence-transformers models that use a single pooling mode."
        )

    from sentence_transformers.models import Pooling

    model._modules[module_name] = Pooling(
        word_dimension,
        pooling_mode=selected_method,
        include_prompt=bool(getattr(pooling_module, "include_prompt", True)),
    )
    return model


def _load_sentence_transformer_model(
    model_name,
    device="auto",
    similarity_function="cosine",
    pooling_method="mean",
    max_token_length=None,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        similarity_fn_name=normalize_similarity_function_name(similarity_function),
    )
    model = configure_sentence_transformer_pooling(model, pooling_method=pooling_method)
    return configure_model_max_token_length(model, max_token_length=max_token_length)


def _load_model2vec_model(model_name, max_token_length=None):
    from model2vec import StaticModel

    if hasattr(StaticModel, "from_pretrained"):
        model = StaticModel.from_pretrained(model_name)
    else:
        model = StaticModel(model_name)
    return configure_model_max_token_length(model, max_token_length=max_token_length)


def _load_pylate_model(model_name, device="auto", max_token_length=None):
    try:
        from pylate import models as pylate_models
    except ImportError:
        pylate_models = None

    if pylate_models is not None:
        for attr_name in ("ColBERT", "SentenceTransformer", "LateInteractionModel"):
            model_class = getattr(pylate_models, attr_name, None)
            if model_class is None:
                continue
            if hasattr(model_class, "from_pretrained"):
                try:
                    model = model_class.from_pretrained(model_name, device=device)
                except TypeError:
                    model = model_class.from_pretrained(model_name)
                return configure_model_max_token_length(model, max_token_length=max_token_length)
            try:
                model = model_class(model_name, device=device)
            except TypeError:
                model = model_class(model_name)
            return configure_model_max_token_length(model, max_token_length=max_token_length)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return configure_model_max_token_length(model, max_token_length=max_token_length)


def load_vector_model(
    model_name,
    vector_backend="auto",
    device="cpu",
    similarity_function="cosine",
    pooling_method="mean",
    max_token_length=None,
):
    backend = normalize_vector_backend_name(vector_backend)
    if backend == "static_hash":
        return None
    if backend == "model2vec":
        return _load_model2vec_model(model_name, max_token_length=max_token_length)
    if backend == "pylate":
        return _load_pylate_model(model_name, device=device, max_token_length=max_token_length)
    return _load_sentence_transformer_model(
        model_name,
        device=device,
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
    )


def _extract_scalar_score(value):
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


def _pairwise_similarity_with_sentence_transformers(left, right, similarity_function="cosine"):
    from sentence_transformers import util

    left_vector = np.asarray(left, dtype=float).reshape(1, -1)
    right_vector = np.asarray(right, dtype=float).reshape(1, -1)
    selected = normalize_similarity_function_name(similarity_function)
    if selected == "dot":
        score = util.pairwise_dot_score(left_vector, right_vector)
    elif selected == "euclidean":
        score = util.pairwise_euclidean_sim(left_vector, right_vector)
    elif selected == "manhattan":
        score = util.pairwise_manhattan_sim(left_vector, right_vector)
    else:
        score = util.pairwise_cos_sim(left_vector, right_vector)
    return _extract_scalar_score(score)


def single_vector_similarity(left, right, similarity_function="cosine"):
    selected = normalize_similarity_function_name(similarity_function)
    try:
        return _pairwise_similarity_with_sentence_transformers(left, right, similarity_function=selected)
    except Exception:
        left_vector = np.asarray(left, dtype=float)
        right_vector = np.asarray(right, dtype=float)
        if selected == "dot":
            return float(np.dot(left_vector, right_vector))
        if selected == "euclidean":
            return float(-np.linalg.norm(left_vector - right_vector))
        if selected == "manhattan":
            return float(-np.abs(left_vector - right_vector).sum())
        denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        if denominator <= 0:
            return 0.0
        score = np.dot(left_vector, right_vector) / denominator
        return float(np.clip(score, -1.0, 1.0))


def _encode_to_numpy(model, inputs):
    if hasattr(model, "encode"):
        try:
            vectors = model.encode(inputs, convert_to_numpy=True)
        except TypeError:
            vectors = model.encode(inputs)
        return np.asarray(vectors, dtype=float)
    raise ValueError("The selected model does not provide an encode() method.")


def _is_pylate_model(model):
    if model is None:
        return False
    model_type = type(model)
    return "pylate" in str(model_type.__module__).lower() or model_type.__name__.lower() == "colbert"


def _encode_multivector_to_numpy(model, inputs, is_query=False):
    if _is_pylate_model(model):
        try:
            return model.encode(inputs, convert_to_numpy=True, is_query=is_query)
        except TypeError:
            return model.encode(inputs, convert_to_numpy=True)
    if hasattr(model, "encode_as_sequence"):
        try:
            vectors = model.encode_as_sequence(inputs, convert_to_numpy=True)
        except TypeError:
            vectors = model.encode_as_sequence(inputs)
        return vectors
    return _encode_to_numpy(model, inputs)


def _ensure_2d_vectors(vectors):
    array = np.asarray(vectors, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _stack_multivectors(vectors):
    if isinstance(vectors, list):
        matrices = []
        for item in vectors:
            matrix = _ensure_2d_vectors(item)
            if matrix.size:
                matrices.append(matrix)
        if not matrices:
            return np.zeros((0, 1), dtype=float)
        return np.vstack(matrices)

    array = np.asarray(vectors, dtype=float)
    if array.ndim == 3:
        return array.reshape(-1, array.shape[-1])
    return _ensure_2d_vectors(array)


def encode_single_vectors(model, codes, vector_backend="sentence_transformers", static_vector_dim=256, static_vector_lowercase=True):
    backend = normalize_vector_backend_name(vector_backend)
    if backend == "static_hash":
        return build_static_hash_vectors(codes, dim=static_vector_dim, lowercase=static_vector_lowercase)
    if not codes:
        return []

    vectors = _encode_to_numpy(model, codes if len(codes) > 1 else codes[0])
    if len(codes) == 1:
        return [np.asarray(vectors, dtype=float)]
    return [np.asarray(vector, dtype=float) for vector in vectors]


def build_chunked_single_vectors(
    model,
    codes,
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_language="text",
    chunker_options=None,
):
    embeddings_by_doc = []
    for code in codes:
        chunks = chunk_text(
            code,
            method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
        )
        vectors = _encode_to_numpy(model, chunks if len(chunks) > 1 else chunks[0])
        vectors = _ensure_2d_vectors(vectors)
        embeddings_by_doc.append(vectors)
    return embeddings_by_doc


def build_multivector_embeddings(
    model,
    codes,
    chunking_method="none",
    chunk_size=120,
    chunk_overlap=20,
    max_chunks=0,
    chunk_language="text",
    chunker_options=None,
):
    embeddings_by_doc = []

    for code in codes:
        method = (chunking_method or "none").strip().lower()
        use_full_document = method in ("none", "document")

        if use_full_document:
            inputs = code or ""
        else:
            chunks = chunk_text(
                code,
                method=method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chunks=max_chunks,
                chunk_language=chunk_language,
                chunker_options=chunker_options,
            )
            inputs = chunks if len(chunks) > 1 else chunks[0]
        chunk_embeddings = _encode_multivector_to_numpy(model, inputs, is_query=False)
        chunk_embeddings = _stack_multivectors(chunk_embeddings)
        embeddings_by_doc.append(chunk_embeddings)

    return embeddings_by_doc


def multivector_directional_maxsim(left, right):
    left_vectors = _ensure_2d_vectors(left)
    right_vectors = _ensure_2d_vectors(right)

    if left_vectors.size == 0 or right_vectors.size == 0:
        return 0.0

    left_norms = np.linalg.norm(left_vectors, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_vectors, axis=1, keepdims=True)
    left_safe = np.where(left_norms > 0, left_vectors / np.maximum(left_norms, 1e-12), 0.0)
    right_safe = np.where(right_norms > 0, right_vectors / np.maximum(right_norms, 1e-12), 0.0)

    similarity_matrix = np.matmul(left_safe, right_safe.T)
    best_matches = similarity_matrix.max(axis=1)
    return float(np.clip(best_matches.mean(), -1.0, 1.0))


def multivector_similarity(left, right, bidirectional=True, vector_backend=None):
    if normalize_vector_backend_name(vector_backend or "pylate") == "pylate" and _pylate_scores_available():
        return _pylate_multivector_similarity(left, right, bidirectional=bidirectional)

    forward = multivector_directional_maxsim(left, right)
    if not bidirectional:
        return float(forward)
    reverse = multivector_directional_maxsim(right, left)
    return float((forward + reverse) * 0.5)


def _pylate_scores_available():
    try:
        from pylate.scores import colbert_scores_pairwise  # noqa: F401
    except ImportError:
        return False
    return True


def _pylate_directional_score(left, right):
    from pylate.scores import colbert_scores_pairwise

    left_vectors = _ensure_2d_vectors(left)
    right_vectors = _ensure_2d_vectors(right)
    if left_vectors.size == 0 or right_vectors.size == 0:
        return 0.0

    raw_score = colbert_scores_pairwise([left_vectors], [right_vectors])[0]
    raw_value = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
    token_count = max(1, int(left_vectors.shape[0]))
    return float(np.clip(raw_value / float(token_count), -1.0, 1.0))


def _pylate_multivector_similarity(left, right, bidirectional=True):
    forward = _pylate_directional_score(left, right)
    if not bidirectional:
        return forward
    reverse = _pylate_directional_score(right, left)
    return float((forward + reverse) * 0.5)
