import re
from threading import RLock
import zlib

import numpy as np

from .chunking import chunk_text
from .model_routing import (
    load_hf_model_info,
    normalize_vector_backend_name,
    resolve_vector_backend,
)


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
_MODEL_NAME_TOKEN_LENGTH_CACHE = {}
_MODEL_NAME_TOKEN_LENGTH_CACHE_LOCK = RLock()


def available_similarity_functions():
    return ("cosine", "dot", "euclidean", "manhattan")


def similarity_function_score_range(name, normalize_score=False):
    selected = normalize_similarity_function_name(name)
    if normalize_score:
        return (0.0, 1.0)
    if selected == "cosine":
        return (-1.0, 1.0)
    if selected == "dot":
        return (float("-inf"), float("inf"))
    return (float("-inf"), 0.0)


def similarity_function_uses_distance_scale(name):
    selected = normalize_similarity_function_name(name)
    return selected in ("euclidean", "manhattan")


def similarity_function_is_unbounded(name):
    selected = normalize_similarity_function_name(name)
    return selected in ("dot", "euclidean", "manhattan")


def normalized_similarity_score(score, similarity_function):
    selected = normalize_similarity_function_name(similarity_function)
    numeric_score = float(score)
    if similarity_function_uses_distance_scale(selected):
        return _distance_to_similarity(abs(numeric_score))
    return float(np.clip(numeric_score, 0.0, 1.0))


def raw_distance_to_similarity(distance):
    numeric_distance = max(0.0, float(distance))
    return 1.0 / (1.0 + numeric_distance)


def _distance_to_similarity(distance):
    return raw_distance_to_similarity(distance)


def _maybe_normalize_single_vector_score(score, similarity_function, normalize_score=False):
    if normalize_score:
        return normalized_similarity_score(score, similarity_function)
    return float(score)


def _distance_score(distance, normalize_score=False):
    numeric_distance = max(0.0, float(distance))
    if normalize_score:
        return _distance_to_similarity(numeric_distance)
    return -numeric_distance


def _raw_cosine_score(left_vector, right_vector):
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 0:
        return 0.0
    score = np.dot(left_vector, right_vector) / denominator
    return float(np.clip(score, -1.0, 1.0))


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
    with _MODEL_NAME_TOKEN_LENGTH_CACHE_LOCK:
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

    requested = _coerce_token_length(max_token_length)
    if requested is None:
        return model
    detected = _detect_token_length_from_model(model)
    fallback_cap = resolve_max_token_length(
        requested,
        detected_max_token_length=detected,
    )

    targets = [model]
    targets.extend((getattr(model, "_modules", None) or {}).values())
    for target in targets:
        for attr_name in (
            "document_length",
            "query_length",
            "max_seq_length",
            "max_length",
        ):
            _lower_model_length_attribute(
                target,
                attr_name,
                requested,
                fallback_cap=fallback_cap,
            )
        _lower_query_expansion(target, requested)
        tokenizer = getattr(target, "tokenizer", None)
        if tokenizer is not None:
            _lower_model_length_attribute(
                tokenizer,
                "model_max_length",
                requested,
                fallback_cap=fallback_cap,
            )
    return model


def _lower_model_length_attribute(target, attr_name, requested, fallback_cap=None):
    try:
        current_value = getattr(target, attr_name)
    except Exception:
        return
    current = _coerce_token_length(current_value)
    current_cap = current if current is not None else fallback_cap
    effective = min(requested, current_cap) if current_cap is not None else requested
    try:
        setattr(target, attr_name, effective)
    except Exception:
        pass


def _lower_query_expansion(target, requested):
    expansion = getattr(target, "query_expansion", None)
    if not isinstance(expansion, dict):
        return
    current = _coerce_token_length(expansion.get("length"))
    if current is None or current <= requested:
        return
    updated = dict(expansion)
    updated["length"] = requested
    try:
        target.query_expansion = updated
    except Exception:
        pass


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
        from sentence_transformers.sentence_transformer.modules import Pooling
    except ImportError:  # pragma: no cover - optional dependency during partial installs
        return None, None

    for module_name, module in reversed(list(getattr(model, "_modules", {}).items())):
        if isinstance(module, Pooling):
            return module_name, module
    return None, None


def _detect_current_pooling_method(pooling_module):
    pooling_mode = getattr(pooling_module, "pooling_mode", None)
    if isinstance(pooling_mode, str):
        return pooling_mode
    modes = tuple(pooling_mode or ())
    if len(modes) == 1:
        return modes[0]
    return modes or None


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
    if isinstance(current_method, tuple):
        raise ValueError(
            "Custom pooling_method is only supported for sentence-transformers models that use a single pooling mode."
        )

    word_dimension = int(getattr(pooling_module, "embedding_dimension", 0) or 0)
    output_dimension = None
    get_dimension = getattr(pooling_module, "get_embedding_dimension", None)
    if callable(get_dimension):
        output_dimension = int(get_dimension() or 0)
    if word_dimension <= 0:
        return model
    if output_dimension not in (None, 0, word_dimension):
        raise ValueError(
            "Custom pooling_method is only supported for sentence-transformers models that use a single pooling mode."
        )

    from sentence_transformers.sentence_transformer.modules import Pooling

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


def _load_multivector_model(model_name, device="auto", max_token_length=None):
    try:
        from sentence_transformers import MultiVectorEncoder
    except ImportError as exc:
        raise ImportError(
            "Multi-vector scoring requires the optional 'sentence-transformers>=6' package."
        ) from exc
    model = MultiVectorEncoder(
        model_name,
        device=device,
        similarity_fn_name="meanmaxsim",
    )
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
    if backend == "auto":
        backend = resolve_vector_backend(
            backend,
            model_name=model_name,
            model_info=load_hf_model_info(model_name),
        )
    if backend == "static_hash":
        return None
    if backend == "model2vec":
        return _load_model2vec_model(model_name, max_token_length=max_token_length)
    if backend == "multivector":
        return _load_multivector_model(
            model_name,
            device=device,
            max_token_length=max_token_length,
        )
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


def _pairwise_similarity_with_sentence_transformers(
    left,
    right,
    similarity_function="cosine",
    normalize_score=False,
):
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
    return _maybe_normalize_single_vector_score(
        _extract_scalar_score(score),
        selected,
        normalize_score=normalize_score,
    )


def single_vector_similarity(left, right, similarity_function="cosine", normalize_score=False):
    selected = normalize_similarity_function_name(similarity_function)
    try:
        return _pairwise_similarity_with_sentence_transformers(
            left,
            right,
            similarity_function=selected,
            normalize_score=normalize_score,
        )
    except Exception:
        left_vector = np.asarray(left, dtype=float)
        right_vector = np.asarray(right, dtype=float)
        if selected == "dot":
            return _maybe_normalize_single_vector_score(
                np.dot(left_vector, right_vector),
                selected,
                normalize_score=normalize_score,
            )
        if selected == "euclidean":
            return _distance_score(
                np.linalg.norm(left_vector - right_vector),
                normalize_score=normalize_score,
            )
        if selected == "manhattan":
            return _distance_score(
                np.abs(left_vector - right_vector).sum(),
                normalize_score=normalize_score,
            )
        return _maybe_normalize_single_vector_score(
            _raw_cosine_score(left_vector, right_vector),
            selected,
            normalize_score=normalize_score,
        )


def _encode_to_numpy(model, inputs):
    if hasattr(model, "encode"):
        try:
            vectors = model.encode(inputs, convert_to_numpy=True)
        except TypeError:
            vectors = model.encode(inputs)
        return np.asarray(vectors, dtype=float)
    raise ValueError("The selected model does not provide an encode() method.")


def _encode_multivector_to_numpy(model, inputs, is_query=False):
    method_name = "encode_query" if is_query else "encode_document"
    encoder = getattr(model, method_name, None)
    if not callable(encoder):
        raise ValueError(
            "The selected multi-vector model must provide encode_query() and encode_document()."
        )
    vectors = encoder(
        list(inputs),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(vectors, list):
        return [_ensure_2d_vectors(item) for item in vectors]
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 3:
        return [_ensure_2d_vectors(item) for item in array]
    if len(inputs) == 1:
        return [_ensure_2d_vectors(array)]
    raise ValueError("Multi-vector encoding returned an unexpected result shape.")


def _ensure_2d_vectors(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _stack_multivectors(vectors):
    if isinstance(vectors, (list, tuple)):
        matrices = []
        for item in vectors:
            matrix = _ensure_2d_vectors(item)
            if matrix.size:
                matrices.append(matrix)
        if not matrices:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(matrices)

    array = np.asarray(vectors, dtype=np.float32)
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
    is_query=False,
):
    inputs = []
    document_slices = []
    for code in codes:
        method = (chunking_method or "none").strip().lower()
        use_full_document = method in ("none", "document")

        if use_full_document:
            document_inputs = [code or ""]
        else:
            document_inputs = chunk_text(
                code,
                method=method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chunks=max_chunks,
                chunk_language=chunk_language,
                chunker_options=chunker_options,
            )
        start = len(inputs)
        inputs.extend(document_inputs)
        document_slices.append((start, len(inputs)))

    encoded = _encode_multivector_to_numpy(model, inputs, is_query=bool(is_query))
    return [_stack_multivectors(encoded[start:end]) for start, end in document_slices]


def multivector_similarity_matrix(left_embeddings, right_embeddings=None, bidirectional=False):
    from sentence_transformers.util import mean_maxsim

    left = [_ensure_2d_vectors(item) for item in left_embeddings]
    same_collection = right_embeddings is None
    right = left if same_collection else [
        _ensure_2d_vectors(item) for item in right_embeddings
    ]
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)

    scores = mean_maxsim(left, right).detach().cpu().numpy()
    if bidirectional:
        reverse = (
            scores.T
            if same_collection
            else mean_maxsim(right, left).detach().cpu().numpy().T
        )
        scores = (scores + reverse) * 0.5
    return np.clip(np.asarray(scores, dtype=np.float32), -1.0, 1.0)


def multivector_similarity(left, right, bidirectional=True, vector_backend=None):
    normalize_vector_backend_name(vector_backend or "multivector")
    scores = multivector_similarity_matrix(
        [left],
        [right],
        bidirectional=bidirectional,
    )
    return float(scores[0, 0])
