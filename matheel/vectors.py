import re
import zlib

import numpy as np

from .chunking import chunk_text
from .model_routing import available_vector_backends, normalize_vector_backend_name


_STATIC_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]")


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


def _load_sentence_transformer_model(model_name, device="auto"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _load_model2vec_model(model_name):
    from model2vec import StaticModel

    if hasattr(StaticModel, "from_pretrained"):
        return StaticModel.from_pretrained(model_name)
    return StaticModel(model_name)


def _load_pylate_model(model_name, device="auto"):
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
                    return model_class.from_pretrained(model_name, device=device)
                except TypeError:
                    return model_class.from_pretrained(model_name)
            try:
                return model_class(model_name, device=device)
            except TypeError:
                return model_class(model_name)

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def load_vector_model(model_name, vector_backend="auto", device="cpu"):
    backend = normalize_vector_backend_name(vector_backend)
    if backend == "static_hash":
        return None
    if backend == "model2vec":
        return _load_model2vec_model(model_name)
    if backend == "pylate":
        return _load_pylate_model(model_name, device=device)
    return _load_sentence_transformer_model(model_name, device=device)


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
    chunking_method="tokens",
    chunk_size=120,
    chunk_overlap=20,
    max_chunks=0,
    chunk_language="text",
    chunker_options=None,
):
    embeddings_by_doc = []

    for code in codes:
        method = chunking_method
        use_full_document = _is_pylate_model(model) and (method or "none").strip().lower() in ("none", "document")
        if not use_full_document and (method or "none").strip().lower() in ("none", "document"):
            method = "tokens"

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
