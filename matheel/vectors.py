import re
import zlib

import numpy as np

from .chunking import chunk_text


_STATIC_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]")


def available_vector_backends():
    return ("transformer", "static_hash", "multivector")


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


def build_multivector_embeddings(
    model,
    codes,
    chunking_method="tokens",
    chunk_size=120,
    chunk_overlap=20,
    max_chunks=0,
):
    embeddings_by_doc = []

    for code in codes:
        method = chunking_method
        if (method or "none").strip().lower() in ("none", "document"):
            method = "tokens"

        chunks = chunk_text(
            code,
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
        chunk_embeddings = np.asarray(chunk_embeddings, dtype=float)
        if chunk_embeddings.ndim == 1:
            chunk_embeddings = chunk_embeddings.reshape(1, -1)
        embeddings_by_doc.append(chunk_embeddings)

    return embeddings_by_doc


def multivector_directional_maxsim(left, right):
    left_vectors = np.asarray(left, dtype=float)
    right_vectors = np.asarray(right, dtype=float)

    if left_vectors.ndim == 1:
        left_vectors = left_vectors.reshape(1, -1)
    if right_vectors.ndim == 1:
        right_vectors = right_vectors.reshape(1, -1)
    if left_vectors.size == 0 or right_vectors.size == 0:
        return 0.0

    left_norms = np.linalg.norm(left_vectors, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_vectors, axis=1, keepdims=True)
    left_safe = np.where(left_norms > 0, left_vectors / np.maximum(left_norms, 1e-12), 0.0)
    right_safe = np.where(right_norms > 0, right_vectors / np.maximum(right_norms, 1e-12), 0.0)

    similarity_matrix = np.matmul(left_safe, right_safe.T)
    best_matches = similarity_matrix.max(axis=1)
    return float(np.clip(best_matches.mean(), -1.0, 1.0))


def multivector_similarity(left, right, bidirectional=True):
    forward = multivector_directional_maxsim(left, right)
    if not bidirectional:
        return float(forward)
    reverse = multivector_directional_maxsim(right, left)
    return float((forward + reverse) * 0.5)
