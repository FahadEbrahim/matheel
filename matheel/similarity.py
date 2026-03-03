import os
import zipfile
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .code_metrics import parse_component_weights, prepare_crystalbleu_context, score_code_metric_pair
from .preprocessing import preprocess_code
from .vectors import (
    available_vector_backends,
    build_multivector_embeddings,
    build_static_hash_vectors,
    multivector_similarity,
)
from .chunking import chunk_text


DEFAULT_MODEL_NAME = "uclanlp/plbart-java-cs"
RESULT_COLUMNS = ["file_name_1", "file_name_2", "similarity_score"]


def load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def available_runtime_devices():
    devices = ["cpu"]
    torch = load_torch()
    if torch is None:
        return tuple(devices)
    if getattr(torch.cuda, "is_available", lambda: False)():
        devices.insert(0, "cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend and getattr(mps_backend, "is_available", lambda: False)():
        insert_at = 1 if devices and devices[0] == "cuda" else 0
        devices.insert(insert_at, "mps")
    return tuple(dict.fromkeys(devices))


def detect_default_device():
    devices = available_runtime_devices()
    if "cuda" in devices:
        return "cuda"
    if "mps" in devices:
        return "mps"
    return "cpu"


def normalize_device(device):
    requested = (device or "auto").strip().lower()
    supported = available_runtime_devices()
    if requested in ("", "auto"):
        return detect_default_device()
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported runtime device: {device}")
    if requested != "cpu" and requested not in supported:
        raise ValueError(
            f"Requested runtime device '{requested}' is not available on this machine. Supported devices: {', '.join(supported)}"
        )
    return requested


def load_model(model_name, device="auto"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name or DEFAULT_MODEL_NAME, device=normalize_device(device))


def resolve_file_path(file_path):
    if isinstance(file_path, os.PathLike):
        return os.fspath(file_path)
    if hasattr(file_path, "name"):
        return os.fspath(file_path.name)
    return os.fspath(file_path)


def is_hidden_name(path_name):
    parts = str(path_name).replace("\\", "/").split("/")
    return any(part.startswith(".") for part in parts if part not in ("", "."))


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def read_directory_source(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            relative_name = os.path.relpath(full_path, directory_path)
            if is_hidden_name(relative_name):
                continue
            file_paths.append((relative_name, full_path))

    file_paths.sort(key=lambda item: item[0])
    file_names = [relative_name for relative_name, _ in file_paths]
    codes = [read_text_file(full_path) for _, full_path in file_paths]
    return file_names, codes


def read_zip_source(zip_path):
    with zipfile.ZipFile(zip_path, "r") as archive:
        file_names = [
            name for name in archive.namelist()
            if not name.endswith("/") and not is_hidden_name(name)
        ]
        codes = [archive.read(name).decode("utf-8", errors="ignore") for name in file_names]
    return file_names, codes


def extract_and_read_source(source_path):
    resolved_path = resolve_file_path(source_path)
    if os.path.isdir(resolved_path):
        return read_directory_source(resolved_path)
    return read_zip_source(resolved_path)


def _validate_weight(name, value):
    numeric_value = float(value)
    if numeric_value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric_value


def validate_weights(weight_semantic, weight_levenshtein, weight_jaro_winkler):
    return (
        _validate_weight("Ws", weight_semantic),
        _validate_weight("Wl", weight_levenshtein),
        _validate_weight("Wj", weight_jaro_winkler),
    )


def validate_code_metric_options(code_metric_weight, codebleu_component_weights):
    return (
        _validate_weight("code_metric_weight", code_metric_weight),
        parse_component_weights(codebleu_component_weights),
    )


def validate_vector_options(vector_backend, static_vector_dim):
    backend = (vector_backend or "transformer").strip().lower()
    if backend not in available_vector_backends():
        raise ValueError(f"Unsupported vector backend: {vector_backend}")
    return backend, max(8, int(static_vector_dim or 0))


def cosine_similarity(left, right):
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 0:
        return 0.0
    score = np.dot(left_vector, right_vector) / denominator
    return float(np.clip(score, -1.0, 1.0))


def semantic_similarity(embedding1, embedding2, vector_backend="transformer", multivector_bidirectional=True):
    backend = (vector_backend or "transformer").strip().lower()
    if backend == "multivector":
        return multivector_similarity(embedding1, embedding2, bidirectional=multivector_bidirectional)
    return cosine_similarity(embedding1, embedding2)


def aggregate_chunk_embeddings(chunk_embeddings, chunk_aggregation="mean"):
    vectors = np.asarray(chunk_embeddings, dtype=float)
    if vectors.ndim == 1:
        return vectors
    if len(vectors) == 0:
        return np.zeros(1, dtype=float)
    if chunk_aggregation == "max":
        return vectors.max(axis=0)
    if chunk_aggregation == "first":
        return vectors[0]
    return vectors.mean(axis=0)


def prepare_code(code, preprocess_mode="none"):
    return preprocess_code(code, mode=preprocess_mode)


def build_document_embedding(
    model,
    code,
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
):
    chunks = chunk_text(
        code,
        method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )
    if len(chunks) == 1:
        return np.asarray(model.encode(chunks[0], convert_to_numpy=True), dtype=float)
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
    return aggregate_chunk_embeddings(chunk_embeddings, chunk_aggregation=chunk_aggregation)


def build_document_embeddings(
    model,
    codes,
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    vector_backend="transformer",
    static_vector_dim=256,
    static_vector_lowercase=True,
):
    if not codes:
        return []

    backend = (vector_backend or "transformer").strip().lower()
    if backend == "static_hash":
        return build_static_hash_vectors(codes, dim=static_vector_dim, lowercase=static_vector_lowercase)

    if backend == "multivector":
        return build_multivector_embeddings(
            model,
            codes,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )

    if (chunking_method or "none").strip().lower() in ("none", "document"):
        vectors = model.encode(codes, convert_to_numpy=True)
        if len(codes) == 1:
            return [np.asarray(vectors, dtype=float)]
        return [np.asarray(vector, dtype=float) for vector in vectors]

    return [
        build_document_embedding(
            model,
            code,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_aggregation=chunk_aggregation,
        )
        for code in codes
    ]


def combined_similarity_from_embeddings(
    code1,
    code2,
    embedding1,
    embedding2,
    weight_semantic,
    weight_levenshtein,
    weight_jaro_winkler,
    code_metric_score=0.0,
    code_metric_weight=0.0,
    vector_backend="transformer",
    multivector_bidirectional=True,
):
    semantic_score = semantic_similarity(
        embedding1,
        embedding2,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
    )
    levenshtein_score = Levenshtein.normalized_similarity(code1, code2)
    jaro_winkler_score = JaroWinkler.normalized_similarity(code1, code2)
    combined_score = (
        weight_semantic * semantic_score
        + weight_levenshtein * levenshtein_score
        + weight_jaro_winkler * jaro_winkler_score
        + code_metric_weight * float(code_metric_score)
    )
    return float(combined_score)


def paraphrase_mining_with_combined_score(
    model,
    sentences,
    weight_semantic,
    weight_levenshtein,
    weight_jaro_winkler,
    k=100,
):
    embeddings = build_document_embeddings(model, list(sentences))
    results = rank_code_pairs(
        list(sentences),
        embeddings,
        weight_semantic,
        weight_levenshtein,
        weight_jaro_winkler,
    )
    limit = max(0, int(k or 0))
    if limit:
        return results[:limit]
    return results


def rank_code_pairs(
    codes,
    embeddings,
    weight_semantic,
    weight_levenshtein,
    weight_jaro_winkler,
    code_metric="none",
    code_metric_weight=0.0,
    code_language="java",
    code_metric_bidirectional=True,
    codebleu_component_weights=None,
    crystalbleu_context=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=500,
    vector_backend="transformer",
    multivector_bidirectional=True,
):
    results = []
    for i, j in combinations(range(len(codes)), 2):
        code_metric_score = 0.0
        if float(code_metric_weight) > 0.0 and (code_metric or "none").strip().lower() not in ("none", ""):
            code_metric_score = score_code_metric_pair(
                codes[i],
                codes[j],
                metric_name=code_metric,
                language=code_language,
                bidirectional=code_metric_bidirectional,
                component_weights=codebleu_component_weights,
                crystalbleu_context=crystalbleu_context,
                reference_index=i,
                prediction_index=j,
                crystalbleu_max_order=crystalbleu_max_order,
                crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
            )
        score = combined_similarity_from_embeddings(
            codes[i],
            codes[j],
            embeddings[i],
            embeddings[j],
            weight_semantic,
            weight_levenshtein,
            weight_jaro_winkler,
            code_metric_score=code_metric_score,
            code_metric_weight=code_metric_weight,
            vector_backend=vector_backend,
            multivector_bidirectional=multivector_bidirectional,
        )
        results.append((score, i, j))
    return sorted(results, reverse=True)


def get_sim_list(
    zipped_file,
    Ws,
    Wl,
    Wj,
    model_name,
    threshold,
    number_results,
    preprocess_mode="none",
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    code_metric="none",
    code_metric_weight=0.0,
    code_language="java",
    code_metric_bidirectional=True,
    codebleu_component_weights=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=500,
    vector_backend="transformer",
    static_vector_dim=256,
    static_vector_lowercase=True,
    multivector_bidirectional=True,
    device="auto",
):
    weight_semantic, weight_levenshtein, weight_jaro_winkler = validate_weights(Ws, Wl, Wj)
    code_metric_weight, component_weights = validate_code_metric_options(
        code_metric_weight,
        codebleu_component_weights,
    )
    vector_backend, static_vector_dim = validate_vector_options(vector_backend, static_vector_dim)
    file_names, raw_codes = extract_and_read_source(zipped_file)
    codes = [prepare_code(code, preprocess_mode=preprocess_mode) for code in raw_codes]
    model = None if vector_backend == "static_hash" else load_model(model_name, device=device)
    embeddings = build_document_embeddings(
        model,
        codes,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_aggregation=chunk_aggregation,
        vector_backend=vector_backend,
        static_vector_dim=static_vector_dim,
        static_vector_lowercase=static_vector_lowercase,
    )
    crystalbleu_context = None
    if (code_metric or "none").strip().lower() == "crystalbleu" and code_metric_weight > 0.0:
        crystalbleu_context = prepare_crystalbleu_context(codes, max_order=crystalbleu_max_order)
    code_pairs = rank_code_pairs(
        codes,
        embeddings,
        weight_semantic,
        weight_levenshtein,
        weight_jaro_winkler,
        code_metric=code_metric,
        code_metric_weight=code_metric_weight,
        code_language=code_language,
        code_metric_bidirectional=code_metric_bidirectional,
        codebleu_component_weights=component_weights,
        crystalbleu_context=crystalbleu_context,
        crystalbleu_max_order=crystalbleu_max_order,
        crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
    )

    pairs_results = [
        {
            "file_name_1": file_names[i],
            "file_name_2": file_names[j],
            "similarity_score": round(score, 2),
        }
        for score, i, j in code_pairs
        if score >= float(threshold)
    ]

    similarity_df = pd.DataFrame(pairs_results, columns=RESULT_COLUMNS)
    return similarity_df.head(max(1, int(number_results)))


def calculate_similarity(
    code1,
    code2,
    Ws,
    Wl,
    Wj,
    model_name,
    preprocess_mode="none",
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    code_metric="none",
    code_metric_weight=0.0,
    code_language="java",
    code_metric_bidirectional=True,
    codebleu_component_weights=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=500,
    vector_backend="transformer",
    static_vector_dim=256,
    static_vector_lowercase=True,
    multivector_bidirectional=True,
    device="auto",
):
    weight_semantic, weight_levenshtein, weight_jaro_winkler = validate_weights(Ws, Wl, Wj)
    code_metric_weight, component_weights = validate_code_metric_options(
        code_metric_weight,
        codebleu_component_weights,
    )
    vector_backend, static_vector_dim = validate_vector_options(vector_backend, static_vector_dim)
    prepared_code1 = prepare_code(code1, preprocess_mode=preprocess_mode)
    prepared_code2 = prepare_code(code2, preprocess_mode=preprocess_mode)
    model = None if vector_backend == "static_hash" else load_model(model_name, device=device)
    embeddings = build_document_embeddings(
        model,
        [prepared_code1, prepared_code2],
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_aggregation=chunk_aggregation,
        vector_backend=vector_backend,
        static_vector_dim=static_vector_dim,
        static_vector_lowercase=static_vector_lowercase,
    )
    code_metric_score = 0.0
    if code_metric_weight > 0.0 and (code_metric or "none").strip().lower() not in ("none", ""):
        code_metric_score = score_code_metric_pair(
            prepared_code1,
            prepared_code2,
            metric_name=code_metric,
            language=code_language,
            bidirectional=code_metric_bidirectional,
            component_weights=component_weights,
            crystalbleu_max_order=crystalbleu_max_order,
            crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        )
    return combined_similarity_from_embeddings(
        prepared_code1,
        prepared_code2,
        embeddings[0],
        embeddings[1],
        weight_semantic,
        weight_levenshtein,
        weight_jaro_winkler,
        code_metric_score=code_metric_score,
        code_metric_weight=code_metric_weight,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
    )
