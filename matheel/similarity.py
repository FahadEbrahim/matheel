import hashlib
import os
import zipfile
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .code_metrics import (
    parse_component_weights,
    prepare_codebertscore_context,
    prepare_crystalbleu_context,
    prepare_ruby_context,
    prepare_tsed_context,
    score_code_metric_pair,
    tokenize_for_code_metrics,
)
from .feature_weights import combine_weighted_scores, resolve_feature_weights
from .model_routing import (
    backend_is_multivector,
    load_hf_model_info,
    resolve_vector_backend,
)
from .preprocessing import preprocess_code
from .vectors import (
    build_multivector_embeddings,
    build_static_hash_vectors,
    build_chunked_single_vectors,
    configure_sentence_transformer_pooling,
    detect_model_max_token_length,
    encode_single_vectors,
    load_vector_model,
    multivector_similarity,
    normalize_pooling_method_name,
    resolve_max_token_length,
    normalize_similarity_function_name,
    single_vector_similarity,
)


DEFAULT_MODEL_NAME = "huggingface/CodeBERTa-small-v1"
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


def load_model(
    model_name,
    device="auto",
    similarity_function="cosine",
    pooling_method="mean",
    max_token_length=None,
):
    return load_vector_model(
        model_name or DEFAULT_MODEL_NAME,
        device=normalize_device(device),
        vector_backend="sentence_transformers",
        similarity_function=similarity_function,
        pooling_method=pooling_method,
        max_token_length=max_token_length,
    )


def load_backend_model(
    model_name,
    vector_backend="auto",
    device="auto",
    similarity_function="cosine",
    pooling_method="mean",
    max_token_length=None,
):
    normalized_device = normalize_device(device)
    selected_similarity = normalize_similarity_function_name(similarity_function)
    selected_pooling = normalize_pooling_method_name(pooling_method)
    try:
        return load_vector_model(
            model_name or DEFAULT_MODEL_NAME,
            vector_backend=vector_backend,
            device=normalized_device,
            similarity_function=selected_similarity,
            pooling_method=selected_pooling,
            max_token_length=max_token_length,
        )
    except ImportError:
        if vector_backend == "model2vec":
            return None
        if vector_backend == "pylate":
            return load_model(
                model_name or DEFAULT_MODEL_NAME,
                device=normalized_device,
                similarity_function=selected_similarity,
                pooling_method=selected_pooling,
                max_token_length=max_token_length,
            )
        raise


def inspect_model_settings(
    model_name,
    vector_backend="auto",
    device="auto",
    similarity_function="cosine",
    pooling_method="mean",
    max_token_length=None,
):
    model_key = model_name or DEFAULT_MODEL_NAME
    requested_backend = (vector_backend or "auto").strip() or "auto"
    normalized_device = normalize_device(device)
    selected_similarity = normalize_similarity_function_name(similarity_function)
    selected_pooling = normalize_pooling_method_name(pooling_method)

    model_info = None
    if requested_backend.lower() == "auto":
        model_info = load_hf_model_info(model_key)
    resolved_backend = resolve_vector_backend(
        requested_backend,
        model_name=model_key,
        model_info=model_info,
    )

    detected_max_token_length = detect_model_max_token_length(
        model_name=model_key,
    )
    configured_max_token_length = resolve_max_token_length(
        max_token_length,
        detected_max_token_length=detected_max_token_length,
    )

    return {
        "model_name": model_key,
        "requested_vector_backend": requested_backend,
        "resolved_vector_backend": resolved_backend,
        "runtime_device": normalized_device,
        "similarity_function": selected_similarity,
        "pooling_method": selected_pooling,
        "detected_max_token_length": int(detected_max_token_length),
        "configured_max_token_length": (
            int(configured_max_token_length)
            if configured_max_token_length is not None
            else None
        ),
        "supports_custom_max_token_length": resolved_backend in (
            "sentence_transformers",
            "model2vec",
            "pylate",
        ),
    }


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
        file_names.sort()
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


def validate_code_metric_options(code_metric_weight, codebleu_component_weights):
    return (
        _validate_weight("code_metric_weight", code_metric_weight),
        parse_component_weights(codebleu_component_weights),
    )


def validate_edit_distance_options(levenshtein_weights=None, jaro_winkler_prefix_weight=0.1):
    if levenshtein_weights in (None, ""):
        parsed_levenshtein_weights = (1, 1, 1)
    elif isinstance(levenshtein_weights, str):
        values = [item.strip() for item in levenshtein_weights.split(",") if item.strip()]
        if len(values) != 3:
            raise ValueError("levenshtein_weights must contain exactly 3 values: insert, delete, substitute.")
        parsed_levenshtein_weights = tuple(max(1, int(float(value))) for value in values)
    else:
        values = list(levenshtein_weights)
        if len(values) != 3:
            raise ValueError("levenshtein_weights must contain exactly 3 values: insert, delete, substitute.")
        parsed_levenshtein_weights = tuple(max(1, int(float(value))) for value in values)

    prefix_weight = float(jaro_winkler_prefix_weight)
    if prefix_weight < 0 or prefix_weight > 0.25:
        raise ValueError("jaro_winkler_prefix_weight must be between 0.0 and 0.25.")

    return parsed_levenshtein_weights, prefix_weight


def validate_lexical_baseline_options(
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
):
    parsed_winnowing_kgram = int(float(winnowing_kgram or 0))
    parsed_winnowing_window = int(float(winnowing_window or 0))
    parsed_gst_min_match_length = int(float(gst_min_match_length or 0))

    if parsed_winnowing_kgram < 1:
        raise ValueError("winnowing_kgram must be at least 1.")
    if parsed_winnowing_window < 1:
        raise ValueError("winnowing_window must be at least 1.")
    if parsed_gst_min_match_length < 1:
        raise ValueError("gst_min_match_length must be at least 1.")

    return (
        parsed_winnowing_kgram,
        parsed_winnowing_window,
        parsed_gst_min_match_length,
    )


def validate_vector_options(vector_backend, static_vector_dim, model_name=None, model_info=None):
    backend = resolve_vector_backend(
        vector_backend,
        model_name=model_name or DEFAULT_MODEL_NAME,
        model_info=model_info,
    )
    return backend, max(8, int(static_vector_dim or 0))


def semantic_similarity(
    embedding1,
    embedding2,
    vector_backend="sentence_transformers",
    multivector_bidirectional=False,
    similarity_function="cosine",
):
    if backend_is_multivector(vector_backend):
        return multivector_similarity(
            embedding1,
            embedding2,
            bidirectional=multivector_bidirectional,
            vector_backend=vector_backend,
        )
    return single_vector_similarity(
        embedding1,
        embedding2,
        similarity_function=similarity_function,
    )


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


def _should_use_chunking(chunking_method):
    return (chunking_method or "none").strip().lower() != "none"


def _stable_token_hash(tokens):
    payload = "\x1f".join(tokens).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _tokenize_for_lexical_matching(code):
    return tokenize_for_code_metrics(code)


def _winnowing_fingerprints(tokens, kgram_size=5, window_size=4):
    if not tokens:
        return set()

    effective_kgram_size = min(max(1, int(kgram_size)), len(tokens))
    hashes = [
        _stable_token_hash(tokens[index : index + effective_kgram_size])
        for index in range(len(tokens) - effective_kgram_size + 1)
    ]
    if not hashes:
        return set()

    effective_window_size = min(max(1, int(window_size)), len(hashes))
    selected = {}
    for start in range(len(hashes) - effective_window_size + 1):
        current_window = hashes[start : start + effective_window_size]
        minimum_hash = min(current_window)
        relative_index = max(
            offset for offset, value in enumerate(current_window) if value == minimum_hash
        )
        absolute_index = start + relative_index
        selected[absolute_index] = minimum_hash
    if not selected:
        selected[0] = min(hashes)
    return set(selected.values())


def winnowing_similarity_from_tokens(tokens1, tokens2, kgram_size=5, window_size=4):
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    fingerprints1 = _winnowing_fingerprints(
        tokens1,
        kgram_size=kgram_size,
        window_size=window_size,
    )
    fingerprints2 = _winnowing_fingerprints(
        tokens2,
        kgram_size=kgram_size,
        window_size=window_size,
    )
    union = fingerprints1 | fingerprints2
    if not union:
        return 1.0
    return len(fingerprints1 & fingerprints2) / float(len(union))


def _ranges_overlap(start1, length1, start2, length2):
    end1 = start1 + length1
    end2 = start2 + length2
    return start1 < end2 and start2 < end1


def gst_similarity_from_tokens(tokens1, tokens2, min_match_length=5):
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    effective_min_match_length = min(
        max(1, int(min_match_length)),
        min(len(tokens1), len(tokens2)),
    )
    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    covered_tokens = 0
    total_tokens = float(len(tokens1) + len(tokens2))

    # Repeatedly extract the longest non-overlapping matching tiles, then score
    # the proportion of both token streams that the tiles cover.
    while True:
        best_length = effective_min_match_length - 1
        matches = []
        positions_by_token = {}
        for index, token in enumerate(tokens2):
            if not marked2[index]:
                positions_by_token.setdefault(token, []).append(index)

        for index1, token in enumerate(tokens1):
            if marked1[index1]:
                continue
            for index2 in positions_by_token.get(token, ()):
                if marked2[index2]:
                    continue

                length = 0
                while index1 + length < len(tokens1) and index2 + length < len(tokens2):
                    if marked1[index1 + length] or marked2[index2 + length]:
                        break
                    if tokens1[index1 + length] != tokens2[index2 + length]:
                        break
                    length += 1

                if length < effective_min_match_length:
                    continue
                if length > best_length:
                    best_length = length
                    matches = [(index1, index2, length)]
                elif length == best_length:
                    matches.append((index1, index2, length))

        if best_length < effective_min_match_length or not matches:
            return (2.0 * covered_tokens) / total_tokens

        accepted_matches = []
        for index1, index2, length in sorted(matches, key=lambda item: (-item[2], item[0], item[1])):
            if any(marked1[index1 + offset] or marked2[index2 + offset] for offset in range(length)):
                continue
            if any(
                _ranges_overlap(index1, length, existing1, existing_length)
                or _ranges_overlap(index2, length, existing2, existing_length)
                for existing1, existing2, existing_length in accepted_matches
            ):
                continue
            accepted_matches.append((index1, index2, length))

        if not accepted_matches:
            return (2.0 * covered_tokens) / total_tokens

        for index1, index2, length in accepted_matches:
            for offset in range(length):
                marked1[index1 + offset] = True
                marked2[index2 + offset] = True
            covered_tokens += length


def build_document_embeddings(
    model,
    codes,
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    chunk_language="text",
    chunker_options=None,
    vector_backend="sentence_transformers",
    static_vector_dim=256,
    static_vector_lowercase=True,
    pooling_method="mean",
):
    if not codes:
        return []

    if vector_backend == "static_hash":
        return build_static_hash_vectors(codes, dim=static_vector_dim, lowercase=static_vector_lowercase)
    if vector_backend == "model2vec" and model is None:
        return build_static_hash_vectors(codes, dim=static_vector_dim, lowercase=static_vector_lowercase)

    if backend_is_multivector(vector_backend):
        return build_multivector_embeddings(
            model,
            codes,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
        )

    if vector_backend == "sentence_transformers" and model is not None:
        model = configure_sentence_transformer_pooling(model, pooling_method=pooling_method)

    if not _should_use_chunking(chunking_method):
        return encode_single_vectors(
            model,
            codes,
            vector_backend=vector_backend,
            static_vector_dim=static_vector_dim,
            static_vector_lowercase=static_vector_lowercase,
        )

    chunk_vectors = build_chunked_single_vectors(
        model,
        codes,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_language=chunk_language,
        chunker_options=chunker_options,
    )
    return [
        aggregate_chunk_embeddings(item, chunk_aggregation=chunk_aggregation)
        for item in chunk_vectors
    ]


def build_feature_scores(
    code1,
    code2,
    embedding1,
    embedding2,
    code_metric_score=0.0,
    vector_backend="sentence_transformers",
    multivector_bidirectional=False,
    extra_feature_scores=None,
    similarity_function="cosine",
    levenshtein_weights=(1, 1, 1),
    jaro_winkler_prefix_weight=0.1,
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
    active_features=None,
):
    requested_features = None
    if active_features is not None:
        requested_features = {str(name) for name in active_features}

    use_all_features = requested_features is None
    needs_token_features = use_all_features or "winnowing" in requested_features or "gst" in requested_features
    lexical_tokens1 = None
    lexical_tokens2 = None
    if needs_token_features:
        lexical_tokens1 = _tokenize_for_lexical_matching(code1)
        lexical_tokens2 = _tokenize_for_lexical_matching(code2)

    feature_scores = {
        "semantic": (
            semantic_similarity(
                embedding1,
                embedding2,
                vector_backend=vector_backend,
                multivector_bidirectional=multivector_bidirectional,
                similarity_function=similarity_function,
            )
            if (use_all_features or "semantic" in requested_features)
            and embedding1 is not None
            and embedding2 is not None
            else 0.0
        ),
        "levenshtein": (
            Levenshtein.normalized_similarity(code1, code2, weights=levenshtein_weights)
            if use_all_features or "levenshtein" in requested_features
            else 0.0
        ),
        "jaro_winkler": (
            JaroWinkler.normalized_similarity(
                code1,
                code2,
                prefix_weight=jaro_winkler_prefix_weight,
            )
            if use_all_features or "jaro_winkler" in requested_features
            else 0.0
        ),
        "winnowing": (
            winnowing_similarity_from_tokens(
                lexical_tokens1,
                lexical_tokens2,
                kgram_size=winnowing_kgram,
                window_size=winnowing_window,
            )
            if use_all_features or "winnowing" in requested_features
            else 0.0
        ),
        "gst": (
            gst_similarity_from_tokens(
                lexical_tokens1,
                lexical_tokens2,
                min_match_length=gst_min_match_length,
            )
            if use_all_features or "gst" in requested_features
            else 0.0
        ),
        "code_metric": float(code_metric_score),
    }
    for name, value in (extra_feature_scores or {}).items():
        feature_scores[str(name)] = float(value)
    return feature_scores


def combined_similarity_from_embeddings(
    code1,
    code2,
    embedding1,
    embedding2,
    feature_weights,
    code_metric_score=0.0,
    vector_backend="sentence_transformers",
    multivector_bidirectional=False,
    extra_feature_scores=None,
    similarity_function="cosine",
    levenshtein_weights=(1, 1, 1),
    jaro_winkler_prefix_weight=0.1,
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
):
    active_features = {
        name for name, value in (feature_weights or {}).items() if float(value) > 0.0
    }
    feature_scores = build_feature_scores(
        code1,
        code2,
        embedding1,
        embedding2,
        code_metric_score=code_metric_score,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
        extra_feature_scores=extra_feature_scores,
        similarity_function=similarity_function,
        levenshtein_weights=levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
        active_features=active_features,
    )
    return combine_weighted_scores(feature_scores, feature_weights)


def rank_code_pairs(
    codes,
    embeddings,
    feature_weights,
    code_metric="none",
    code_language="java",
    code_metric_bidirectional=False,
    codebleu_component_weights=None,
    crystalbleu_context=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=50,
    ruby_context=None,
    ruby_max_order=4,
    ruby_epsilon=1e-12,
    ruby_mode="auto",
    ruby_tokenizer="tranx",
    ruby_denominator="max",
    ruby_graph_timeout_seconds=1.0,
    ruby_graph_use_edge_cost=True,
    ruby_graph_include_leaf_edges=True,
    ruby_tree_max_nodes=180,
    ruby_tree_max_depth=10,
    ruby_tree_max_children=8,
    tsed_context=None,
    tsed_delete_cost=1.0,
    tsed_insert_cost=1.0,
    tsed_rename_cost=1.0,
    tsed_max_nodes=180,
    tsed_max_depth=10,
    tsed_max_children=8,
    codebertscore_context=None,
    codebertscore_model="microsoft/codebert-base",
    codebertscore_num_layers=None,
    codebertscore_batch_size=16,
    codebertscore_max_length=0,
    codebertscore_device="auto",
    codebertscore_lang=None,
    codebertscore_idf=False,
    codebertscore_rescale_with_baseline=False,
    codebertscore_use_fast_tokenizer=False,
    codebertscore_nthreads=4,
    codebertscore_verbose=False,
    vector_backend="sentence_transformers",
    multivector_bidirectional=False,
    similarity_function="cosine",
    levenshtein_weights=(1, 1, 1),
    jaro_winkler_prefix_weight=0.1,
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
):
    results = []
    code_metric_name = (code_metric or "none").strip().lower()
    use_code_metric = code_metric_name not in ("none", "") and feature_weights.get("code_metric", 0.0) > 0.0

    for i, j in combinations(range(len(codes)), 2):
        code_metric_score = 0.0
        if use_code_metric:
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
                ruby_context=ruby_context,
                ruby_max_order=ruby_max_order,
                ruby_epsilon=ruby_epsilon,
                ruby_mode=ruby_mode,
                ruby_tokenizer=ruby_tokenizer,
                ruby_denominator=ruby_denominator,
                ruby_graph_timeout_seconds=ruby_graph_timeout_seconds,
                ruby_graph_use_edge_cost=ruby_graph_use_edge_cost,
                ruby_graph_include_leaf_edges=ruby_graph_include_leaf_edges,
                ruby_tree_max_nodes=ruby_tree_max_nodes,
                ruby_tree_max_depth=ruby_tree_max_depth,
                ruby_tree_max_children=ruby_tree_max_children,
                tsed_context=tsed_context,
                tsed_delete_cost=tsed_delete_cost,
                tsed_insert_cost=tsed_insert_cost,
                tsed_rename_cost=tsed_rename_cost,
                tsed_max_nodes=tsed_max_nodes,
                tsed_max_depth=tsed_max_depth,
                tsed_max_children=tsed_max_children,
                codebertscore_context=codebertscore_context,
                codebertscore_model=codebertscore_model,
                codebertscore_num_layers=codebertscore_num_layers,
                codebertscore_batch_size=codebertscore_batch_size,
                codebertscore_max_length=codebertscore_max_length,
                codebertscore_device=codebertscore_device,
                codebertscore_lang=codebertscore_lang,
                codebertscore_idf=codebertscore_idf,
                codebertscore_rescale_with_baseline=codebertscore_rescale_with_baseline,
                codebertscore_use_fast_tokenizer=codebertscore_use_fast_tokenizer,
                codebertscore_nthreads=codebertscore_nthreads,
                codebertscore_verbose=codebertscore_verbose,
            )
        score = combined_similarity_from_embeddings(
            codes[i],
            codes[j],
            embeddings[i],
            embeddings[j],
            feature_weights,
            code_metric_score=code_metric_score,
            vector_backend=vector_backend,
            multivector_bidirectional=multivector_bidirectional,
            similarity_function=similarity_function,
            levenshtein_weights=levenshtein_weights,
            jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
            winnowing_kgram=winnowing_kgram,
            winnowing_window=winnowing_window,
            gst_min_match_length=gst_min_match_length,
        )
        results.append((score, i, j))
    return sorted(results, reverse=True)


def get_sim_list(
    zipped_file,
    model_name=DEFAULT_MODEL_NAME,
    threshold=0.0,
    number_results=10,
    preprocess_mode="none",
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    chunk_language="text",
    chunker_options=None,
    code_metric="none",
    code_metric_weight=0.0,
    code_language="java",
    code_metric_bidirectional=False,
    codebleu_component_weights=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=50,
    ruby_max_order=4,
    ruby_epsilon=1e-12,
    ruby_mode="auto",
    ruby_tokenizer="tranx",
    ruby_denominator="max",
    ruby_graph_timeout_seconds=1.0,
    ruby_graph_use_edge_cost=True,
    ruby_graph_include_leaf_edges=True,
    ruby_tree_max_nodes=180,
    ruby_tree_max_depth=10,
    ruby_tree_max_children=8,
    tsed_delete_cost=1.0,
    tsed_insert_cost=1.0,
    tsed_rename_cost=1.0,
    tsed_max_nodes=180,
    tsed_max_depth=10,
    tsed_max_children=8,
    codebertscore_model="microsoft/codebert-base",
    codebertscore_num_layers=None,
    codebertscore_batch_size=16,
    codebertscore_max_length=0,
    codebertscore_device="auto",
    codebertscore_lang=None,
    codebertscore_idf=False,
    codebertscore_rescale_with_baseline=False,
    codebertscore_use_fast_tokenizer=False,
    codebertscore_nthreads=4,
    codebertscore_verbose=False,
    vector_backend="auto",
    similarity_function="cosine",
    static_vector_dim=256,
    static_vector_lowercase=True,
    pooling_method="mean",
    multivector_bidirectional=False,
    device="auto",
    feature_weights=None,
    max_token_length=None,
    levenshtein_weights=None,
    jaro_winkler_prefix_weight=0.1,
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
):
    code_metric_weight, component_weights = validate_code_metric_options(
        code_metric_weight,
        codebleu_component_weights,
    )
    levenshtein_weights, jaro_winkler_prefix_weight = validate_edit_distance_options(
        levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
    )
    winnowing_kgram, winnowing_window, gst_min_match_length = validate_lexical_baseline_options(
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
    )
    model_info = None
    resolved_feature_weights = resolve_feature_weights(
        feature_weights=feature_weights,
        code_metric_weight=code_metric_weight,
    )
    use_semantic = resolved_feature_weights.get("semantic", 0.0) > 0.0
    if use_semantic and (vector_backend or "auto").strip().lower() in ("", "auto"):
        model_info = load_hf_model_info(model_name or DEFAULT_MODEL_NAME)
    vector_backend, static_vector_dim = validate_vector_options(
        vector_backend,
        static_vector_dim,
        model_name=model_name,
        model_info=model_info,
    )
    selected_similarity = normalize_similarity_function_name(similarity_function)
    selected_pooling = normalize_pooling_method_name(pooling_method)
    file_names, raw_codes = extract_and_read_source(zipped_file)
    codes = [prepare_code(code, preprocess_mode=preprocess_mode) for code in raw_codes]
    if use_semantic:
        model = load_backend_model(
            model_name,
            vector_backend=vector_backend,
            device=device,
            similarity_function=selected_similarity,
            pooling_method=selected_pooling,
            max_token_length=max_token_length,
        )
        embeddings = build_document_embeddings(
            model,
            codes,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_aggregation=chunk_aggregation,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
            vector_backend=vector_backend,
            static_vector_dim=static_vector_dim,
            static_vector_lowercase=static_vector_lowercase,
            pooling_method=selected_pooling,
        )
    else:
        embeddings = [None] * len(codes)
    code_metric_name = (code_metric or "none").strip().lower()
    use_code_metric = code_metric_name not in ("none", "") and resolved_feature_weights.get("code_metric", 0.0) > 0.0
    crystalbleu_context = None
    ruby_context = None
    tsed_context = None
    codebertscore_context = None
    if use_code_metric and code_metric_name == "crystalbleu":
        crystalbleu_context = prepare_crystalbleu_context(codes, max_order=crystalbleu_max_order)
    if use_code_metric and code_metric_name == "ruby":
        ruby_context = prepare_ruby_context(
            codes,
            language=code_language,
            max_order=ruby_max_order,
            tokenizer=ruby_tokenizer,
            tree_max_nodes=ruby_tree_max_nodes,
            tree_max_depth=ruby_tree_max_depth,
            tree_max_children=ruby_tree_max_children,
            graph_include_leaf_edges=ruby_graph_include_leaf_edges,
        )
    if use_code_metric and code_metric_name == "tsed":
        tsed_context = prepare_tsed_context(
            codes,
            language=code_language,
            max_nodes=tsed_max_nodes,
            max_depth=tsed_max_depth,
            max_children=tsed_max_children,
        )
    if use_code_metric and code_metric_name == "codebertscore":
        codebertscore_context = prepare_codebertscore_context(codes)
    code_pairs = rank_code_pairs(
        codes,
        embeddings,
        resolved_feature_weights,
        code_metric=code_metric,
        code_language=code_language,
        code_metric_bidirectional=code_metric_bidirectional,
        codebleu_component_weights=component_weights,
        crystalbleu_context=crystalbleu_context,
        crystalbleu_max_order=crystalbleu_max_order,
        crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        ruby_context=ruby_context,
        ruby_max_order=ruby_max_order,
        ruby_epsilon=ruby_epsilon,
        ruby_mode=ruby_mode,
        ruby_tokenizer=ruby_tokenizer,
        ruby_denominator=ruby_denominator,
        ruby_graph_timeout_seconds=ruby_graph_timeout_seconds,
        ruby_graph_use_edge_cost=ruby_graph_use_edge_cost,
        ruby_graph_include_leaf_edges=ruby_graph_include_leaf_edges,
        ruby_tree_max_nodes=ruby_tree_max_nodes,
        ruby_tree_max_depth=ruby_tree_max_depth,
        ruby_tree_max_children=ruby_tree_max_children,
        tsed_context=tsed_context,
        tsed_delete_cost=tsed_delete_cost,
        tsed_insert_cost=tsed_insert_cost,
        tsed_rename_cost=tsed_rename_cost,
        tsed_max_nodes=tsed_max_nodes,
        tsed_max_depth=tsed_max_depth,
        tsed_max_children=tsed_max_children,
        codebertscore_context=codebertscore_context,
        codebertscore_model=codebertscore_model,
        codebertscore_num_layers=codebertscore_num_layers,
        codebertscore_batch_size=codebertscore_batch_size,
        codebertscore_max_length=codebertscore_max_length,
        codebertscore_device=codebertscore_device,
        codebertscore_lang=codebertscore_lang,
        codebertscore_idf=codebertscore_idf,
        codebertscore_rescale_with_baseline=codebertscore_rescale_with_baseline,
        codebertscore_use_fast_tokenizer=codebertscore_use_fast_tokenizer,
        codebertscore_nthreads=codebertscore_nthreads,
        codebertscore_verbose=codebertscore_verbose,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
        similarity_function=selected_similarity,
        levenshtein_weights=levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
    )

    pairs_results = [
        {
            "file_name_1": file_names[i],
            "file_name_2": file_names[j],
            "similarity_score": round(score, 4),
        }
        for score, i, j in code_pairs
        if score >= float(threshold)
    ]

    similarity_df = pd.DataFrame(pairs_results, columns=RESULT_COLUMNS)
    return similarity_df.head(max(1, int(number_results)))


def calculate_similarity(
    code1,
    code2,
    model_name=DEFAULT_MODEL_NAME,
    preprocess_mode="none",
    chunking_method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_aggregation="mean",
    chunk_language="text",
    chunker_options=None,
    code_metric="none",
    code_metric_weight=0.0,
    code_language="java",
    code_metric_bidirectional=False,
    codebleu_component_weights=None,
    crystalbleu_max_order=4,
    crystalbleu_trivial_ngram_count=50,
    ruby_max_order=4,
    ruby_epsilon=1e-12,
    ruby_mode="auto",
    ruby_tokenizer="tranx",
    ruby_denominator="max",
    ruby_graph_timeout_seconds=1.0,
    ruby_graph_use_edge_cost=True,
    ruby_graph_include_leaf_edges=True,
    ruby_tree_max_nodes=180,
    ruby_tree_max_depth=10,
    ruby_tree_max_children=8,
    tsed_delete_cost=1.0,
    tsed_insert_cost=1.0,
    tsed_rename_cost=1.0,
    tsed_max_nodes=180,
    tsed_max_depth=10,
    tsed_max_children=8,
    codebertscore_model="microsoft/codebert-base",
    codebertscore_num_layers=None,
    codebertscore_batch_size=16,
    codebertscore_max_length=0,
    codebertscore_device="auto",
    codebertscore_lang=None,
    codebertscore_idf=False,
    codebertscore_rescale_with_baseline=False,
    codebertscore_use_fast_tokenizer=False,
    codebertscore_nthreads=4,
    codebertscore_verbose=False,
    vector_backend="auto",
    similarity_function="cosine",
    static_vector_dim=256,
    static_vector_lowercase=True,
    pooling_method="mean",
    multivector_bidirectional=False,
    device="auto",
    feature_weights=None,
    max_token_length=None,
    levenshtein_weights=None,
    jaro_winkler_prefix_weight=0.1,
    winnowing_kgram=5,
    winnowing_window=4,
    gst_min_match_length=5,
):
    code_metric_weight, component_weights = validate_code_metric_options(
        code_metric_weight,
        codebleu_component_weights,
    )
    levenshtein_weights, jaro_winkler_prefix_weight = validate_edit_distance_options(
        levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
    )
    winnowing_kgram, winnowing_window, gst_min_match_length = validate_lexical_baseline_options(
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
    )
    model_info = None
    resolved_feature_weights = resolve_feature_weights(
        feature_weights=feature_weights,
        code_metric_weight=code_metric_weight,
    )
    use_semantic = resolved_feature_weights.get("semantic", 0.0) > 0.0
    if use_semantic and (vector_backend or "auto").strip().lower() in ("", "auto"):
        model_info = load_hf_model_info(model_name or DEFAULT_MODEL_NAME)
    vector_backend, static_vector_dim = validate_vector_options(
        vector_backend,
        static_vector_dim,
        model_name=model_name,
        model_info=model_info,
    )
    selected_similarity = normalize_similarity_function_name(similarity_function)
    selected_pooling = normalize_pooling_method_name(pooling_method)
    prepared_code1 = prepare_code(code1, preprocess_mode=preprocess_mode)
    prepared_code2 = prepare_code(code2, preprocess_mode=preprocess_mode)
    if use_semantic:
        model = load_backend_model(
            model_name,
            vector_backend=vector_backend,
            device=device,
            similarity_function=selected_similarity,
            pooling_method=selected_pooling,
            max_token_length=max_token_length,
        )
        embeddings = build_document_embeddings(
            model,
            [prepared_code1, prepared_code2],
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_aggregation=chunk_aggregation,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
            vector_backend=vector_backend,
            static_vector_dim=static_vector_dim,
            static_vector_lowercase=static_vector_lowercase,
            pooling_method=selected_pooling,
        )
    else:
        embeddings = [None, None]
    code_metric_score = 0.0
    if (code_metric or "none").strip().lower() not in ("none", "") and resolved_feature_weights.get("code_metric", 0.0) > 0.0:
        code_metric_score = score_code_metric_pair(
            prepared_code1,
            prepared_code2,
            metric_name=code_metric,
            language=code_language,
            bidirectional=code_metric_bidirectional,
            component_weights=component_weights,
            crystalbleu_max_order=crystalbleu_max_order,
            crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
            ruby_max_order=ruby_max_order,
            ruby_epsilon=ruby_epsilon,
            ruby_mode=ruby_mode,
            ruby_tokenizer=ruby_tokenizer,
            ruby_denominator=ruby_denominator,
            ruby_graph_timeout_seconds=ruby_graph_timeout_seconds,
            ruby_graph_use_edge_cost=ruby_graph_use_edge_cost,
            ruby_graph_include_leaf_edges=ruby_graph_include_leaf_edges,
            ruby_tree_max_nodes=ruby_tree_max_nodes,
            ruby_tree_max_depth=ruby_tree_max_depth,
            ruby_tree_max_children=ruby_tree_max_children,
            tsed_delete_cost=tsed_delete_cost,
            tsed_insert_cost=tsed_insert_cost,
            tsed_rename_cost=tsed_rename_cost,
            tsed_max_nodes=tsed_max_nodes,
            tsed_max_depth=tsed_max_depth,
            tsed_max_children=tsed_max_children,
            codebertscore_model=codebertscore_model,
            codebertscore_num_layers=codebertscore_num_layers,
            codebertscore_batch_size=codebertscore_batch_size,
            codebertscore_max_length=codebertscore_max_length,
            codebertscore_device=codebertscore_device,
            codebertscore_lang=codebertscore_lang,
            codebertscore_idf=codebertscore_idf,
            codebertscore_rescale_with_baseline=codebertscore_rescale_with_baseline,
            codebertscore_use_fast_tokenizer=codebertscore_use_fast_tokenizer,
            codebertscore_nthreads=codebertscore_nthreads,
            codebertscore_verbose=codebertscore_verbose,
        )
    return combined_similarity_from_embeddings(
        prepared_code1,
        prepared_code2,
        embeddings[0],
        embeddings[1],
        resolved_feature_weights,
        code_metric_score=code_metric_score,
        vector_backend=vector_backend,
        multivector_bidirectional=multivector_bidirectional,
        similarity_function=selected_similarity,
        levenshtein_weights=levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
    )
