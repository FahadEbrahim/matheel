"""Input coverage diagnostics and comparisons using existing chunked scorers.

No token counts are guessed from bytes or lines. Token offsets describe input
coverage, not attribution of a final similarity score to individual source spans.
"""

from ._experiments import (
    content_id, experiment_metadata, experiment_runs, score_experiment_pairs, scoring_options,
    text_hash, write_experiment_artifacts,
)
from .chunking import chunk_text
from .feature_weights import resolve_feature_weights
from .model_routing import load_hf_model_info, normalize_vector_backend_name, resolve_vector_backend
from .similarity import load_backend_model, normalize_chunk_aggregation, prepare_code
from .vectors import tokenize_for_static_vectors


def _positive_limit(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if numeric == value and 0 < numeric < 1_000_000 else None


def _token_limit(model, tokenizer, requested, backend, is_query):
    if isinstance(requested, bool):
        raise ValueError("max_token_length must be a positive integer, not a boolean.")
    if backend == "static_hash":
        return None
    limits = []
    if requested is not None:
        limit = _positive_limit(requested)
        if limit is None:
            raise ValueError("max_token_length must be a positive integer below tokenizer sentinels.")
        limits.append(limit)
    role_limit = "query_length" if is_query else "document_length"
    for obj, names in (
        (model, (role_limit, "max_seq_length", "max_length")),
        (tokenizer, ("model_max_length", "max_seq_length")),
        (getattr(model, "config", None), ("max_position_embeddings", "n_positions")),
    ):
        for name in names:
            limit = _positive_limit(getattr(obj, name, None))
            if limit is not None:
                limits.append(limit)
    truncation = getattr(tokenizer, "truncation", None)
    if isinstance(truncation, dict):
        limit = _positive_limit(truncation.get("max_length"))
        if limit is not None:
            limits.append(limit)
    return min(limits) if limits else None


def _count_encoding(tokenizer, text, special):
    if getattr(tokenizer, "truncation", None):
        raise ValueError("Tokenizer already has truncation enabled; full token count is unavailable.")
    try:
        encoded = tokenizer.encode(text, add_special_tokens=special, truncation=False)
    except TypeError:
        encoded = tokenizer.encode(text, add_special_tokens=special)
    return len(encoded.ids if hasattr(encoded, "ids") else encoded)


def _token_coverage(text, tokenizer, limit, prefix):
    combined = prefix + text
    if getattr(tokenizer, "truncation", None):
        # Rust tokenizers may already truncate encode() and offer no untruncated
        # counting call. Do not mutate the caller's tokenizer or block scoring.
        return {"token_measurement": "unavailable", "input_tokens": None,
                "retained_input_tokens": None, "special_tokens": None,
                "source_tokens": None, "retained_source_tokens": None,
                "truncated": None, "source_spans": None,
                "tokenization_note": "Tokenizer has truncation configured; untruncated counts are unavailable."}
    # Fast Hugging Face offsets let us measure the actual truncation operation,
    # including retained special tokens, left truncation, and prompt boundaries.
    try:
        common = {"return_offsets_mapping": True, "return_special_tokens_mask": True,
                  "add_special_tokens": True, "padding": False}
        full = tokenizer(combined, truncation=False, **common)
        total = len(full["input_ids"])
        special_count = sum(full["special_tokens_mask"])
        if limit is not None and limit < special_count:
            raise ValueError("Configured limit cannot fit the tokenizer's required special tokens.")
        kept = (tokenizer(combined, truncation=True, max_length=limit, **common)
                if limit is not None and total > limit else full)
        spans = [(max(0, start - len(prefix)), end - len(prefix))
                 for start, end in kept["offset_mapping"] if end > len(prefix)]
        all_source = sum(end > len(prefix) for start, end in full["offset_mapping"])
        visible_source = sum(end > len(prefix) for start, end in kept["offset_mapping"])
        return {
            "token_measurement": "tokenizer_offsets", "input_tokens": total,
            "retained_input_tokens": len(kept["input_ids"]), "special_tokens": special_count,
            "source_tokens": all_source, "retained_source_tokens": visible_source,
            "truncated": len(kept["input_ids"]) < total,
            "source_spans": spans, "tokenization_note": None,
        }
    except (TypeError, NotImplementedError, KeyError):
        total = _count_encoding(tokenizer, combined, True)
        content_count = _count_encoding(tokenizer, combined, False)
        return {
            "token_measurement": "count_only", "input_tokens": total,
            "retained_input_tokens": None, "special_tokens": total - content_count,
            "source_tokens": None, "retained_source_tokens": None,
            "truncated": total > limit if limit is not None else None,
            "source_spans": [(0, len(text))] if limit is None or total <= limit else None,
            "tokenization_note": "No truncation offsets: retained source bytes/tokens are unknown.",
        }


def _chunk_spans(source, chunks):
    spans, cursor, ambiguous = [], 0, False
    for chunk in chunks:
        start = source.find(chunk, cursor)
        if start < 0:
            spans.append(None)
            ambiguous = True
            continue
        if source.find(chunk, start + 1) >= 0:
            ambiguous = True
        spans.append((start, start + len(chunk)))
        cursor = start + 1  # Permit source overlap while retaining deterministic order.
    return spans, ambiguous


def _coverage(source, spans):
    if any(span is None for span in spans):
        return {"utf8_bytes": None, "byte_fraction": None, "lines": None}
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    byte_count = sum(len(source[start:end].encode("utf-8")) for start, end in merged)
    line_ranges, offset = [], 0
    for line in source.splitlines(keepends=True):
        line_ranges.append((offset, offset + len(line)))
        offset += len(line)
    line_count = sum(any(start < line_end and end > line_start for start, end in merged)
                     for line_start, line_end in line_ranges)
    total = len(source.encode("utf-8"))
    return {"utf8_bytes": byte_count, "byte_fraction": byte_count / total if total else 1.0,
            "lines": line_count}


def preflight_long_file(text, *, options=None, model=None, tokenizer=None, prefix=None, is_query=False):
    """Inspect scorer inputs without loading models or contacting a model hub.

    Supply the actual loaded backend model/tokenizer for transformer truncation
    measurements. Without one, semantic token coverage remains unknown while
    UTF-8 byte/line and chunk-selection diagnostics remain available offline.
    ``prefix`` must match the encoder's actual prompt, when manually supplied.
    """
    options = dict(options or {})
    prepared = prepare_code(text, options.get("preprocess_mode", "none"),
                            options.get("code_language", "python"))
    backend = normalize_vector_backend_name(options.get("vector_backend", "static_hash"))
    method = options.get("chunking_method", "none")
    aggregation = normalize_chunk_aggregation(options.get("chunk_aggregation", "mean"))
    maximum = options.get("max_chunks", 0)
    if isinstance(maximum, bool) or int(maximum) != maximum or maximum < 0:
        raise ValueError("max_chunks must be a nonnegative integer.")
    chunks = chunk_text(
        prepared, method=method, chunk_size=options.get("chunk_size", 200),
        chunk_overlap=options.get("chunk_overlap", 0), max_chunks=0,
        chunk_language=options.get("chunk_language", "text"),
        chunker_options=options.get("chunker_options"),
    )
    selected = chunks[:maximum] if maximum else chunks
    spans, ambiguous = _chunk_spans(prepared, chunks)
    selected_spans = spans[:len(selected)]
    tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    limit = _token_limit(model, tokenizer, options.get("max_token_length"), backend, is_query)
    prompt_name = getattr(model, "default_prompt_name", None)
    if prefix is None:
        prefix = (getattr(model, "prompts", None) or {}).get(prompt_name, "")
        prefix_origin = "model.default_prompt_name" if prompt_name else "no default prompt inferred"
    else:
        prefix_origin = "explicit caller-supplied encoder prefix"
    notes = ["Coverage concerns encoder inputs, not semantic correctness or score attribution."]
    if backend == "auto":
        notes.append("Backend auto-routing is unresolved in offline preflight; supply its actual model.")
    if ambiguous:
        notes.append("Repeated or rewritten chunks make substring alignment an estimate.")
    if backend == "static_hash" and options.get("max_token_length") is not None:
        notes.append("static_hash does not truncate at max_token_length; the requested cap is ignored.")
    if backend != "static_hash" and tokenizer is None:
        notes.append("No tokenizer supplied: bytes/lines are not estimates of transformer token counts.")
    if backend == "multivector":
        notes.append("Verify query/document-specific prompts; multi-vector scoring ignores chunk_aggregation.")
    if aggregation == "first" and backend != "multivector":
        notes.append("All selected chunks are encoded, but first aggregation uses only the first chunk.")
    if model is not None:
        notes.append("Tokenizer offsets assume no additional backend-specific text normalization.")
    chunk_rows, visible_spans = [], []
    effective_count = 1 if aggregation == "first" and backend != "multivector" else len(selected)
    for index, chunk in enumerate(selected):
        if backend == "static_hash":
            tokens = len(tokenize_for_static_vectors(chunk))
            measured = {"token_measurement": "matheel_static_hash", "input_tokens": tokens,
                        "retained_input_tokens": tokens, "source_tokens": tokens,
                        "retained_source_tokens": tokens, "special_tokens": 0,
                        "truncated": False, "source_spans": [(0, len(chunk))],
                        "tokenization_note": None}
        elif tokenizer is not None:
            measured = _token_coverage(chunk, tokenizer, limit, prefix)
        else:
            measured = {"token_measurement": "unavailable", "input_tokens": None,
                        "retained_input_tokens": None, "source_tokens": None,
                        "retained_source_tokens": None, "special_tokens": None,
                        "truncated": None, "source_spans": None, "tokenization_note": None}
        model_spans = measured.pop("source_spans")
        source_span = selected_spans[index]
        if index < effective_count:
            if source_span is None or model_spans is None:
                visible_spans.append(None)
            else:
                visible_spans.extend((source_span[0] + start, source_span[0] + end)
                                     for start, end in model_spans)
        chunk_rows.append({"chunk_index": index, "sha256": text_hash(chunk),
                           "utf8_bytes": len(chunk.encode("utf-8")), "lines": len(chunk.splitlines()),
                           "source_span": source_span, "used_by_aggregation": index < effective_count,
                           **measured})
    return {
        "source_sha256": text_hash(text), "prepared_sha256": text_hash(prepared),
        "source_utf8_bytes": len(text.encode("utf-8")), "source_lines": len(text.splitlines()),
        "prepared_utf8_bytes": len(prepared.encode("utf-8")),
        "prepared_lines": len(prepared.splitlines()), "vector_backend": backend,
        "model_name": options.get("model_name"),
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "effective_max_input_tokens": limit, "requested_max_token_length": options.get("max_token_length"),
        "encoder_role": "query" if is_query else "document", "prefix": prefix,
        "prefix_origin": prefix_origin, "chunking_method": method,
        "chunk_aggregation": aggregation, "generated_chunks": len(chunks),
        "selected_chunks": len(selected), "omitted_chunks": len(chunks) - len(selected),
        "effective_chunks": effective_count, "alignment_ambiguous": ambiguous,
        "selected_chunk_coverage": _coverage(prepared, selected_spans),
        "effective_model_input_coverage": _coverage(prepared, visible_spans),
        "truncated_chunks": sum(row["truncated"] is True for row in chunk_rows),
        "unknown_truncation_chunks": sum(row["truncated"] is None for row in chunk_rows),
        "chunks": chunk_rows, "notes": notes,
    }


def default_long_file_runs():
    base = {"feature_weights": {"semantic": 1.0}, "vector_backend": "static_hash"}
    chunks = {"chunking_method": "chonkie_token", "chunk_size": 128,
              "chunk_overlap": 16, "chunker_options": {"tokenizer": "character"}}
    return [
        {"run_name": "raw_static_hash", "options": base},
        *[{"run_name": f"chunked_{aggregation}",
           "options": {**base, **chunks, "chunk_aggregation": aggregation}}
          for aggregation in ("mean", "max", "first")],
        {"run_name": "chunked_mean_cap_2", "options": {**base, **chunks, "max_chunks": 2}},
    ]


def compare_long_files(left, right, runs=None, *, inspect_models=True, output_dir=None):
    """Compare declared existing scorers and report each run's actual input plan.

    Semantic model inspection may load the explicitly requested backend model;
    preflight_long_file itself never loads anything. Set inspect_models=False
    for byte/line diagnostics without a second model load during scoring.
    """
    runs = experiment_runs(default_long_file_runs() if runs is None else runs)
    inputs = {name: {"sha256": text_hash(text), "text": text} for name, text in
              (("left", left), ("right", right))}
    report = {**experiment_metadata("long_files", runs, inputs), "inputs": inputs,
              "preflight": [], "scores": [],
              "interpretation": "Declared scoring configurations; no change to frozen benchmark protocols."}
    for run in runs:
        options, error, score = scoring_options(run), None, None
        preflight = {}
        try:
            if "algorithm" in options or "algorithm_path" in options:
                raise ValueError("Long-file coverage requires declared built-in scorer options.")
            semantic = resolve_feature_weights(
                options.get("feature_weights"), code_metric_weight=options.get("code_metric_weight", 0),
            ).get("semantic", 0) > 0
            backend = options.get("vector_backend", "auto")
            if semantic and inspect_models and backend == "auto":
                backend = resolve_vector_backend(
                    backend, model_name=options["model_name"],
                    model_info=load_hf_model_info(options["model_name"]),
                )
            inspected = {**options, "vector_backend": backend}
            model = None
            if semantic and inspect_models and backend != "static_hash":
                model = load_backend_model(
                    options["model_name"], vector_backend=backend, device=options.get("device", "cpu"),
                    similarity_function=options.get("similarity_function", "cosine"),
                    pooling_method=options.get("pooling_method", "mean"),
                    max_token_length=options.get("max_token_length"),
                )
            for name, text in (("left", left), ("right", right)):
                diagnostic = preflight_long_file(text, options=inspected, model=model)
                diagnostic["semantic_scoring_active"] = semantic
                if not semantic:
                    diagnostic["notes"].append(
                        "Lexical/code-only scoring uses full prepared text; these vector controls do not affect it.",
                    )
                preflight[name] = diagnostic
            del model
            score = score_experiment_pairs([("comparison", left, right)], run)["comparison"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        report["preflight"].append({"run_name": run["run_name"], "files": preflight, "error": error})
        report["scores"].append({"run_name": run["run_name"], "run_id": content_id(run),
                                 "similarity_score": score, "status": "error" if error else "ok",
                                 "error": error})
    if output_dir is not None:
        return report, write_experiment_artifacts(report, output_dir)
    return report, {}
