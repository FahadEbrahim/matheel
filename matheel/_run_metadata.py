from time import perf_counter


_METADATA_DECIMALS = 4
INACTIVE_SEMANTIC_BACKEND = "inactive"


def active_feature_names(feature_weights):
    names = [
        str(name)
        for name, value in (feature_weights or {}).items()
        if float(value) > 0.0
    ]
    return tuple(sorted(names)) or ("none",)


def format_feature_set(feature_weights):
    return ",".join(active_feature_names(feature_weights))


def semantic_is_active(feature_weights):
    return float((feature_weights or {}).get("semantic", 0.0)) > 0.0


def semantic_vector_backend_metadata(feature_weights, vector_backend):
    if semantic_is_active(feature_weights):
        return vector_backend
    return INACTIVE_SEMANTIC_BACKEND


def elapsed_seconds_since(start_time):
    return elapsed_seconds_between(start_time, perf_counter())


def elapsed_seconds_between(start_time, end_time):
    return round(max(0.0, float(end_time) - float(start_time)), _METADATA_DECIMALS)


def attach_run_metadata(
    frame,
    *,
    elapsed_seconds,
    feature_weights,
    vector_backend,
    code_metric,
    chunking_method,
    lexical_tokenizer=None,
):
    frame.attrs["elapsed_seconds"] = round(float(elapsed_seconds), _METADATA_DECIMALS)
    frame.attrs["feature_set"] = format_feature_set(feature_weights)
    frame.attrs["vector_backend"] = semantic_vector_backend_metadata(feature_weights, vector_backend)
    frame.attrs["code_metric"] = code_metric
    frame.attrs["chunking_method"] = chunking_method
    if lexical_tokenizer is not None:
        frame.attrs["lexical_tokenizer"] = lexical_tokenizer
    return frame
