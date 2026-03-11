_BACKEND_ALIASES = {
    "auto": "auto",
    "transformer": "sentence_transformers",
    "sentence-transformers": "sentence_transformers",
    "sentence_transformer": "sentence_transformers",
    "sentence_transformers": "sentence_transformers",
    "sbert": "sentence_transformers",
    "model2vec": "model2vec",
    "static": "model2vec",
    "static_vector": "model2vec",
    "pylate": "pylate",
    "multivector": "pylate",
    "late_interaction": "pylate",
    "colbert": "pylate",
    "static_hash": "static_hash",
}
_PUBLIC_VECTOR_BACKENDS = ("auto", "sentence_transformers", "model2vec", "pylate")
_DEPRECATED_VECTOR_BACKENDS = ("static_hash",)
_HF_MODEL_INFO_CACHE = {}


def available_vector_backends(include_deprecated=False):
    if include_deprecated:
        return _PUBLIC_VECTOR_BACKENDS + _DEPRECATED_VECTOR_BACKENDS
    return _PUBLIC_VECTOR_BACKENDS


def normalize_vector_backend_name(name):
    key = str(name or "auto").strip().lower()
    normalized = _BACKEND_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(available_vector_backends())
        raise ValueError(f"Unsupported vector backend: {name}. Supported backends: {supported}")
    return normalized


def load_hf_model_info(model_name):
    if not model_name:
        return None
    model_key = str(model_name).strip()
    if model_key in _HF_MODEL_INFO_CACHE:
        return _HF_MODEL_INFO_CACHE[model_key]
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - sentence-transformers usually installs this
        return None

    try:
        model_info = HfApi().model_info(model_key)
    except Exception:  # pragma: no cover - network and auth are environment-specific
        return None
    _HF_MODEL_INFO_CACHE[model_key] = model_info
    return model_info


def _library_name_from_info(model_info):
    if model_info is None:
        return ""
    return str(getattr(model_info, "library_name", "") or "").strip().lower()


def _tags_from_info(model_info):
    if model_info is None:
        return ()
    tags = getattr(model_info, "tags", None) or ()
    return tuple(str(tag).strip().lower() for tag in tags if str(tag).strip())


def infer_model_capabilities(model_name, model_info=None):
    library_name = _library_name_from_info(model_info)
    tags = _tags_from_info(model_info)
    model_name_key = str(model_name or "").strip().lower()
    has_static_tag = "static-embeddings" in tags
    has_pylate_tag = "pylate" in tags
    has_colbert_tag = "colbert" in tags

    supports_multivector = (
        library_name in ("pylate",)
        or has_pylate_tag
        or has_colbert_tag
        or "late-interaction" in tags
        or "pylate" in model_name_key
        or "colbert" in model_name_key
    )
    supports_static = (
        library_name in ("model2vec",)
        or "model2vec" in tags
        or has_static_tag
        or "model2vec" in model_name_key
        or "/m2v" in model_name_key
        or model_name_key.startswith("m2v-")
    )

    if library_name in ("pylate",):
        preferred_backend = "pylate"
    elif library_name in ("model2vec",):
        preferred_backend = "model2vec"
    elif library_name in ("sentence-transformers", "sentence_transformers"):
        preferred_backend = "sentence_transformers"
    elif has_pylate_tag or has_colbert_tag or "late-interaction" in tags:
        preferred_backend = "pylate"
    elif any(tag in ("sentence-transformers", "feature-extraction", "sentence-similarity") for tag in tags):
        preferred_backend = "sentence_transformers"
    elif "model2vec" in tags:
        preferred_backend = "model2vec"
    elif "pylate" in model_name_key or "colbert" in model_name_key:
        preferred_backend = "pylate"
    elif "model2vec" in model_name_key or "/m2v" in model_name_key or model_name_key.startswith("m2v-"):
        preferred_backend = "model2vec"
    else:
        preferred_backend = "sentence_transformers"

    return {
        "preferred_backend": preferred_backend,
        "supports_static": bool(supports_static),
        "supports_multivector": bool(supports_multivector),
    }


def infer_model_backend(model_name, model_info=None):
    return infer_model_capabilities(model_name, model_info=model_info)["preferred_backend"]


def resolve_vector_backend(requested_backend, model_name=None, model_info=None):
    backend = normalize_vector_backend_name(requested_backend)
    if backend != "auto":
        return backend
    return infer_model_backend(model_name, model_info=model_info)


def backend_is_multivector(vector_backend):
    return normalize_vector_backend_name(vector_backend) == "pylate"
