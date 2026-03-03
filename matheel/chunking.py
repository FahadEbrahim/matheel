import inspect


_BASE_METHODS = ("none", "lines", "tokens", "characters")
_CHONKIE_METHODS = (
    "code",
    "codechunker",
    "chonkie_code",
    "chonkie_token",
    "chonkie_sentence",
    "chonkie_recursive",
    "chonkie_fast",
)
_CHONKIE_CLASS_NAMES = {
    "code": "CodeChunker",
    "codechunker": "CodeChunker",
    "chonkie_code": "CodeChunker",
    "chonkie_token": "TokenChunker",
    "chonkie_sentence": "SentenceChunker",
    "chonkie_recursive": "RecursiveChunker",
    "chonkie_fast": "FastChunker",
}
_CHONKIE_FALLBACK_METHODS = {
    "code": "lines",
    "codechunker": "lines",
    "chonkie_code": "lines",
    "chonkie_token": "tokens",
    "chonkie_sentence": "lines",
    "chonkie_recursive": "characters",
    "chonkie_fast": "lines",
}
_CHONKIE_PARAMETER_NAMES = {
    "code": ("chunk_size", "language", "include_nodes", "tokenizer"),
    "codechunker": ("chunk_size", "language", "include_nodes", "tokenizer"),
    "chonkie_code": ("chunk_size", "language", "include_nodes", "tokenizer"),
    "chonkie_token": ("chunk_size", "chunk_overlap", "tokenizer"),
    "chonkie_sentence": (
        "chunk_size",
        "chunk_overlap",
        "min_sentences_per_chunk",
        "min_characters_per_sentence",
        "approximate",
        "delim",
        "include_delim",
        "tokenizer",
    ),
    "chonkie_recursive": ("chunk_size", "min_characters_per_chunk", "tokenizer"),
    "chonkie_fast": ("chunk_size", "delimiters", "pattern", "prefix", "consecutive", "forward_fallback"),
}


def available_chunking_methods():
    return _BASE_METHODS + _CHONKIE_METHODS


def chunker_parameter_names(method):
    selected_method = (method or "").strip().lower()
    if selected_method in ("none", "document", "lines", "tokens", "characters"):
        return ()
    return _CHONKIE_PARAMETER_NAMES.get(selected_method, ())


def available_chunk_aggregations():
    return ("mean", "max", "first")


def _normalize_chunk_settings(chunk_size, chunk_overlap):
    size = int(chunk_size or 0)
    overlap = int(chunk_overlap or 0)
    if size <= 0:
        size = 1
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = size - 1
    return size, overlap


def _window_chunks(items, chunk_size, chunk_overlap, separator):
    if not items:
        return []

    size, overlap = _normalize_chunk_settings(chunk_size, chunk_overlap)
    step = max(1, size - overlap)
    chunks = []
    start = 0

    while start < len(items):
        current = items[start : start + size]
        if not current:
            break
        value = separator.join(current).strip()
        if value:
            chunks.append(value)
        start += step

    return chunks


def chunk_by_lines(text, chunk_size=20, chunk_overlap=0):
    lines = (text or "").splitlines()
    if not lines:
        return [""]
    return _window_chunks(lines, chunk_size, chunk_overlap, "\n")


def chunk_by_tokens(text, chunk_size=200, chunk_overlap=0):
    tokens = (text or "").split()
    if not tokens:
        return [""]
    return _window_chunks(tokens, chunk_size, chunk_overlap, " ")


def chunk_by_characters(text, chunk_size=1000, chunk_overlap=0):
    value = text or ""
    if not value:
        return [""]

    size, overlap = _normalize_chunk_settings(chunk_size, chunk_overlap)
    step = max(1, size - overlap)
    chunks = []
    start = 0

    while start < len(value):
        current = value[start : start + size]
        if not current:
            break
        chunks.append(current)
        start += step

    return chunks


def parse_chunker_options(chunker_options):
    if chunker_options is None:
        return {}
    if isinstance(chunker_options, dict):
        return dict(chunker_options)
    if isinstance(chunker_options, str):
        parts = [part.strip() for part in chunker_options.split(",") if part.strip()]
        return parse_chunker_options(parts)

    parsed = {}
    for item in chunker_options:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError(f"Chunker options must look like name=value. Got: {item}")
            key, value = item.split("=", 1)
            parsed[key.strip()] = _coerce_chunker_option_value(value.strip())
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            parsed[str(item[0]).strip()] = _coerce_chunker_option_value(item[1])
            continue
        raise ValueError(f"Unsupported chunker option entry: {item!r}")
    return parsed


def _coerce_chunker_option_value(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_chonkie_class(class_name):
    try:
        import chonkie
    except ImportError:  # pragma: no cover - optional dependency
        return None

    direct = getattr(chonkie, class_name, None)
    if direct is not None:
        return direct

    for attr_name in ("chunkers", "chunker"):
        namespace = getattr(chonkie, attr_name, None)
        if namespace is None:
            continue
        candidate = getattr(namespace, class_name, None)
        if candidate is not None:
            return candidate
    return None


def _filter_callable_kwargs(callable_obj, values):
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(values)
    return {
        key: value
        for key, value in values.items()
        if key in parameters
    }


def _normalize_chunk_output_items(items):
    chunks = []
    for item in items:
        if isinstance(item, str):
            value = item
        else:
            value = getattr(item, "text", None)
            if value is None:
                value = getattr(item, "content", None)
            if value is None:
                value = getattr(item, "chunk", None)
            if value is None:
                value = str(item)
        value = str(value or "").strip()
        if value:
            chunks.append(value)
    return chunks or [""]


def _call_chunker(chunker, text):
    if hasattr(chunker, "chunk"):
        return chunker.chunk(text)
    if hasattr(chunker, "split_text"):
        return chunker.split_text(text)
    if callable(chunker):
        return chunker(text)
    raise ValueError("Unsupported Chonkie chunker instance.")


def _build_chonkie_chunker(method, chunk_size, chunk_overlap, chunk_language="text", chunker_options=None):
    class_name = _CHONKIE_CLASS_NAMES.get(method)
    if class_name is None:
        return None
    chunker_class = _load_chonkie_class(class_name)
    if chunker_class is None:
        return None

    options = {
        "chunk_size": int(chunk_size or 0),
        "chunk_overlap": int(chunk_overlap or 0),
        "language": chunk_language or "text",
    }
    options.update(parse_chunker_options(chunker_options))
    filtered = _filter_callable_kwargs(chunker_class, options)
    try:
        return chunker_class(**filtered)
    except Exception:
        return None


def _chunk_with_chonkie(text, method, chunk_size, chunk_overlap, max_chunks=0, chunk_language="text", chunker_options=None):
    chunker = _build_chonkie_chunker(
        method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_language=chunk_language,
        chunker_options=chunker_options,
    )
    if chunker is None:
        fallback_method = _CHONKIE_FALLBACK_METHODS.get(method, "tokens")
        return chunk_text(
            text,
            method=fallback_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
        )

    chunks = _normalize_chunk_output_items(_call_chunker(chunker, text or ""))
    max_chunk_count = int(max_chunks or 0)
    if max_chunk_count > 0:
        return chunks[:max_chunk_count]
    return chunks


def chunk_text(
    text,
    method="none",
    chunk_size=200,
    chunk_overlap=0,
    max_chunks=0,
    chunk_language="text",
    chunker_options=None,
):
    selected_method = (method or "none").strip().lower()

    if selected_method in ("none", "document"):
        chunks = [text or ""]
    elif selected_method == "lines":
        chunks = chunk_by_lines(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif selected_method == "tokens":
        chunks = chunk_by_tokens(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif selected_method == "characters":
        chunks = chunk_by_characters(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif selected_method in _CHONKIE_METHODS:
        return _chunk_with_chonkie(
            text,
            method=selected_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            chunk_language=chunk_language,
            chunker_options=chunker_options,
        )
    else:
        raise ValueError(f"Unsupported chunking method: {method}")

    max_chunk_count = int(max_chunks or 0)
    if max_chunk_count > 0:
        return chunks[:max_chunk_count]
    return chunks
