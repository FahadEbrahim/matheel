def available_chunking_methods():
    return ("none", "lines", "tokens", "characters")


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
        text = separator.join(current).strip()
        if text:
            chunks.append(text)
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


def chunk_text(text, method="none", chunk_size=200, chunk_overlap=0, max_chunks=0):
    selected_method = (method or "none").strip().lower()

    if selected_method in ("none", "document"):
        chunks = [text or ""]
    elif selected_method == "lines":
        chunks = chunk_by_lines(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif selected_method == "tokens":
        chunks = chunk_by_tokens(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif selected_method == "characters":
        chunks = chunk_by_characters(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")

    max_chunk_count = int(max_chunks or 0)
    if max_chunk_count > 0:
        return chunks[:max_chunk_count]
    return chunks
