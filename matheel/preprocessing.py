import re


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_INLINE_HASH_COMMENT_RE = re.compile(r"\s#.*$")
_CPP_DIRECTIVES = (
    "#include",
    "#define",
    "#if",
    "#ifdef",
    "#ifndef",
    "#elif",
    "#else",
    "#endif",
    "#pragma",
)


def available_preprocess_modes():
    return ("none", "normalize", "basic", "synsem_basic")


def normalize_newlines(text):
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def trim_trailing_whitespace(text):
    lines = [line.rstrip() for line in normalize_newlines(text).split("\n")]
    return "\n".join(lines)


def drop_blank_lines(text):
    lines = [line for line in normalize_newlines(text).split("\n") if line.strip()]
    return "\n".join(lines)


def strip_block_comments(text):
    return _BLOCK_COMMENT_RE.sub(" ", normalize_newlines(text))


def _strip_hash_comment(line):
    stripped = line.lstrip()
    if stripped.startswith(_CPP_DIRECTIVES):
        return line
    if stripped.startswith("#"):
        return ""
    if _INLINE_HASH_COMMENT_RE.search(line):
        return _INLINE_HASH_COMMENT_RE.sub("", line).rstrip()
    return line


def strip_line_comments(text):
    cleaned_lines = []
    for line in normalize_newlines(text).split("\n"):
        current = line
        if "//" in current:
            current = current.split("//", 1)[0].rstrip()
        current = _strip_hash_comment(current)
        cleaned_lines.append(current)
    return "\n".join(cleaned_lines)


def collapse_whitespace(text):
    parts = [piece for piece in re.split(r"\s+", text) if piece]
    return " ".join(parts)


def preprocess_code(text, mode="none"):
    selected_mode = (mode or "none").strip().lower()
    value = trim_trailing_whitespace(text)

    if selected_mode in ("none", "raw"):
        return value.strip()

    if selected_mode in ("normalize", "basic_normalize"):
        return drop_blank_lines(value).strip()

    if selected_mode in ("basic", "synsem_basic"):
        value = strip_block_comments(value)
        value = strip_line_comments(value)
        value = drop_blank_lines(value)
        return collapse_whitespace(value)

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def preprocess_codes(texts, mode="none"):
    return [preprocess_code(text, mode=mode) for text in texts]
