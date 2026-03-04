import re


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_INLINE_HASH_COMMENT_RE = re.compile(r"\s#.*$")
_INLINE_SLASH_COMMENT_RE = re.compile(r"//.*$")
_STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_NUMBER_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|<STR>|<NUM>|[^\w\s]")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IMPORT_LIKE_RE = re.compile(
    r"^\s*(?:import\s+.+|from\s+\S+\s+import\s+.+|package\s+\S+|#\s*include\s+.+|using\s+namespace\s+\S+)\s*;?\s*$"
)
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
_KEYWORDS = {
    "and",
    "as",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "do",
    "double",
    "elif",
    "else",
    "enum",
    "except",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "if",
    "import",
    "in",
    "include",
    "int",
    "interface",
    "lambda",
    "long",
    "namespace",
    "new",
    "none",
    "null",
    "package",
    "pass",
    "private",
    "protected",
    "public",
    "raise",
    "return",
    "short",
    "signed",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "typename",
    "using",
    "void",
    "volatile",
    "while",
    "with",
    "yield",
}


def available_preprocess_modes():
    return ("none", "normalize", "basic", "advanced")


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
        current = _INLINE_SLASH_COMMENT_RE.sub("", line).rstrip()
        current = _strip_hash_comment(current)
        cleaned_lines.append(current)
    return "\n".join(cleaned_lines)


def strip_import_like_lines(text):
    lines = [line for line in normalize_newlines(text).split("\n") if not _IMPORT_LIKE_RE.match(line)]
    return "\n".join(lines)


def collapse_whitespace(text):
    parts = [piece for piece in re.split(r"\s+", text or "") if piece]
    return " ".join(parts)


def normalize_literal_placeholders(text):
    value = _STRING_LITERAL_RE.sub(" <STR> ", text or "")
    return _NUMBER_LITERAL_RE.sub(" <NUM> ", value)


def canonicalize_identifiers(text):
    mapping = {}
    next_index = 1
    normalized = []
    for token in _TOKEN_RE.findall(text or ""):
        if token in ("<STR>", "<NUM>"):
            normalized.append(token)
            continue
        if _IDENTIFIER_RE.match(token):
            lowered = token.lower()
            if lowered in _KEYWORDS:
                normalized.append(lowered)
                continue
            if token not in mapping:
                mapping[token] = f"id{next_index}"
                next_index += 1
            normalized.append(mapping[token])
            continue
        normalized.append(token)
    return " ".join(normalized)


def preprocess_code(text, mode="none"):
    selected_mode = (mode or "none").strip().lower()
    value = trim_trailing_whitespace(text)

    if selected_mode == "none":
        return value.strip()

    if selected_mode == "normalize":
        return drop_blank_lines(value).strip()

    if selected_mode == "basic":
        value = strip_block_comments(value)
        value = strip_line_comments(value)
        value = drop_blank_lines(value)
        return collapse_whitespace(value)

    if selected_mode == "advanced":
        value = strip_block_comments(value)
        value = strip_line_comments(value)
        value = strip_import_like_lines(value)
        value = drop_blank_lines(value)
        value = normalize_literal_placeholders(value)
        value = canonicalize_identifiers(value)
        return collapse_whitespace(value)

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def preprocess_codes(texts, mode="none"):
    return [preprocess_code(text, mode=mode) for text in texts]
