import re


_STRING_LITERAL_RE = re.compile(r"`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
_NUMBER_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|<STR>|<NUM>|[^\w\s]")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IMPORT_BLOCK_START_RE = re.compile(r"^\s*import\s*\(\s*$")
_IMPORT_LIKE_RE = re.compile(
    r"""
    ^\s*(?:
        import\s+.+ |
        from\s+\S+\s+import\s+.+ |
        pragma\s+.+ |
        package\s+\S+.* |
        \#\s*include\s+.+ |
        \#\s*import\s+.+ |
        (?:global\s+)?using\s+(?:namespace\s+)?(?:static\s+)?[\w.]+(?:\s*=\s*[\w.]+)?\s*;? |
        library\s*\(.+\)\s*;? |
        source\s*\(.+\)\s*;? |
        use\s+[\w\\:,*{}\s]+;? |
        local\s+\w+\s*=\s*require\s*\(.+\)\s*;? |
        extern\s+crate\s+\S+.* |
        require(?:_once|_relative)?\s*(?:\(?\s*.+) |
        include(?:_once)?\s*(?:\(?\s*.+) |
        (?:const|let|var)\s+.+?=\s*require\s*\(.+\)\s*;?
    )\s*$
    """,
    re.VERBOSE,
)
_HASH_DIRECTIVES = (
    "#include",
    "#define",
    "#if",
    "#ifdef",
    "#ifndef",
    "#elif",
    "#else",
    "#endif",
    "#pragma",
    "#undef",
    "#line",
    "#error",
    "#warning",
    "#region",
    "#endregion",
    "#nullable",
)
_KEYWORDS = {
    "__halt_compiler",
    "abstract",
    "any",
    "and",
    "as",
    "asserts",
    "async",
    "auto",
    "await",
    "base",
    "bigint",
    "begin",
    "boolean",
    "bool",
    "break",
    "case",
    "catch",
    "chan",
    "char",
    "class",
    "clone",
    "const",
    "constructor",
    "continue",
    "contract",
    "crate",
    "crossinline",
    "data",
    "debugger",
    "defer",
    "def",
    "default",
    "deferred",
    "delegate",
    "delete",
    "declare",
    "deinit",
    "do",
    "double",
    "dynamic",
    "echo",
    "elsif",
    "elif",
    "else",
    "elseif",
    "end",
    "ensure",
    "enum",
    "event",
    "except",
    "export",
    "extends",
    "extension",
    "extern",
    "external",
    "fallthrough",
    "factory",
    "false",
    "field",
    "file",
    "final",
    "finally",
    "float",
    "fn",
    "for",
    "foreach",
    "forsome",
    "from",
    "func",
    "function",
    "fun",
    "get",
    "global",
    "go",
    "goto",
    "guard",
    "hide",
    "if",
    "immutable",
    "impl",
    "implicit",
    "import",
    "in",
    "include",
    "include_once",
    "infer",
    "instanceof",
    "interface",
    "int",
    "internal",
    "indexed",
    "indirect",
    "infix",
    "init",
    "inout",
    "is",
    "isset",
    "keyof",
    "late",
    "lateinit",
    "lambda",
    "lazy",
    "let",
    "library",
    "lock",
    "long",
    "loop",
    "macro_rules",
    "map",
    "match",
    "memory",
    "mixin",
    "mod",
    "modifier",
    "module",
    "mut",
    "namespace",
    "never",
    "new",
    "nameof",
    "next",
    "nil",
    "noinline",
    "none",
    "not",
    "number",
    "null",
    "object",
    "on",
    "open",
    "operator",
    "or",
    "out",
    "override",
    "package",
    "pass",
    "part",
    "payable",
    "private",
    "pragma",
    "precedencegroup",
    "protected",
    "public",
    "pub",
    "raise",
    "range",
    "readonly",
    "redo",
    "ref",
    "reified",
    "repeat",
    "required",
    "require",
    "require_once",
    "require_relative",
    "rethrow",
    "rethrows",
    "return",
    "returns",
    "revert",
    "rescue",
    "retry",
    "sealed",
    "self",
    "set",
    "show",
    "select",
    "short",
    "signed",
    "some",
    "static",
    "storage",
    "string",
    "struct",
    "subscript",
    "super",
    "suspend",
    "switch",
    "sync",
    "tailrec",
    "this",
    "throw",
    "throws",
    "trait",
    "true",
    "try",
    "type",
    "typealias",
    "typedef",
    "uint",
    "uint256",
    "template",
    "then",
    "typename",
    "unless",
    "unknown",
    "undefined",
    "use",
    "using",
    "val",
    "unsafe",
    "until",
    "var",
    "virtual",
    "void",
    "volatile",
    "when",
    "where",
    "while",
    "willset",
    "with",
    "yield",
    "view",
    "weak",
    "symbol",
    "satisfies",
    "unique",
}


_SUPPORTED_PREPROCESS_LANGUAGES = (
    "java",
    "python",
    "c",
    "cpp",
    "go",
    "javascript",
    "typescript",
    "kotlin",
    "scala",
    "swift",
    "solidity",
    "dart",
    "php",
    "ruby",
    "rust",
    "csharp",
    "lua",
    "julia",
    "r",
    "objc",
)
_PREPROCESS_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "c#": "csharp",
    "cs": "csharp",
    "js": "javascript",
    "objective-c": "objc",
    "objectivec": "objc",
    "py": "python",
    "ts": "typescript",
}
_SLASH_LINE_COMMENT_LANGUAGES = {
    "c",
    "cpp",
    "csharp",
    "dart",
    "go",
    "java",
    "javascript",
    "kotlin",
    "objc",
    "php",
    "rust",
    "scala",
    "solidity",
    "swift",
    "typescript",
}
_LUA_LINE_COMMENT_LANGUAGES = {"lua"}
_SLASH_BLOCK_COMMENT_LANGUAGES = _SLASH_LINE_COMMENT_LANGUAGES
_LUA_BLOCK_COMMENT_LANGUAGES = {"lua"}
_GENERIC_LINE_COMMENT_MARKERS = ("//", "#")
_GENERIC_BLOCK_COMMENT_MARKERS = (("/*", "*/"), ("--[[", "]]"))


def available_preprocess_modes():
    return ("none", "normalize", "basic", "advanced")


def available_preprocess_languages():
    return _SUPPORTED_PREPROCESS_LANGUAGES


def normalize_preprocess_language(language):
    key = (language or "").strip().lower().replace("_", "-")
    return _PREPROCESS_LANGUAGE_ALIASES.get(key, key)


def normalize_newlines(text):
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def trim_trailing_whitespace(text):
    lines = [line.rstrip() for line in normalize_newlines(text).split("\n")]
    return "\n".join(lines)


def drop_blank_lines(text):
    lines = [line for line in normalize_newlines(text).split("\n") if line.strip()]
    return "\n".join(lines)


def _block_comment_markers(language=None):
    normalized_language = normalize_preprocess_language(language)
    if not normalized_language:
        return _GENERIC_BLOCK_COMMENT_MARKERS

    markers = []
    if normalized_language in _SLASH_BLOCK_COMMENT_LANGUAGES:
        markers.append(("/*", "*/"))
    if normalized_language in _LUA_BLOCK_COMMENT_LANGUAGES:
        markers.append(("--[[", "]]"))
    if normalized_language not in _SUPPORTED_PREPROCESS_LANGUAGES:
        markers.extend(_GENERIC_BLOCK_COMMENT_MARKERS)
    return tuple(markers)


def _line_comment_markers(language=None):
    normalized_language = normalize_preprocess_language(language)
    if not normalized_language:
        return _GENERIC_LINE_COMMENT_MARKERS

    markers = [] if normalized_language in _LUA_LINE_COMMENT_LANGUAGES else ["#"]
    if normalized_language in _SLASH_LINE_COMMENT_LANGUAGES:
        markers.append("//")
    if normalized_language in _LUA_LINE_COMMENT_LANGUAGES:
        markers.append("--")
    if normalized_language not in _SUPPORTED_PREPROCESS_LANGUAGES:
        for marker in _GENERIC_LINE_COMMENT_MARKERS:
            if marker not in markers:
                markers.append(marker)
    return tuple(markers)


def _strip_block_comments_with_markers(text, markers):
    value = normalize_newlines(text)
    if not markers:
        return value

    cleaned = []
    index = 0
    quote = None
    escaped = False
    block_end = None

    while index < len(value):
        if block_end:
            if value.startswith(block_end, index):
                cleaned.append(" ")
                index += len(block_end)
                block_end = None
                continue
            if value[index] == "\n":
                cleaned.append("\n")
            index += 1
            continue

        char = value[index]
        if quote:
            cleaned.append(char)
            if escaped:
                escaped = False
            elif char == "\\" and quote != "`":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ("'", '"', "`"):
            quote = char
            cleaned.append(char)
            index += 1
            continue

        matched = False
        for start_marker, end_marker in markers:
            if value.startswith(start_marker, index):
                cleaned.append(" ")
                index += len(start_marker)
                block_end = end_marker
                matched = True
                break
        if matched:
            continue

        cleaned.append(char)
        index += 1

    return "".join(cleaned)


def strip_block_comments(text, language=None):
    return _strip_block_comments_with_markers(text, _block_comment_markers(language))


def _strip_line_comment_with_markers(line, markers):
    if not markers:
        return line
    if "#" in markers and line.lstrip().startswith(_HASH_DIRECTIVES):
        return line
    if "#" in markers and line.lstrip().startswith("#"):
        return ""

    index = 0
    quote = None
    escaped = False

    while index < len(line):
        char = line[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote != "`":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ("'", '"', "`"):
            quote = char
            index += 1
            continue

        for marker in markers:
            if line.startswith(marker, index):
                return line[:index].rstrip()
        index += 1

    return line


def strip_line_comments(text, language=None):
    markers = _line_comment_markers(language)
    cleaned_lines = []
    for line in normalize_newlines(text).split("\n"):
        cleaned_lines.append(_strip_line_comment_with_markers(line, markers).rstrip())
    return "\n".join(cleaned_lines)


def strip_import_like_lines(text):
    cleaned_lines = []
    in_import_block = False
    for line in normalize_newlines(text).split("\n"):
        stripped = line.strip()
        if in_import_block:
            if stripped == ")":
                in_import_block = False
            continue
        if _IMPORT_BLOCK_START_RE.match(line):
            in_import_block = True
            continue
        if _IMPORT_LIKE_RE.match(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


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


def preprocess_code(text, mode="none", language=None):
    selected_mode = (mode or "none").strip().lower()
    value = trim_trailing_whitespace(text)

    if selected_mode == "none":
        return value.strip()

    if selected_mode == "normalize":
        return drop_blank_lines(value).strip()

    if selected_mode == "basic":
        value = strip_block_comments(value, language=language)
        value = strip_line_comments(value, language=language)
        value = drop_blank_lines(value)
        return collapse_whitespace(value)

    if selected_mode == "advanced":
        value = strip_block_comments(value, language=language)
        value = strip_line_comments(value, language=language)
        value = strip_import_like_lines(value)
        value = drop_blank_lines(value)
        value = normalize_literal_placeholders(value)
        value = canonicalize_identifiers(value)
        return collapse_whitespace(value)

    raise ValueError(f"Unsupported preprocess mode: {mode}")
