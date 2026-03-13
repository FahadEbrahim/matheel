import re


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LUA_BLOCK_COMMENT_RE = re.compile(r"--\[\[.*?\]\]", re.DOTALL)
_INLINE_HASH_COMMENT_RE = re.compile(r"\s#.*$")
_INLINE_SLASH_COMMENT_RE = re.compile(r"//.*$")
_INLINE_LUA_COMMENT_RE = re.compile(r"--.*$")
_STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
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
    "sealed",
    "select",
    "short",
    "signed",
    "some",
    "static",
    "storage",
    "static",
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
    "super",
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
    "when",
    "where",
    "while",
    "willset",
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


def available_preprocess_modes():
    return ("none", "normalize", "basic", "advanced")


def available_preprocess_languages():
    return _SUPPORTED_PREPROCESS_LANGUAGES


def normalize_newlines(text):
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def trim_trailing_whitespace(text):
    lines = [line.rstrip() for line in normalize_newlines(text).split("\n")]
    return "\n".join(lines)


def drop_blank_lines(text):
    lines = [line for line in normalize_newlines(text).split("\n") if line.strip()]
    return "\n".join(lines)


def strip_block_comments(text):
    value = _BLOCK_COMMENT_RE.sub(" ", normalize_newlines(text))
    return _LUA_BLOCK_COMMENT_RE.sub(" ", value)


def _strip_hash_comment(line):
    stripped = line.lstrip()
    if stripped.startswith(_HASH_DIRECTIVES):
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
        current = _INLINE_LUA_COMMENT_RE.sub("", current).rstrip()
        current = _strip_hash_comment(current)
        cleaned_lines.append(current)
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
