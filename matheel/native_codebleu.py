import math
import re
import tokenize
from collections import Counter
from io import StringIO


try:
    from tree_sitter import Parser
except ImportError:  # pragma: no cover - optional dependency
    Parser = None

try:
    from tree_sitter_language_pack import get_language as get_tree_sitter_language
except ImportError:  # pragma: no cover - optional dependency
    get_tree_sitter_language = None


AVAILABLE_LANGUAGES = (
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

_ASSIGNMENT_OPERATORS = {
    "=",
    "<-",
    ":=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
    ">>>=",
}
_SELF_REFERENTIAL_OPERATORS = _ASSIGNMENT_OPERATORS - {"=", "<-", ":="}
_STRING_LITERAL_NODE_TYPES = {
    "string_literal",
    "string",
    "character_literal",
    "raw_string_literal",
    "interpreted_string_literal",
}
_IDENTIFIER_NODE_TYPES = {
    "identifier",
    "simple_identifier",
    "name",
    "variable_name",
}
_DECLARATION_NODE_TYPES = {
    "default_parameter",
    "variable_declarator",
    "var_definition",
    "val_definition",
    "property_declaration",
    "local_variable_declaration",
    "variable_declaration_statement",
    "declaration",
    "init_declarator",
    "simple_parameter",
    "parameter",
    "let_declaration",
    "var_spec",
    "short_var_declaration",
    "keyword_parameter",
    "variable_declaration",
}
_ASSIGNMENT_NODE_TYPES = {
    "assignment",
    "assignment_expression",
    "assignment_statement",
    "augmented_assignment_expression",
    "operator_assignment",
    "binary_operator",
    "for_in_clause",
}
_INCREMENT_NODE_TYPES = {
    "update_expression",
    "postfix_unary_expression",
    "inc_statement",
}
_IF_NODE_TYPES = {
    "if_statement",
    "if_expression",
    "if",
    "elsif",
    "elif_clause",
    "else_clause",
    "else",
    "unless",
    "when",
}
_FOR_NODE_TYPES = {
    "for_statement",
    "enhanced_for_statement",
    "for_each_statement",
    "for_expression",
    "for",
}
_WHILE_NODE_TYPES = {
    "while_statement",
    "while_modifier",
    "until",
}
_UNIVERSAL_KEYWORDS = {
    "abstract",
    "add",
    "alias",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "base",
    "begin",
    "bool",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "contract",
    "crate",
    "def",
    "default",
    "defer",
    "delegate",
    "delete",
    "do",
    "double",
    "else",
    "elseif",
    "elsif",
    "end",
    "enum",
    "error",
    "event",
    "except",
    "export",
    "extends",
    "extern",
    "false",
    "final",
    "finally",
    "float",
    "fn",
    "for",
    "foreach",
    "from",
    "func",
    "function",
    "fun",
    "get",
    "global",
    "go",
    "goto",
    "guard",
    "if",
    "implements",
    "import",
    "in",
    "include",
    "include_once",
    "indirect",
    "infer",
    "init",
    "inline",
    "int",
    "Int",
    "interface",
    "internal",
    "is",
    "let",
    "library",
    "local",
    "long",
    "loop",
    "map",
    "match",
    "memory",
    "mod",
    "module",
    "mutable",
    "namespace",
    "native",
    "new",
    "nil",
    "None",
    "not",
    "null",
    "number",
    "object",
    "operator",
    "or",
    "out",
    "override",
    "package",
    "pass",
    "pragma",
    "private",
    "protected",
    "protocol",
    "public",
    "pub",
    "raise",
    "range",
    "readonly",
    "ref",
    "require",
    "require_once",
    "require_relative",
    "return",
    "returns",
    "sealed",
    "select",
    "self",
    "set",
    "short",
    "signed",
    "source",
    "static",
    "storage",
    "string",
    "String",
    "struct",
    "super",
    "switch",
    "template",
    "then",
    "this",
    "throw",
    "throws",
    "trait",
    "true",
    "True",
    "try",
    "type",
    "typealias",
    "typedef",
    "uint",
    "uint256",
    "union",
    "unsafe",
    "unsigned",
    "use",
    "using",
    "val",
    "var",
    "virtual",
    "void",
    "volatile",
    "when",
    "where",
    "while",
    "with",
    "yield",
}
_LANGUAGE_KEYWORDS = {
    "c": {
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
        "else", "enum", "extern", "float", "for", "goto", "if", "int", "long", "register",
        "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
        "union", "unsigned", "void", "volatile", "while",
    },
    "cpp": {
        "auto", "bool", "break", "case", "catch", "char", "class", "const", "continue",
        "default", "delete", "do", "double", "else", "enum", "false", "float", "for", "if",
        "int", "long", "namespace", "new", "operator", "private", "protected", "public",
        "return", "short", "signed", "static", "struct", "switch", "template", "this",
        "throw", "true", "try", "typedef", "typename", "using", "void", "while",
    },
    "java": {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
        "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
        "finally", "float", "for", "if", "implements", "import", "instanceof", "int",
        "interface", "long", "native", "new", "package", "private", "protected", "public",
        "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
        "throw", "throws", "transient", "try", "void", "volatile", "while",
    },
    "python": {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
        "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "match", "nonlocal", "not", "or",
        "pass", "raise", "return", "try", "while", "with", "yield",
    },
    "go": {
        "break", "case", "chan", "const", "continue", "default", "defer", "else",
        "fallthrough", "for", "func", "go", "goto", "if", "import", "interface", "map",
        "package", "range", "return", "select", "struct", "switch", "type", "var",
    },
    "javascript": {
        "await", "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "enum", "export", "extends", "finally", "for",
        "function", "if", "implements", "import", "in", "instanceof", "interface", "let",
        "new", "package", "private", "protected", "public", "return", "static", "super",
        "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield",
    },
    "typescript": {
        "abstract", "any", "as", "asserts", "async", "await", "bigint", "boolean", "break",
        "case", "catch", "class", "const", "constructor", "continue", "declare", "default",
        "delete", "do", "else", "enum", "export", "extends", "finally", "for", "function",
        "get", "if", "implements", "import", "in", "infer", "instanceof", "interface",
        "is", "keyof", "let", "module", "namespace", "never", "new", "number", "object",
        "override", "package", "private", "protected", "public", "readonly", "return",
        "satisfies", "set", "static", "string", "super", "switch", "symbol", "this",
        "throw", "try", "type", "typeof", "undefined", "unique", "unknown", "var", "void",
        "while", "yield",
    },
    "kotlin": {
        "as", "break", "class", "companion", "constructor", "continue", "data", "do", "else",
        "enum", "false", "finally", "for", "fun", "if", "import", "in", "interface", "is",
        "null", "object", "open", "operator", "override", "package", "private", "protected",
        "public", "return", "sealed", "super", "suspend", "this", "throw", "true", "try",
        "typealias", "val", "var", "when", "where", "while",
    },
    "scala": {
        "abstract", "case", "catch", "class", "def", "do", "else", "extends", "false",
        "final", "finally", "for", "forSome", "if", "implicit", "import", "lazy", "match",
        "new", "null", "object", "override", "package", "private", "protected", "return",
        "sealed", "super", "this", "throw", "trait", "true", "try", "type", "val", "var",
        "while", "with", "yield",
    },
    "swift": {
        "actor", "as", "associatedtype", "async", "await", "break", "case", "catch", "class",
        "continue", "default", "defer", "deinit", "do", "else", "enum", "extension",
        "fallthrough", "fileprivate", "final", "for", "func", "get", "guard", "if", "import",
        "in", "indirect", "infix", "init", "inout", "internal", "is", "let", "mutating",
        "nil", "nonmutating", "open", "operator", "override", "postfix", "prefix", "private",
        "protocol", "public", "repeat", "required", "rethrows", "return", "self", "set",
        "static", "struct", "subscript", "super", "switch", "throw", "throws", "try",
        "typealias", "var", "weak", "where", "while",
    },
    "solidity": {
        "abstract", "address", "anonymous", "as", "assembly", "bool", "break", "bytes",
        "calldata", "constant", "constructor", "continue", "contract", "delete", "do", "else",
        "emit", "enum", "error", "event", "external", "fallback", "for", "function", "if",
        "immutable", "import", "indexed", "interface", "internal", "is", "library", "mapping",
        "memory", "modifier", "new", "override", "payable", "pragma", "private", "public",
        "pure", "return", "returns", "revert", "storage", "struct", "try", "type", "uint",
        "uint256", "using", "view", "virtual", "while",
    },
    "dart": {
        "abstract", "as", "assert", "async", "await", "break", "case", "catch", "class",
        "const", "continue", "covariant", "default", "deferred", "do", "dynamic", "else",
        "enum", "export", "extends", "extension", "external", "factory", "final", "finally",
        "for", "get", "hide", "if", "implements", "import", "in", "interface", "is", "late",
        "library", "mixin", "new", "on", "operator", "part", "required", "rethrow", "return",
        "set", "show", "static", "super", "switch", "sync", "this", "throw", "try", "typedef",
        "var", "void", "while", "with", "yield",
    },
    "php": {
        "__halt_compiler", "abstract", "and", "array", "as", "break", "callable", "case",
        "catch", "class", "clone", "const", "continue", "declare", "default", "die", "do",
        "echo", "else", "elseif", "empty", "eval", "exit", "extends", "final", "for",
        "foreach", "function", "global", "goto", "if", "implements", "include",
        "include_once", "instanceof", "insteadof", "interface", "isset", "list", "namespace",
        "new", "or", "print", "private", "protected", "public", "require", "require_once",
        "return", "static", "switch", "throw", "trait", "try", "unset", "use", "var", "while",
        "xor",
    },
    "ruby": {
        "__ENCODING__", "__FILE__", "__LINE__", "BEGIN", "END", "alias", "and", "begin",
        "break", "case", "class", "def", "defined?", "do", "else", "elsif", "end", "ensure",
        "false", "for", "if", "in", "module", "next", "nil", "not", "or", "redo", "rescue",
        "retry", "return", "self", "super", "then", "true", "undef", "unless", "until",
        "when", "while", "yield",
    },
    "rust": {
        "as", "async", "await", "bool", "break", "char", "const", "continue", "crate", "dyn",
        "else", "enum", "extern", "f32", "f64", "false", "fn", "for", "i128", "i16", "i32",
        "i64", "i8", "if", "impl", "in", "isize", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "static", "str", "struct", "super", "trait",
        "true", "type", "u128", "u16", "u32", "u64", "u8", "unsafe", "use", "usize", "where",
        "while", "yield",
    },
    "csharp": {
        "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
        "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
        "enum", "event", "explicit", "extern", "finally", "fixed", "float", "for", "foreach",
        "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long",
        "namespace", "new", "object", "operator", "out", "override", "params", "private",
        "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
        "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "try",
        "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual", "void",
        "volatile", "while",
    },
    "lua": {
        "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto",
        "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
        "while",
    },
    "julia": {
        "abstract", "begin", "break", "catch", "const", "continue", "do", "else", "elseif",
        "end", "false", "finally", "for", "function", "global", "if", "import", "in", "let",
        "local", "macro", "module", "mutable", "quote", "return", "struct", "true", "try",
        "using", "where", "while",
    },
    "r": {
        "break", "else", "FALSE", "for", "function", "if", "in", "Inf", "NA", "NaN", "next",
        "NULL", "repeat", "return", "TRUE", "while",
    },
    "objc": {
        "break", "case", "catch", "char", "class", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "id", "if", "implementation",
        "import", "int", "interface", "long", "nil", "nonatomic", "property", "protocol",
        "readonly", "return", "selector", "self", "short", "signed", "static", "struct",
        "super", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    },
}


def native_runtime_available():
    return Parser is not None and get_tree_sitter_language is not None


def _make_parser(tree_sitter_language):
    parser = Parser()
    try:
        parser.language = tree_sitter_language
    except AttributeError:  # pragma: no cover - older bindings
        parser.set_language(tree_sitter_language)
    return parser


def _node_text(node, lines):
    return index_to_code_token((node.start_point, node.end_point), lines)


def _strip_line_comment(line, prefix):
    in_single = False
    in_double = False
    index = 0
    while index < len(line):
        current = line[index]
        if current == "\\" and (in_single or in_double):
            index += 2
            continue
        if current == "'" and not in_double:
            in_single = not in_single
        elif current == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double and line.startswith(prefix, index):
            return line[:index]
        index += 1
    return line


def _strip_prefixed_comments(source, prefixes):
    cleaned = []
    for line in source.split("\n"):
        current = line
        for prefix in prefixes:
            current = _strip_line_comment(current, prefix)
        if current.strip():
            cleaned.append(current)
    return "\n".join(cleaned)


def remove_comments_and_docstrings(source, lang):
    value = source or ""
    if lang == "python":
        io_obj = StringIO(value)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE and start_col > 0:
                    out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        return "\n".join(line for line in out.split("\n") if line.strip())
    if lang == "lua":
        value = re.sub(r"--\[\[.*?\]\]", " ", value, flags=re.DOTALL)
        return _strip_prefixed_comments(value, ("--",))
    if lang in {"ruby", "r", "julia", "php"}:
        return _strip_prefixed_comments(value, ("#",))
    value = re.sub(r"/\*.*?\*/", " ", value, flags=re.DOTALL)
    value = _strip_prefixed_comments(value, ("//",))
    if lang == "objc":
        value = _strip_prefixed_comments(value, ("#",))
    return value


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = ((left_pad_symbol,) * (n - 1)) + tuple(sequence)
    else:
        sequence = tuple(sequence)
    if pad_right:
        sequence = sequence + ((right_pad_symbol,) * (n - 1))
    return iter(sequence)


def ngrams(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    sequence = pad_sequence(sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)
    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def _closest_ref_length(references, hyp_len, weighted=False):
    if weighted:
        ref_lens = (len(reference[0]) for reference in references)
    else:
        ref_lens = (len(reference) for reference in references)
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))


def _brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - (closest_ref_len / hyp_len))


def _modified_recall(references, hypothesis, n, weighted=False):
    numerator = 0.0
    denominator = 0.0
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    for reference_entry in references:
        if weighted:
            reference = reference_entry[0]
            token_weights = reference_entry[1]
        else:
            reference = reference_entry
            token_weights = {}
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        clipped_counts = {ngram: min(count, counts[ngram]) for ngram, count in reference_counts.items()}
        if weighted and n == 1 and len(token_weights) == len(reference_counts):
            def weighted_sum(counter):
                total = 0.0
                for ngram, count in counter.items():
                    total += count * (token_weights.get(ngram[0], 1.0))
                return total

            numerator += weighted_sum(clipped_counts)
            denominator += max(1.0, weighted_sum(reference_counts))
        else:
            numerator += float(sum(clipped_counts.values()))
            denominator += max(1.0, float(sum(reference_counts.values())))
    return numerator, denominator


def _smooth_precisions(precisions, epsilon=0.1):
    return [((numer + epsilon), denom) if numer == 0 else (numer, denom) for numer, denom in precisions]


def _corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), weighted=False):
    p_numerators = Counter()
    p_denominators = Counter()
    hyp_lengths = 0
    ref_lengths = 0
    if len(list_of_references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match.")
    for references, hypothesis in zip(list_of_references, hypotheses):
        for index, _ in enumerate(weights, start=1):
            numerator, denominator = _modified_recall(references, hypothesis, index, weighted=weighted)
            p_numerators[index] += numerator
            p_denominators[index] += denominator
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += _closest_ref_length(references, hyp_len, weighted=weighted)
    if p_numerators[1] == 0:
        return 0.0
    precisions = [(p_numerators[index], p_denominators[index]) for index, _ in enumerate(weights, start=1)]
    precisions = _smooth_precisions(precisions)
    brevity_penalty = _brevity_penalty(ref_lengths, hyp_lengths)
    return float(
        brevity_penalty
        * math.exp(math.fsum(weight * math.log(numerator / denominator) for weight, (numerator, denominator) in zip(weights, precisions)))
    )


def _is_variable_token(node_type, token, language):
    value = token or ""
    if not value:
        return False
    if re.match(r"^\d+(?:\.\d+)?$", value):
        return True
    if node_type not in _IDENTIFIER_NODE_TYPES:
        return False
    if value in _LANGUAGE_KEYWORDS.get(language, set()) or value in _UNIVERSAL_KEYWORDS:
        return False
    if value.lower() in _UNIVERSAL_KEYWORDS:
        return False
    if re.match(r"^[$@%][A-Za-z_][A-Za-z0-9_]*$", value):
        return True
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value))


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type in _STRING_LITERAL_NODE_TYPES) and root_node.type != "comment":
        return [(root_node.start_point, root_node.end_point)]
    code_tokens = []
    for child in root_node.children:
        code_tokens += tree_to_token_index(child)
    return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        return code[start_point[0]][start_point[1] : end_point[1]]
    value = code[start_point[0]][start_point[1] :]
    for row in range(start_point[0] + 1, end_point[0]):
        value += code[row]
    value += code[end_point[0]][: end_point[1]]
    return value


def _collect_variable_indices(root_node, index_to_code, language):
    if (len(root_node.children) == 0 or root_node.type in _STRING_LITERAL_NODE_TYPES) and root_node.type != "comment":
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if _is_variable_token(root_node.type, code, language):
            return [index]
        return []
    indices = []
    for child in root_node.children:
        indices += _collect_variable_indices(child, index_to_code, language)
    return indices


def _child_by_any_field(node, names):
    for name in names:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    return None


def _merge_states(state_maps):
    merged = {}
    for state_map in state_maps:
        for key, values in state_map.items():
            merged.setdefault(key, [])
            merged[key].extend(values)
    for key in merged:
        merged[key] = sorted(set(merged[key]))
    return merged


def _dedupe_dfg(entries):
    deduped = {}
    for entry in entries:
        key = (entry[0], entry[1], entry[2])
        if key not in deduped:
            deduped[key] = [list(entry[3]), list(entry[4])]
            continue
        deduped[key][0] = list(set(deduped[key][0] + list(entry[3])))
        deduped[key][1] = sorted(set(deduped[key][1] + list(entry[4])))
    return [(key[0], key[1], key[2], value[0], value[1]) for key, value in sorted(deduped.items(), key=lambda item: item[0][1])]


def _extract_declaration_parts(node, index_to_code, language):
    declarator = _child_by_any_field(node, ("declarator",))
    if declarator is not None:
        name = _child_by_any_field(declarator, ("declarator", "name", "pattern"))
        value = _child_by_any_field(declarator, ("value", "default_value", "right", "rhs", "result"))
        if name is not None:
            return name, value
    name = _child_by_any_field(node, ("name", "pattern", "left", "lhs", "target", "item", "variable"))
    value = _child_by_any_field(node, ("value", "default_value", "right", "rhs", "result", "collection", "sequence"))
    named_children = list(getattr(node, "named_children", []))
    if name is None:
        for child in named_children:
            if _collect_variable_indices(child, index_to_code, language):
                name = child
                break
    if value is None:
        for child in reversed(named_children):
            if child is name:
                continue
            if child.end_byte <= getattr(name, "end_byte", -1):
                continue
            value = child
            break
    return name, value


def _assignment_operator(node, lines):
    operator = _child_by_any_field(node, ("operator",))
    if operator is not None:
        return _node_text(operator, lines).strip()
    for child in node.children:
        child_text = _node_text(child, lines).strip()
        if child_text in _ASSIGNMENT_OPERATORS:
            return child_text
    return None


def _extract_assignment_parts(node, index_to_code, language):
    left = _child_by_any_field(node, ("left", "lhs", "target", "pattern", "item", "variable"))
    right = _child_by_any_field(node, ("right", "rhs", "result", "value", "collection", "sequence"))
    named_children = list(getattr(node, "named_children", []))
    if left is None:
        for child in named_children:
            if _collect_variable_indices(child, index_to_code, language):
                left = child
                break
    if right is None:
        for child in reversed(named_children):
            if child is left:
                continue
            if child.end_byte <= getattr(left, "end_byte", -1):
                continue
            right = child
            break
    return left, right


def _build_dfg(root_node, index_to_code, lines, states, language):
    states = states.copy()
    if (len(root_node.children) == 0 or root_node.type in _STRING_LITERAL_NODE_TYPES) and root_node.type != "comment":
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if not _is_variable_token(root_node.type, code, language):
            return [], states
        if code in states:
            return [(code, idx, "comesFrom", [code], states[code].copy())], states
        states[code] = [idx]
        return [(code, idx, "comesFrom", [], [])], states

    if root_node.type in _IF_NODE_TYPES:
        dfg = []
        condition_states = states.copy()
        branch_states = []
        saw_branch = False
        for index, child in enumerate(root_node.children):
            field_name = None
            try:
                field_name = root_node.field_name_for_child(index)
            except Exception:  # pragma: no cover - tree-sitter compatibility
                field_name = None
            is_branch = field_name in {"consequence", "alternative", "body"} or "else" in child.type
            if is_branch:
                saw_branch = True
                temp, new_states = _build_dfg(child, index_to_code, lines, condition_states.copy(), language)
                dfg += temp
                branch_states.append(new_states)
            else:
                temp, condition_states = _build_dfg(child, index_to_code, lines, condition_states, language)
                dfg += temp
        if not saw_branch:
            return _dedupe_dfg(dfg), condition_states
        merged = _merge_states(branch_states + [condition_states])
        return _dedupe_dfg(dfg), merged

    if root_node.type in _FOR_NODE_TYPES or root_node.type in _WHILE_NODE_TYPES:
        dfg = []
        loop_states = states.copy()
        for _ in range(2):
            for child in root_node.children:
                temp, loop_states = _build_dfg(child, index_to_code, lines, loop_states, language)
                dfg += temp
        return _dedupe_dfg(dfg), loop_states

    if root_node.type in _DECLARATION_NODE_TYPES:
        name_node, value_node = _extract_declaration_parts(root_node, index_to_code, language)
        if name_node is not None:
            dfg = []
            if value_node is not None:
                temp, states = _build_dfg(value_node, index_to_code, lines, states, language)
                dfg += temp
            name_indices = _collect_variable_indices(name_node, index_to_code, language)
            value_indices = [] if value_node is None else _collect_variable_indices(value_node, index_to_code, language)
            if not name_indices:
                return [], states
            if not value_indices:
                for index in name_indices:
                    idx, code = index_to_code[index]
                    dfg.append((code, idx, "comesFrom", [], []))
                    states[code] = [idx]
                return _dedupe_dfg(dfg), states
            for index_left in name_indices:
                left_idx, left_code = index_to_code[index_left]
                dfg.append(
                    (
                        left_code,
                        left_idx,
                        "comesFrom",
                        [index_to_code[index_right][1] for index_right in value_indices],
                        [index_to_code[index_right][0] for index_right in value_indices],
                    )
                )
                states[left_code] = [left_idx]
            return _dedupe_dfg(dfg), states

    if root_node.type in _ASSIGNMENT_NODE_TYPES:
        operator = _assignment_operator(root_node, lines)
        if operator in _ASSIGNMENT_OPERATORS or root_node.type == "for_in_clause":
            left_node, right_node = _extract_assignment_parts(root_node, index_to_code, language)
            if left_node is not None:
                dfg = []
                if right_node is not None:
                    temp, states = _build_dfg(right_node, index_to_code, lines, states, language)
                    dfg += temp
                left_indices = _collect_variable_indices(left_node, index_to_code, language)
                right_indices = [] if right_node is None else _collect_variable_indices(right_node, index_to_code, language)
                if operator in _SELF_REFERENTIAL_OPERATORS:
                    right_indices = list(right_indices) + list(left_indices)
                for index_left in left_indices:
                    left_idx, left_code = index_to_code[index_left]
                    dfg.append(
                        (
                            left_code,
                            left_idx,
                            "computedFrom",
                            [index_to_code[index_right][1] for index_right in right_indices],
                            [index_to_code[index_right][0] for index_right in right_indices],
                        )
                    )
                    states[left_code] = [left_idx]
                return _dedupe_dfg(dfg), states

    if root_node.type in _INCREMENT_NODE_TYPES:
        indices = _collect_variable_indices(root_node, index_to_code, language)
        dfg = []
        for index_left in indices:
            left_idx, left_code = index_to_code[index_left]
            dfg.append((left_code, left_idx, "computedFrom", [left_code], [left_idx]))
            states[left_code] = [left_idx]
        return _dedupe_dfg(dfg), states

    dfg = []
    for child in root_node.children:
        temp, states = _build_dfg(child, index_to_code, lines, states, language)
        dfg += temp
    return _dedupe_dfg(dfg), states


def get_data_flow(code, language, tree_sitter_language):
    try:
        parser = _make_parser(tree_sitter_language)
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        token_indices = tree_to_token_index(root_node)
        lines = code.split("\n")
        code_tokens = [index_to_code_token(index, lines) for index in token_indices]
        index_to_code = {index: (position, token) for position, (index, token) in enumerate(zip(token_indices, code_tokens))}
        dfg, _ = _build_dfg(root_node, index_to_code, lines, {}, language)
        dfg = sorted(dfg, key=lambda item: item[1])
        relevant_indices = set()
        for item in dfg:
            if len(item[-1]) != 0:
                relevant_indices.add(item[1])
            for value in item[-1]:
                relevant_indices.add(value)
        dfg = [item for item in dfg if item[1] in relevant_indices]
    except Exception:
        dfg = []
    merged = {}
    for item in dfg:
        if item[1] not in merged:
            merged[item[1]] = item
            continue
        merged[item[1]] = (
            item[0],
            item[1],
            item[2],
            list(set(merged[item[1]][3] + item[3])),
            list(set(merged[item[1]][4] + item[4])),
        )
    return list(merged.values())


def normalize_dataflow(dataflow):
    variable_mapping = {}
    next_index = 0
    normalized = []
    for item in dataflow:
        variable_name = item[0]
        relationship = item[2]
        parent_names = item[3]
        for name in parent_names:
            if name not in variable_mapping:
                variable_mapping[name] = f"var_{next_index}"
                next_index += 1
        if variable_name not in variable_mapping:
            variable_mapping[variable_name] = f"var_{next_index}"
            next_index += 1
        normalized.append((variable_mapping[variable_name], relationship, [variable_mapping[name] for name in parent_names]))
    return normalized


def corpus_syntax_match(references, candidates, lang):
    if not native_runtime_available():
        raise ImportError("Native CodeBLEU requires tree-sitter and tree-sitter-language-pack.")
    tree_sitter_language = get_tree_sitter_language(lang)
    parser = _make_parser(tree_sitter_language)
    match_count = 0
    total_count = 0
    for references_sample, candidate in zip(references, candidates):
        cleaned_candidate = remove_comments_and_docstrings(candidate, lang)
        candidate_tree = parser.parse(bytes(cleaned_candidate, "utf8")).root_node

        def get_all_sub_trees(root_node):
            node_stack = [(root_node, 1)]
            subtree_text = []
            while node_stack:
                current_node, current_depth = node_stack.pop()
                subtree_text.append([str(current_node), current_depth])
                for child_node in current_node.children:
                    if len(child_node.children) != 0:
                        node_stack.append((child_node, current_depth + 1))
            return subtree_text

        candidate_subtrees = [item[0] for item in get_all_sub_trees(candidate_tree)]
        for reference in references_sample:
            cleaned_reference = remove_comments_and_docstrings(reference, lang)
            reference_tree = parser.parse(bytes(cleaned_reference, "utf8")).root_node
            reference_subtrees = [item[0] for item in get_all_sub_trees(reference_tree)]
            for subtree in reference_subtrees:
                if subtree in candidate_subtrees:
                    match_count += 1
            total_count += len(reference_subtrees)
    if total_count == 0:
        return 0.0
    return float(match_count / total_count)


def corpus_dataflow_match(references, candidates, lang):
    if not native_runtime_available():
        raise ImportError("Native CodeBLEU requires tree-sitter and tree-sitter-language-pack.")
    tree_sitter_language = get_tree_sitter_language(lang)
    match_count = 0
    total_count = 0
    for references_sample, candidate in zip(references, candidates):
        cleaned_candidate = remove_comments_and_docstrings(candidate, lang)
        candidate_dfg = normalize_dataflow(get_data_flow(cleaned_candidate, lang, tree_sitter_language))
        for reference in references_sample:
            cleaned_reference = remove_comments_and_docstrings(reference, lang)
            reference_dfg = normalize_dataflow(get_data_flow(cleaned_reference, lang, tree_sitter_language))
            if reference_dfg:
                total_count += len(reference_dfg)
                candidate_pool = list(candidate_dfg)
                for dataflow in reference_dfg:
                    if dataflow in candidate_pool:
                        match_count += 1
                        candidate_pool.remove(dataflow)
    if total_count == 0:
        return 0.0
    return float(match_count / total_count)


def _keyword_set_for_language(lang):
    keywords = set(_LANGUAGE_KEYWORDS.get(lang, set()))
    return keywords or set(_UNIVERSAL_KEYWORDS)


def calc_codebleu(
    references,
    predictions,
    lang,
    weights=(0.25, 0.25, 0.25, 0.25),
    tokenizer=None,
):
    if lang not in AVAILABLE_LANGUAGES:
        supported = ", ".join(AVAILABLE_LANGUAGES)
        raise ValueError(f"Language {lang} is not supported. Available languages: {supported}")
    if len(references) != len(predictions):
        raise ValueError("Number of references and predictions should be the same.")
    if len(weights) != 4:
        raise ValueError("weights should contain exactly 4 values.")
    if tokenizer is None:
        tokenizer = lambda text: text.split()

    normalized_references = [[value.strip() for value in reference] if isinstance(reference, list) else [reference.strip()] for reference in references]
    normalized_predictions = [value.strip() for value in predictions]
    tokenized_hypotheses = [tokenizer(value) for value in normalized_predictions]
    tokenized_references = [[tokenizer(value) for value in reference] for reference in normalized_references]
    keywords = _keyword_set_for_language(lang)

    ngram_match_score = _corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), weighted=False)

    def make_weights(tokens):
        return {token: 1.0 if token in keywords else 0.2 for token in tokens}

    weighted_references = [
        [[reference_tokens, make_weights(reference_tokens)] for reference_tokens in reference_group]
        for reference_group in tokenized_references
    ]
    weighted_ngram_match_score = _corpus_bleu(
        weighted_references,
        tokenized_hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        weighted=True,
    )
    syntax_match_score = corpus_syntax_match(normalized_references, normalized_predictions, lang)
    dataflow_match_score = corpus_dataflow_match(normalized_references, normalized_predictions, lang)

    alpha, beta, gamma, theta = weights
    codebleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )
    return {
        "codebleu": float(codebleu_score),
        "ngram_match_score": float(ngram_match_score),
        "weighted_ngram_match_score": float(weighted_ngram_match_score),
        "syntax_match_score": float(syntax_match_score),
        "dataflow_match_score": float(dataflow_match_score),
    }
