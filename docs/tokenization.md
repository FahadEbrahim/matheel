# Tokenization and Preprocessing Limits

Matheel combines text-first preprocessing, token-level baselines, embedding backends, and optional parser-backed code metrics. These layers are related, but they do not all use the same token stream.

## Pipeline Order

For `calculate_similarity(...)` and `get_sim_list(...)`, the main order is:

1. Read source text.
2. Apply `preprocess_mode`.
3. Build embeddings when the semantic feature is active.
4. Tokenize for lexical baselines and token-based code metrics.
5. Parse code for parser-derived lexical tokens or parser-backed code metrics when those modes are active and their runtimes are available.
6. Blend active feature scores with normalized feature weights.

Preprocessing happens before lexical, semantic, and code-aware scoring. If you enable `advanced`, identifiers and literals may already be canonicalized before token-based metrics run.

## Preprocessing Modes

- `none`
  Trims trailing whitespace and final surrounding whitespace only.
- `normalize`
  Normalizes line endings, trims trailing whitespace, and drops blank lines.
- `basic`
  Removes comments with language-aware string handling, drops blank lines, and collapses whitespace.
- `advanced`
  Runs `basic`, removes import-like lines, replaces string and numeric literals with placeholders, canonicalizes non-keyword identifiers, and collapses whitespace.

Preprocessing is intentionally text-first. It uses language-specific comment and import heuristics, but it is not an AST rewrite.

## Tokenization By Feature

| Feature or metric | Tokenization or parsing strategy |
| --- | --- |
| `levenshtein` | Compares the prepared strings directly. |
| `jaro_winkler` | Compares the prepared strings directly. |
| `winnowing` | Uses Matheel's code-token regex by default, or tree-sitter leaf node types with `lexical_tokenizer="parser"`. |
| `gst` | Uses the same lexical tokenizer selection as Winnowing. |
| `crystalbleu` | Uses the same code-token regex, then builds n-grams and discounts frequent n-grams. |
| `ruby` with `ruby_mode=ngram` | Uses the same code-token regex. |
| `ruby` with `ruby_mode=string` and `ruby_tokenizer=regex` | Uses the same code-token regex. |
| `ruby` with `ruby_tokenizer=tranx` | Splits punctuation and camel-case boundaries for a more granular token edit sequence. |
| CodeBLEU syntax/dataflow | Uses tree-sitter-backed syntax and DFG extraction for supported languages. |
| TSED | Parses syntax trees and compares tree edit distance when optional parser dependencies are installed. |
| CodeBERTScore | Uses the selected transformer tokenizer. |
| Semantic embeddings | Use the selected embedding backend's tokenizer or vectorizer. |

The shared code-token regex recognizes identifiers, numbers, and punctuation. It does not split snake_case or camelCase by default. For example, `totalCount` stays one token in regex mode; RUBY's `tranx` tokenizer splits it into `total` and `Count`.

## Parser-Derived Lexical Tokens

Set `lexical_tokenizer="parser"` in Python or `--lexical-tokenizer parser` in the CLI to use parser-derived leaf node types for `winnowing` and `gst`. The default is `raw`, preserving the regex token stream used in earlier Matheel releases.

Parser-derived lexical tokens make the lexical baselines less sensitive to identifier renaming. They still compare ordered token streams; they are not full AST or data-flow comparisons.

## Limitations

Important limitations:

- lexical token baselines see surface token order and punctuation
- parser-derived lexical tokens and parser-backed metrics depend on parser runtime availability
- unsupported languages fall back only where a metric explicitly supports fallback behavior
- preprocessing language hints improve comment/import handling, but they do not guarantee full parsing
- aggressive `advanced` preprocessing can hide identifier and literal differences that some workflows may care about

For parser-heavy comparisons, report the active preprocessing mode, `lexical_tokenizer`, tokenization-sensitive parameters, selected code metric, language, and parser availability alongside the score.
