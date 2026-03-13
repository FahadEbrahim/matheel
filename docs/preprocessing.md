# Preprocessing

Preprocessing modifies code text before lexical, semantic, or code-aware scoring runs.

## Parameter

- `preprocess_mode`

## Supported Modes

- `none`
  Keeps the original code except for trailing-whitespace cleanup and final trimming.
- `normalize`
  Normalizes newlines, trims trailing whitespace, and removes blank lines.
- `basic`
  Removes block comments, removes line comments, drops blank lines, and collapses repeated whitespace.
- `advanced`
  Applies `basic`, strips import-like lines, normalizes string/number literals, canonicalizes identifiers, and collapses whitespace.

## What It Changes

- `/* ... */` block comments are removed.
- `// ...` line comments are removed.
- Lua `--[[ ... ]]` block comments and `-- ...` line comments are removed.
- `# ...` comments are removed, except common C/C++/C# preprocessor directives such as `#include`, `#define`, `#region`, and `#nullable`.
- Import-like lines (`import`, `from ... import`, `package`, `#include`, `#import`, `using`, `use`, `require`, `include`, `library(...)`, `source(...)`, and Node/Lua-style `require(...)` assignments) are stripped in `advanced`.
- String and numeric literals are normalized to placeholders in `advanced`.
- Non-keyword identifiers are canonicalized (`id1`, `id2`, ...) in `advanced`.
- Excess whitespace is collapsed in `basic` and `advanced`.

## Language Coverage

Preprocessing is still text-first, but the current regression-tested language set is:

- `java`
- `python`
- `c`
- `cpp`
- `go`
- `javascript`
- `typescript`
- `kotlin`
- `scala`
- `swift`
- `solidity`
- `dart`
- `php`
- `ruby`
- `rust`
- `csharp`
- `lua`
- `julia`
- `r`
- `objc`

The broader tree-sitter parser inventory makes future language additions realistic, but they should land with matching import/comment heuristics and tests before they are treated as officially supported preprocessing targets.

## When To Use It

- Use `none` when formatting, comments, or whitespace are meaningful for your task.
- Use `normalize` when you want stable layout without removing comments.
- Use `basic` for most code-similarity runs where comments and formatting should not dominate.
- Use `advanced` when you need stronger normalization across identifier renames and literal changes.

## Python Example

```python
from matheel.preprocessing import preprocess_code

cleaned = preprocess_code(
    """
    def add(a, b):  # helper
        return a + b
    """,
    mode="basic",
)
print(cleaned)
```

## CLI Example

```bash
matheel compare sample_pairs.zip \
  --preprocess-mode basic
```
