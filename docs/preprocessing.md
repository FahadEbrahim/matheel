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
- `synsem_basic`
  Currently behaves the same as `basic`.

## What It Changes

- `/* ... */` block comments are removed.
- `// ...` line comments are removed.
- `# ...` comments are removed, except common C/C++ preprocessor directives such as `#include` and `#define`.
- Excess whitespace is collapsed in the `basic` modes.

## When To Use It

- Use `none` when formatting, comments, or whitespace are meaningful for your task.
- Use `normalize` when you want stable layout without removing comments.
- Use `basic` for most code-similarity runs where comments and formatting should not dominate.

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
