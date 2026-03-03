# Usage Guide

## Feature Weights

Matheel blends feature scores with normalized weights.

You can keep the legacy `Ws`, `Wl`, and `Wj` arguments, or pass arbitrary feature weights:

```python
feature_weights = {
    "semantic": 0.5,
    "levenshtein": 0.2,
    "jaro_winkler": 0.1,
    "code_metric": 0.2,
}
```

The final weights are normalized automatically, so the total score stays in a bounded range as long as each feature score is also bounded.

## Chunking

Built-in chunking works without extra dependencies:

- `lines`
- `tokens`
- `characters`

With `matheel[chunking]`, Matheel will prefer Chonkie chunkers:

- `code` / `codechunker`
- `chonkie_token`
- `chonkie_word`
- `chonkie_sentence`
- `chonkie_recursive`

Extra chunker-specific options can be passed as `name=value` pairs:

```bash
matheel compare codes/ \
  --chunking-method code \
  --chunk-language java \
  --chunker-option include_line_numbers=true
```

## Backend Routing

Use `vector_backend=auto` when the model is hosted on Hugging Face and exposes a known library:

- `sentence-transformers` -> dense single-vector path
- `model2vec` -> static embedding path
- `PyLate` -> multivector late-interaction path

If the optional backend library is not installed, Matheel falls back to the simplest compatible local path.

## Directory Inputs

CLI and Python API accept either:

- a directory
- a ZIP archive

The Gradio app remains ZIP-only for uploads.
