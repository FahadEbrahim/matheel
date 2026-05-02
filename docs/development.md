# Development

This page covers local development checks for contributors and release preparation.

## Editable Install

Install Matheel in editable mode with the development tools:

```bash
python -m pip install -e ".[dev]"
```

## Local Checks

Run the default offline-friendly test suite:

```bash
python -m pytest
```

Run the Ruff lint check:

```bash
python -m ruff check .
```

Real-model integration tests are opt-in because they may need optional backends, cached model weights, or network access:

```bash
python -m pytest -m integration
```

## Package Checks

When preparing release or packaging changes, build the package and check the distribution metadata:

```bash
python -m build
python -m twine check dist/*
```

Use the [release checklist](release_checklist.md) for the full release flow.

## Cache Safety

Matheel uses module-level caches for Hugging Face model metadata, detected tokenizer limits, CodeBLEU keyword sets, tree-sitter parsers, and CodeBERTScore scorer objects.

Request-path cache writes are protected with locks so concurrent API or Gradio calls do not mutate the same cache at the same time. Keyword cache entries are stored as immutable values. Pair-level caches inside prepared run contexts are scoped to that run context and should not be reused across independent requests.
