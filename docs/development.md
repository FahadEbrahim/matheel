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
