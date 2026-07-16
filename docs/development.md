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

Install the full optional stack before running them locally:

```bash
python -m pip install -e ".[all,dev]"
```

The `Real Model Integration Tests` GitHub Actions workflow runs weekly on Python 3.12 and can also be started manually on any supported Python version. It installs `matheel[all,dev]`, caches Hugging Face model files under `~/.cache/huggingface`, and runs `python -m pytest -m integration`. It is not part of default pull-request CI, so normal tests remain offline-friendly.

Pull-request CI also runs selected compatibility checks against the external `codebleu` package and builds the wheel and source distribution. The package job validates metadata, installs the wheel into a clean environment, imports Matheel, and exercises the CLI help command.

## Gradio Checks

The default development install does not include Gradio, so Gradio-specific tests skip when its optional dependencies are unavailable. Install the Gradio and development extras to run the same focused checks as pull-request CI:

```bash
python -m pip install -e ".[dev,gradio]"
python -m pytest tests/test_gradio_app.py tests/test_gradio_html_utils.py
```

These tests cover workflow helpers, the stable tab and primary-action structure, and a live server smoke check of the root page and `/config` endpoint. They intentionally avoid model downloads.

## Package Checks

When preparing release or packaging changes, build the package and check the distribution metadata:

```bash
python -m build
python -m twine check dist/*
```

## Cache Safety

Matheel uses module-level caches for Hugging Face model metadata, detected tokenizer limits, CodeBLEU keyword sets, tree-sitter parsers, and CodeBERTScore scorer objects.

Request-path cache writes are protected with locks so concurrent API or Gradio calls do not mutate the same cache at the same time. Keyword cache entries are stored as immutable values. Pair-level caches inside prepared run contexts are scoped to that run context and should not be reused across independent requests.
