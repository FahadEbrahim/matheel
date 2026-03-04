# Usage Guide

This page is the quick-start entry point. The detailed parameter guides live in the topic pages linked below.

## Install

Base install:

```bash
pip install matheel
```

Full optional install:

```bash
pip install "matheel[all]"
```

For reproducible local work from the repo on Python 3.12:

```bash
pip install -c constraints/py312.txt .
```

## Quick Smoke Test

The repository root includes `sample_pairs.zip`, a small Java archive you can use immediately.

CLI:

```bash
matheel compare sample_pairs.zip \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --feature-weight semantic=0.7 \
  --feature-weight levenshtein=0.3
```

Python:

```python
from matheel.similarity import get_sim_list

results = get_sim_list(
    "sample_pairs.zip",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    feature_weights={"semantic": 0.7, "levenshtein": 0.3},
)
print(results.head())
```

## Documentation Map

- [docs/index.md](index.md)
- [docs/preprocessing.md](preprocessing.md)
- [docs/chunking.md](chunking.md)
- [docs/vectors.md](vectors.md)
- [docs/lexical.md](lexical.md)
- [docs/code_metrics.md](code_metrics.md)
- [docs/comparison_suite.md](comparison_suite.md)

## Interface Notes

- CLI and Python API accept either a directory or a ZIP archive.
- Gradio remains ZIP-only for uploads.
- `feature_weights` is the canonical scoring input.
- `vector_backend=auto` uses Hugging Face metadata and tag heuristics when available.
