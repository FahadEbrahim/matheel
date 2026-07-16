# Matheel

[![Tests](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/matheel.svg?cacheSeconds=3600&release=v0.5.6)](https://pypi.org/project/matheel/)
[![Python versions](https://img.shields.io/pypi/pyversions/matheel.svg?cacheSeconds=3600)](https://pypi.org/project/matheel/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://fahadebrahim.github.io/matheel/)
[![Latest release](https://img.shields.io/github/v/release/FahadEbrahim/matheel.svg)](https://github.com/FahadEbrahim/matheel/releases)
[![License](https://img.shields.io/github/license/FahadEbrahim/matheel.svg)](LICENSE)

Matheel is a Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code evaluation metrics in one workflow.

## Installation

Matheel supports Python `3.10` to `3.13`.

Recommended install:

```bash
pip install "matheel[all]"
```

For a lightweight install without optional semantic, chunking, metrics, and Gradio dependencies:

```bash
pip install matheel
```

Installation options are covered in the [usage guide](https://fahadebrahim.github.io/matheel/usage/#installation).

## Quick Start

Create a tiny archive and compare it with the CLI:

```bash
python - <<'PY'
from zipfile import ZipFile

with ZipFile("sample_pairs.zip", "w") as archive:
    archive.writestr("a.py", "def add(a, b):\n    return a + b\n")
    archive.writestr("b.py", "def add(x, y):\n    return x + y\n")
    archive.writestr("c.py", "def sub(a, b):\n    return a - b\n")
PY

matheel compare sample_pairs.zip \
  --feature-weight levenshtein=1.0 \
  --num 10
```

Or score a pair directly from Python:

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b):\n    return a + b\n",
    "def add(x, y):\n    return x + y\n",
    feature_weights={"levenshtein": 1.0},
)
print(round(score, 4))
```

See the [usage guide](https://fahadebrahim.github.io/matheel/usage/) for semantic models, archives, comparison suites, custom algorithms, chunking, preprocessing, visualization, leaderboards, and code metrics.

## Links

- Documentation: [fahadebrahim.github.io/matheel](https://fahadebrahim.github.io/matheel/)
- Hugging Face Space demo: [buelfhood/matheel-framework](https://huggingface.co/spaces/buelfhood/matheel-framework)
- Core workflows Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/01_core_workflows.ipynb)
- Dataset workflows Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/02_datasets_and_reproducibility.ipynb)
- Custom algorithms Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/03_custom_algorithms.ipynb)
- Gradio Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/04_gradio_app.ipynb)
- Visualization and leaderboard Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/05_visualization_and_leaderboard.ipynb)
- Examples folder: [examples/](https://github.com/FahadEbrahim/matheel/tree/main/examples)
- Development: [development docs](https://fahadebrahim.github.io/matheel/development/)

## Development

Contributor setup, tests, linting, package checks, and release preparation are documented in the [development docs](https://fahadebrahim.github.io/matheel/development/).

## License

This project is licensed under the [MIT License](LICENSE).
