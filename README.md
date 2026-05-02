# Matheel

[![Tests](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/matheel.svg?cacheSeconds=3600)](https://pypi.org/project/matheel/)
[![Python versions](https://img.shields.io/pypi/pyversions/matheel.svg?cacheSeconds=3600)](https://pypi.org/project/matheel/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://fahadebrahim.github.io/matheel/)
[![Latest release](https://img.shields.io/github/v/release/FahadEbrahim/matheel.svg)](https://github.com/FahadEbrahim/matheel/releases)
[![License](https://img.shields.io/github/license/FahadEbrahim/matheel.svg)](LICENSE)

Matheel is a Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code evaluation metrics in one workflow.

## Installation

Matheel supports Python `3.10` to `3.13`.

```bash
pip install matheel
```

Optional semantic, chunking, metrics, and Gradio extras are covered in the [usage guide](https://fahadebrahim.github.io/matheel/usage/#installation).

## Quick Start

```bash
matheel compare sample_pairs.zip \
  --feature-weight levenshtein=1.0 \
  --num 10
```

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b):\n    return a + b\n",
    "def add(x, y):\n    return x + y\n",
    feature_weights={"levenshtein": 1.0},
)
print(round(score, 4))
```

See the [usage guide](https://fahadebrahim.github.io/matheel/usage/) for semantic models, archives, comparison suites, chunking, preprocessing, and code metrics.

## Links

- Documentation: [fahadebrahim.github.io/matheel](https://fahadebrahim.github.io/matheel/)
- Hugging Face Space demo: [buelfhood/matheel-framework](https://huggingface.co/spaces/buelfhood/matheel-framework)
- Gradio Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/gradio_app/matheel_gradio_colab_demo.ipynb)
- Examples Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_examples_colab.ipynb)
- Examples folder: [examples/](https://github.com/FahadEbrahim/matheel/tree/main/examples)
- Development: [development docs](https://fahadebrahim.github.io/matheel/development/)

## Development

Contributor setup, tests, linting, package checks, and release preparation are documented in the [development docs](https://fahadebrahim.github.io/matheel/development/).

## License

This project is licensed under the [MIT License](LICENSE).
