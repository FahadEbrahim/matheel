# Matheel

[![Tests](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/FahadEbrahim/matheel/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/matheel.svg)](https://pypi.org/project/matheel/)
[![Python versions](https://img.shields.io/pypi/pyversions/matheel.svg)](https://pypi.org/project/matheel/)
[![Latest release](https://img.shields.io/github/v/release/FahadEbrahim/matheel.svg)](https://github.com/FahadEbrahim/matheel/releases)
[![License](https://img.shields.io/github/license/FahadEbrahim/matheel.svg)](LICENSE)

Matheel is a Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code evaluation metrics in one workflow.

## Demos

- Hugging Face Space demo: [buelfhood/matheel-framework](https://huggingface.co/spaces/buelfhood/matheel-framework)
- Gradio Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/gradio_app/matheel_gradio_colab_demo.ipynb)
- Examples Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_examples_colab.ipynb)

## Installation

Use Python `3.10` to `3.13`. Installation can take some time.

Base install:

```bash
pip install matheel
```

Optional extras:

```bash
pip install "matheel[semantic]"
pip install "matheel[chunking]"
pip install "matheel[metrics]"
pip install "matheel[gradio]"
pip install "matheel[all]"
```

`matheel[semantic]` installs the supported semantic backends. `matheel[chunking]` installs Chonkie chunkers. `matheel[metrics]` installs optional code metric runtimes. `matheel[gradio]` installs the web app dependencies. `matheel[all]` installs all supported optional backends.

Compatibility extras remain available for narrower installs: `sentence_transformers`, `model2vec`, `pylate`, and `chunking_code`.

Examples that use semantic weights assume `matheel[semantic]` or `matheel[all]` is installed. See the [usage guide](https://fahadebrahim.github.io/matheel/usage/) for more install details.

## Quick Start

```bash
matheel compare sample_pairs.zip \
  --model huggingface/CodeBERTa-small-v1 \
  --feature-weight semantic=0.7 \
  --feature-weight levenshtein=0.3 \
  --threshold 0.2 \
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

See the [usage guide](https://fahadebrahim.github.io/matheel/usage/) for archive, suite, chunking, embedding, and code-metric examples.

## Docs

- Published docs: [fahadebrahim.github.io/matheel](https://fahadebrahim.github.io/matheel/)
- Source docs: [docs/index.md](docs/index.md)
- Usage guide: [docs/usage.md](docs/usage.md)
- Development: [docs/development.md](docs/development.md)

## Development

Install Matheel in editable mode with the development tools, then run the default checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

More development and release checks are in the [development docs](docs/development.md).

## License

This project is licensed under the [MIT License](LICENSE).
