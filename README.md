# Matheel

Matheel is a simple, function-based Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code-aware metrics without forcing a class-heavy API.

## Installation

Use Python `3.10` to `3.12`.

Base install:

```bash
pip install matheel
```

Optional extras:

```bash
pip install "matheel[chunking]"
pip install "matheel[chunking_code]"
pip install "matheel[model2vec]"
pip install "matheel[pylate]"
pip install "matheel[gradio]"
pip install "matheel[all]"
pip install "matheel[dev]"
```

Tested Python 3.12 constraints:

```bash
pip install -c constraints/py312.txt .
```

`matheel[all]` installs the currently supported optional backends in one command: Chonkie code chunking, model2vec, PyLate, and the Gradio app dependencies.

## Quick Start

The repo includes a small Java archive at `sample_pairs.zip` for smoke tests.

CLI:

```bash
matheel compare sample_pairs.zip \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --feature-weight semantic=0.7 \
  --feature-weight levenshtein=0.3 \
  --threshold 0.2 \
  --num 10
```

Python:

```python
from matheel.similarity import get_sim_list

results = get_sim_list(
    "sample_pairs.zip",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.2,
    number_results=10,
    feature_weights={
        "semantic": 0.7,
        "levenshtein": 0.3,
    },
)
print(results.head())
```

## Supported Languages

- Chunking is language-agnostic by default because it can split any text.
- `CodeBLEU`-style metrics are intentionally scoped to `Java`, `Python`, `C`, and `C++`.
- Generic preprocessing works across languages, but code-aware metrics are most defensible in that four-language scope.

## Supported Methods

Similarity features:

- `semantic`
- `levenshtein`
- `jaro_winkler`
- `code_metric`

Code metrics:

- `codebleu`
- `codebleu_ngram`
- `codebleu_weighted_ngram`
- `codebleu_syntax`
- `codebleu_dataflow`
- `crystalbleu`

Chunking methods:

- `none`
- Chonkie-backed when installed: `code`, `codechunker`, `chonkie_code`, `chonkie_token`, `chonkie_sentence`, `chonkie_recursive`, `chonkie_fast`

Vector backends:

- `auto`
- `sentence_transformers`
- `model2vec`
- `pylate`

Single-vector similarity functions:

- `cosine`
- `dot`
- `euclidean`
- `manhattan`

Sentence Transformers pooling methods:

- `mean`
- `max`
- `cls`
- `lasttoken`
- `mean_sqrt_len_tokens`
- `weightedmean`

`auto` inspects Hugging Face model metadata and routes to the correct backend when the model exposes a known library.

## Core Parts

- Preprocessing: whitespace and comment normalization before any scoring.
- Chunking: Chonkie-backed document splitting with per-method options.
- Vectors: dense single-vector, learned static single-vector, and multivector late interaction.
- Edit distance: normalized Levenshtein and Jaro-Winkler lexical metrics.
- Code metrics: built-in CodeBLEU-style metrics and CrystalBLEU.
- Comparison suite: run multiple configurations, rank them, and optionally write summary/detail artifacts.

## CLI

Compare a directory or ZIP archive:

```bash
matheel compare codes/ \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-backend auto \
  --max-token-length 256 \
  --feature-weight semantic=0.6 \
  --feature-weight levenshtein=0.2 \
  --feature-weight jaro_winkler=0.1 \
  --feature-weight code_metric=0.1 \
  --similarity-function dot \
  --pooling-method max \
  --preprocess-mode basic \
  --chunking-method code \
  --chunk-language python \
  --chunker-option include_line_numbers=true \
  --code-metric codebleu \
  --code-language python \
  --threshold 0.5 \
  --num 25
```

Run multiple configurations:

```bash
matheel compare-suite codes/ runs.json \
  --summary-out results/summary.csv \
  --details-dir results/runs
```

## Python API

Pairwise scoring:

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b):\n    return a + b\n",
    "def add(x, y):\n    return x + y\n",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_backend="auto",
    max_token_length=256,
    similarity_function="dot",
    pooling_method="max",
    preprocess_mode="basic",
    chunking_method="code",
    chunk_language="python",
    code_metric="codebleu",
    code_language="python",
    feature_weights={"semantic": 0.5, "code_metric": 0.5},
)
```

Directory or ZIP ranking:

```python
from matheel.similarity import get_sim_list

results = get_sim_list(
    "sample_codes",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.4,
    number_results=50,
    vector_backend="auto",
    max_token_length=256,
    chunking_method="chonkie_token",
    chunk_size=120,
    chunk_overlap=20,
    similarity_function="cosine",
    pooling_method="mean",
    feature_weights={
        "semantic": 0.7,
        "levenshtein": 0.15,
        "jaro_winkler": 0.15,
    },
)
```

## Hugging Face Routing

Matheel can inspect Hugging Face model metadata and route automatically:

- `sentence-transformers` models go to the Sentence Transformers path
- `model2vec` models go to the model2vec static path
- `PyLate` models go to the multivector late-interaction path

If metadata is unavailable, Matheel falls back to simple name and tag heuristics, then defaults to the Sentence Transformers path.

## Docs

- Docs index: [docs/index.md](docs/index.md)
- Quick usage: [docs/usage.md](docs/usage.md)
- Preprocessing: [docs/preprocessing.md](docs/preprocessing.md)
- Chunking: [docs/chunking.md](docs/chunking.md)
- Vectors: [docs/vectors.md](docs/vectors.md)
- Edit distance and feature weights: [docs/lexical.md](docs/lexical.md)
- Code metrics: [docs/code_metrics.md](docs/code_metrics.md)
- Comparison suite: [docs/comparison_suite.md](docs/comparison_suite.md)

The `docs/` folder is already structured well for a later GitHub Pages setup if you decide to publish the docs site from the repository.

## Examples

- Quick archive smoke test: [examples/sample_pairs_demo.py](examples/sample_pairs_demo.py)
- Preprocessing: [examples/preprocessing_demo.py](examples/preprocessing_demo.py)
- Chunking: [examples/chunking_demo.py](examples/chunking_demo.py)
- Vector backends: [examples/vectors_demo.py](examples/vectors_demo.py)
- Edit distance and feature weights: [examples/lexical_demo.py](examples/lexical_demo.py)
- Code metrics: [examples/code_metrics_demo.py](examples/code_metrics_demo.py)
- Comparison suite: [examples/comparison_suite_demo.py](examples/comparison_suite_demo.py)

## Gradio

The Gradio demo stays in `gradio_app/`. The core package and CLI can read either a ZIP archive or a directory. The Gradio upload flow remains ZIP-based.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
