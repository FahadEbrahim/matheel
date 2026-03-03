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
pip install "matheel[metrics]"
pip install "matheel[chunking]"
pip install "matheel[model2vec]"
pip install "matheel[pylate]"
pip install "matheel[dev]"
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

- Built-in: `none`, `lines`, `tokens`, `characters`
- Chonkie-backed when installed: `code`, `codechunker`, `chonkie_code`, `chonkie_token`, `chonkie_word`, `chonkie_sentence`, `chonkie_recursive`

Vector backends:

- `auto`
- `sentence_transformers`
- `model2vec`
- `pylate`
- `static_hash`

`auto` inspects Hugging Face model metadata and routes to the correct backend when the model exposes a known library.

## CLI

Compare a directory or ZIP archive:

```bash
matheel compare codes/ \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-backend auto \
  --feature-weight semantic=0.6 \
  --feature-weight levenshtein=0.2 \
  --feature-weight jaro_winkler=0.1 \
  --feature-weight code_metric=0.1 \
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
    0.7,
    0.2,
    0.1,
    "sentence-transformers/all-MiniLM-L6-v2",
    vector_backend="auto",
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
    0.7,
    0.2,
    0.1,
    "sentence-transformers/all-MiniLM-L6-v2",
    0.4,
    50,
    vector_backend="auto",
    chunking_method="chonkie_token",
    chunk_size=120,
    chunk_overlap=20,
    feature_weights="semantic=0.7,levenshtein=0.15,jaro_winkler=0.15",
)
```

## Hugging Face Routing

Matheel can inspect Hugging Face model metadata and route automatically:

- `sentence-transformers` models go to the Sentence Transformers path
- `model2vec` models go to the model2vec static path
- `PyLate` models go to the multivector late-interaction path

If metadata is unavailable, Matheel falls back to simple name and tag heuristics, then defaults to the Sentence Transformers path.

## Docs and Examples

- Usage guide: [docs/usage.md](docs/usage.md)
- Example run configs: [examples/runs/basic_runs.json](examples/runs/basic_runs.json)

## Gradio

The Gradio demo stays in `gradio_app/`. The core package and CLI can read either a ZIP archive or a directory. The Gradio upload flow remains ZIP-based.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
