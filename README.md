# Matheel

Matheel is a Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code evaluation metrics in one workflow.

## Demos

- Hugging Face Space demo: [buelfhood/matheel-framework](https://huggingface.co/spaces/buelfhood/matheel-framework)
- Gradio Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/gradio_app/matheel_gradio_colab_demo.ipynb)
- Examples Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_examples_colab.ipynb)

## Installation

Use Python `3.10` to `3.12`. Installation can take some time. 

Base install:

```bash
pip install matheel
```

Optional extras:

```bash
pip install "matheel[chunking]"
pip install "matheel[chunking_code]"
pip install "matheel[metrics]"
pip install "matheel[model2vec]"
pip install "matheel[pylate]"
pip install "matheel[gradio]"
pip install "matheel[all]"
pip install "matheel[dev]"
```

`matheel[all]` installs the currently supported optional backends in one command: Chonkie code chunking, metrics runtime dependencies (RUBY graph/tree, TSED, CodeBERTScore), model2vec, PyLate, and the Gradio app dependencies.

Matheel now ships a native CodeBLEU implementation that uses `tree_sitter_language_pack` for parser resolution, so real syntax/dataflow scoring no longer depends on installing the pip `codebleu` package. The pip package is still useful for validation/comparison work if you want to cross-check the native scores on selected examples; Matheel does not currently claim exact pip parity on every possible input.

## Quick Start

The repository root includes `sample_pairs.zip`, a small Java archive with:

- `code_1.java`
- `code_3_plag.java`
- additional plagiarised and non-plagiarised comparisons

CLI archive comparison:

```bash
matheel compare sample_pairs.zip \
  --model huggingface/CodeBERTa-small-v1 \
  --feature-weight semantic=0.7 \
  --feature-weight levenshtein=0.3 \
  --threshold 0.2 \
  --num 10
```

Python pairwise scoring with the sample pair:

```python
from zipfile import ZipFile

from matheel.similarity import calculate_similarity

with ZipFile("sample_pairs.zip") as archive:
    code_a = archive.read("code_1.java").decode("utf-8")
    code_b = archive.read("code_3_plag.java").decode("utf-8")

score = calculate_similarity(
    code_a,
    code_b,
    model_name="huggingface/CodeBERTa-small-v1",
    feature_weights={
        "semantic": 0.7,
        "levenshtein": 0.3,
    },
)

print(round(score, 4))
```

## Supported Scope

Supported languages:

- Chunking remains text-first and can run on any source text.
- Preprocessing heuristics and code-aware metrics are now regression-tested for a unified 20-language scope: `Java`, `Python`, `C`, `C++`, `Go`, `JavaScript`, `TypeScript`, `Kotlin`, `Scala`, `Swift`, `Solidity`, `Dart`, `PHP`, `Ruby`, `Rust`, `C#`, `Lua`, `Julia`, `R`, and `Objective-C` (`objc`).
- Native CodeBLEU with real syntax/dataflow now covers that same 20-language scope.

Similarity features:

- `semantic`
- `levenshtein`
- `jaro_winkler`
- `winnowing`
- `gst`
- `code_metric`

Code metrics:

- `codebleu`
- `codebleu_ngram`
- `codebleu_weighted_ngram`
- `codebleu_syntax`
- `codebleu_dataflow`
- `crystalbleu`
- `ruby`
- `tsed`
- `codebertscore`

Chunking methods:

- `none`
- `code`
- `chonkie_token`
- `chonkie_sentence`
- `chonkie_recursive`
- `chonkie_fast`

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

## CLI

Compare a directory or ZIP archive:

```bash
matheel compare codes/ \
  --model huggingface/CodeBERTa-small-v1 \
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

code_a = "def add(a, b):\n    return a + b\n"
code_b = "def add(x, y):\n    return x + y\n"

score = calculate_similarity(
    code_a,
    code_b,
    model_name="huggingface/CodeBERTa-small-v1",
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
    "sample_pairs.zip",
    model_name="huggingface/CodeBERTa-small-v1",
    threshold=0.4,
    number_results=10,
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
        "jaro_winkler": 0.05,
        "winnowing": 0.05,
        "gst": 0.05,
    },
)

print(results.head())
```

## Docs

- Docs folder landing page: [docs/README.md](docs/README.md)
- Canonical docs index: [docs/index.md](docs/index.md)
- Quick usage: [docs/usage.md](docs/usage.md)
- Preprocessing: [docs/preprocessing.md](docs/preprocessing.md)
- Chunking: [docs/chunking.md](docs/chunking.md)
- Vectors and routing: [docs/vectors.md](docs/vectors.md)
- Lexical metrics and baselines: [docs/lexical.md](docs/lexical.md)
- Code metrics: [docs/code_metrics.md](docs/code_metrics.md)
- Comparison suite: [docs/comparison_suite.md](docs/comparison_suite.md)

## Examples

- Colab walkthrough: [examples/matheel_examples_colab.ipynb](examples/matheel_examples_colab.ipynb)
- Quick archive check: [examples/sample_pairs_demo.py](examples/sample_pairs_demo.py)
- Preprocessing: [examples/preprocessing_demo.py](examples/preprocessing_demo.py)
- Chunking: [examples/chunking_demo.py](examples/chunking_demo.py)
- Vector backends: [examples/vectors_demo.py](examples/vectors_demo.py)
- Lexical metrics and baselines: [examples/lexical_demo.py](examples/lexical_demo.py)
- Code metrics: [examples/code_metrics_demo.py](examples/code_metrics_demo.py)
- Comparison suite: [examples/comparison_suite_demo.py](examples/comparison_suite_demo.py)

All examples use the same sample pair from `sample_pairs.zip`: `code_1.java` and `code_3_plag.java`.

## Gradio

The Gradio demo stays in `gradio_app/` and is aligned with the Hugging Face Space setup. The UI supports embeddings, lexical metrics, baseline algorithms (Winnowing and GST), and the code-aware metrics (CodeBLEU, CrystalBLEU, RUBY, TSED, CodeBERTScore), with metric-specific advanced fields. The core package and CLI can read either a ZIP archive or a directory; the Gradio upload flow remains ZIP-based.

## Acknowledgments

Matheel builds on several open-source libraries:

- [Sentence Transformers](https://pypi.org/project/sentence-transformers/)
- [Chonkie](https://pypi.org/project/chonkie/)
- [model2vec](https://pypi.org/project/model2vec/)
- [PyLate](https://pypi.org/project/pylate/)
- [RapidFuzz](https://pypi.org/project/rapidfuzz/)
- [tree-sitter-language-pack](https://pypi.org/project/tree-sitter-language-pack/)
- [NetworkX](https://pypi.org/project/networkx/)
- [APTED](https://pypi.org/project/apted/)
- [bert-score](https://pypi.org/project/bert-score/)
- [Gradio](https://pypi.org/project/gradio/)

The project also depends on the standard scientific Python stack and related tooling, including NumPy, pandas, Click, SentencePiece, and func-timeout.

## License

This project is licensed under the [MIT License](LICENSE).
