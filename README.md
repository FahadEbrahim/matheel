This is the repository for the demonstration paper "Matheel: A Hybrid Source Code Plagiarism Detection Software".

# Matheel

Matheel is a Python package for source-code similarity analysis. It keeps a simple, function-based interface while supporting preprocessing, chunking, multiple vector backends, and code-aware metrics.

## Features

- Semantic similarity with transformer embeddings, static hashed vectors, or multivector late interaction.
- Lexical similarity with Levenshtein and Jaro-Winkler components.
- Optional code-aware scoring with `CodeBLEU`-style components and `CrystalBLEU`.
- Shared core reused by the CLI, Python API, and Gradio app.
- Comparison suite for running multiple configurations and writing publication-friendly summary tables.

## Installation

Use Python `3.10` to `3.12`. For Apple Silicon, a clean Python `3.12` virtual environment is the safest default.

Example local setup:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
```

Base install:

```bash
pip install matheel
```

Install optional CodeBLEU package support:

```bash
pip install "matheel[metrics]"
```

Install development tools:

```bash
pip install "matheel[dev]"
```

## CLI Usage

Basic comparison over a ZIP archive or a directory:

```bash
matheel compare codes/ \
  --model Salesforce/codet5p-110m-embedding \
  --preprocess-mode basic \
  --chunking-method tokens \
  --chunk-size 120 \
  --vector-backend multivector \
  --code-metric codebleu \
  --code-metric-weight 0.2 \
  --threshold 0.5 \
  --num 50
```

Run a comparison suite from a JSON config file:

```bash
matheel compare-suite codes/ runs.json \
  --summary-out results/summary.csv \
  --details-dir results/runs \
  --format csv
```

Example `runs.json`:

```json
[
  {
    "run_name": "baseline",
    "model_name": "Salesforce/codet5p-110m-embedding",
    "number_results": 25
  },
  {
    "run_name": "mv_codebleu",
    "model_name": "Salesforce/codet5p-110m-embedding",
    "chunking_method": "tokens",
    "chunk_size": 120,
    "vector_backend": "multivector",
    "code_metric": "codebleu",
    "code_metric_weight": 0.2,
    "number_results": 25
  }
]
```

## Python API Usage

Pairwise similarity:

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "int value = 1;",
    "int value = 1;",
    0.7,
    0.2,
    0.1,
    "Salesforce/codet5p-110m-embedding",
    preprocess_mode="basic",
    vector_backend="multivector",
    chunking_method="tokens",
    chunk_size=120,
    code_metric="codebleu",
    code_metric_weight=0.2,
)

print(score)
```

Archive-wide ranking from a ZIP file or a directory:

```python
from matheel.similarity import get_sim_list

results = get_sim_list(
    "sample_codes",
    0.7,
    0.2,
    0.1,
    "Salesforce/codet5p-110m-embedding",
    0.5,
    50,
    preprocess_mode="basic",
    chunking_method="tokens",
    chunk_size=120,
    vector_backend="multivector",
    code_metric="crystalbleu",
    code_metric_weight=0.15,
)

print(results)
```

Comparison suite:

```python
from matheel.comparison_suite import run_comparison_suite

summary, results_by_run = run_comparison_suite(
    "sample_codes",
    [
        {"run_name": "baseline", "model_name": "Salesforce/codet5p-110m-embedding"},
        {
            "run_name": "static_codebleu",
            "model_name": "Salesforce/codet5p-110m-embedding",
            "vector_backend": "static_hash",
            "static_vector_dim": 512,
            "code_metric": "codebleu",
            "code_metric_weight": 0.2,
        },
    ],
    summary_out="results/summary.csv",
    details_dir="results/runs",
)

print(summary)
```

## Gradio App

The `gradio_app/` folder contains the Gradio interface.

- CLI and Python API can read either a ZIP archive or a directory.
- Gradio keeps the upload flow ZIP-only.

## Notes

- Chunking is universal because it only splits text. Preprocessing is mostly generic, with comment/directive handling that is safest for Java, Python, C, and C++ style syntax.
- For publication claims, the code-aware metrics should be treated as officially scoped to `Java`, `Python`, `C`, and `C++`. `CodeBLEU`-style weighting is most defensible in that limited language set.
- `static_hash` is a lightweight dependency-free semantic backend for fast baselines.
- `multivector` reuses the selected embedding model over chunks and scores them with late-interaction MaxSim.
- CodeBLEU works with a local fallback implementation by default and uses the optional `codebleu` package automatically when installed.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgement

- The demo uses code written by SBERT. [Webpage](https://www.sbert.net/index.html), [Repo](https://github.com/UKPLab/sentence-transformers).
- The code is built with Gradio. [Webpage](https://www.gradio.app/), [Repo](https://github.com/gradio-app/gradio).
- The code uses RapidFuzz for edit distance. [Webpage](https://rapidfuzz.github.io/RapidFuzz/), [Repo](https://github.com/rapidfuzz/RapidFuzz).
