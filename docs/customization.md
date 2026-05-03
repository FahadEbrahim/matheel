# Custom Similarity Algorithms

Matheel supports custom pair scorers without editing package internals. A custom scorer is useful when you want to compare a baseline, prototype a new method, or reuse a project-specific signal in dataset evaluation.

## Contract

Create a Python module with a callable `score_pair`:

```python
# my_algorithm.py

def score_pair(code_a, code_b, bias=0.0, dataset_context=None, row=None, **kwargs):
    _ = (dataset_context, row, kwargs)
    base = 1.0 if code_a.strip() == code_b.strip() else 0.0
    return base + float(bias)
```

`score_pair` must return a finite numeric score. Matheel passes:

- `code_a` and `code_b`: the two source-code strings.
- named values from `algorithm_options`.
- `algorithm_options`: the full validated options mapping, when accepted by the function.
- `dataset_context`: the value returned by `prepare_dataset`, when provided.
- `row`: pair or retrieval row metadata, when available.

Options must be a mapping with non-empty string keys and JSON-serializable values.

## Optional Preparation

Define `prepare_dataset` when the algorithm needs reusable dataset-level state:

```python
def prepare_dataset(dataset, prepared_texts=None, bias=0.0, **kwargs):
    _ = kwargs
    return {
        "bias": float(bias),
        "file_count": len(dataset.files),
        "prepared_texts": dict(prepared_texts or {}),
    }


def score_pair(code_a, code_b, dataset_context=None, **kwargs):
    _ = kwargs
    base = 1.0 if code_a == code_b else 0.0
    return base + dataset_context["bias"]
```

For normalized pair datasets, `dataset` is a `PairDataset`. For retrieval datasets, it is a `RetrievalDataset`.
For `matheel compare` archive or directory runs, it is a small mapping with source file names, prepared code strings, and code count.
When accepted by the function, `prepared_texts` contains the exact file-id-to-text mapping that will be scored after preprocessing.

## CLI

Run a custom module against a directory or ZIP archive:

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare sample_pairs.zip \
  --algorithm-path ./my_algorithm.py \
  --algorithm-option bias=0.2 \
  --reproducibility-out results/reproducibility.json
```

Evaluate a normalized pair dataset:

```bash
matheel evaluate-pairs ./data/pairs \
  --algorithm-path ./my_algorithm.py \
  --algorithm-option bias=0.2 \
  --scores-out results/pair_scores.csv \
  --metrics-out results/pair_metrics.json \
  --reproducibility-out results/pair_reproducibility.json
```

Evaluate a normalized retrieval dataset:

```bash
matheel evaluate-retrieval ./data/retrieval \
  --algorithm-path ./my_algorithm.py \
  --algorithm-option bias=0.2 \
  --scores-out results/retrieval_scores.csv \
  --metrics-out results/retrieval_metrics.json \
  --reproducibility-out results/retrieval_reproducibility.json
```

`--algorithm-option` accepts JSON-like scalar values such as `0.2`, `true`, `false`, `null`, quoted strings, lists, and objects.
Path-loaded algorithm files can import helper modules placed beside the algorithm file.

## Comparison Suite

Custom algorithms can be included in comparison-suite configs:

```json
[
  {
    "run_name": "exact_match_custom",
    "algorithm_path": "./my_algorithm.py",
    "algorithm_options": {
      "bias": 0.2
    },
    "preprocess_mode": "basic",
    "number_results": 20
  },
  {
    "run_name": "lexical_baseline",
    "feature_weights": {
      "levenshtein": 1.0
    },
    "number_results": 20
  }
]
```

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare-suite sample_pairs.zip runs.json \
  --summary-out results/summary.csv \
  --details-dir results/runs \
  --reproducibility-out results/suite_reproducibility.json
```

The summary includes algorithm name, function, options, package version when available, and the custom source SHA-256 fingerprint.

## Python API

```python
from matheel.algorithms import resolve_pair_algorithm, score_pair_with_algorithm

algorithm = resolve_pair_algorithm("./my_algorithm.py")
score = score_pair_with_algorithm(
    "print(1)",
    "print(1)",
    algorithm,
    algorithm_options={"bias": 0.2},
)
```

For datasets:

```python
from matheel.evaluation import evaluate_pair_dataset

scored, metrics = evaluate_pair_dataset(
    "./data/pairs",
    algorithm="./my_algorithm.py",
    algorithm_options={"bias": 0.2},
)
```

See `examples/custom/custom_algorithm_demo.py` for a runnable local example.

## Reproducibility

Custom runs attach metadata to scored DataFrames and optional reproducibility JSON files:

- Matheel and dependency package versions.
- source dataset/archive fingerprint.
- custom algorithm name, function, package version, options, and source SHA-256 fingerprint.

Real datasets and private algorithm files are not bundled by Matheel. Keep your own copies and follow the terms of the source that provided them.
