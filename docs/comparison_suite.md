# Comparison Suite

The comparison suite runs multiple configurations against the same directory or ZIP archive and returns:

- a summary table
- one result DataFrame per run
- optional summary/detail files on disk

This is the right interface for repeatable ablations, backend comparisons, and parameter sweeps.

## Main Functions

- `parse_run_configs(...)`
- `load_run_configs(...)`
- `run_comparison_suite(...)`

## Inputs

### Source

`run_comparison_suite(...)` accepts the same source path as `get_sim_list(...)`:

- a directory
- a ZIP archive

### Run Config Format

Each run is a JSON object with:

- `run_name`
- `options`

Minimal example:

```json
[
  {
    "run_name": "dense_baseline",
    "options": {
      "model_name": "huggingface/CodeBERTa-small-v1",
      "vector_backend": "auto",
      "feature_weights": {
        "semantic": 0.7,
        "levenshtein": 0.3
      }
    }
  }
]
```

## Supported Option Normalization

The suite normalizes a few convenience aliases:

- `model` -> `model_name`
- `num` -> `number_results`

If no weights are supplied, the suite applies Matheel’s default feature blend.

## Outputs

### Summary DataFrame

One row per run, including:

- `run_name`
- `pair_count`
- `mean_score`
- `median_score`
- `max_score`
- `min_score`
- `std_score`
- `top_file_1`
- `top_file_2`
- `top_score`
- `vector_backend`
- `code_metric`
- `chunking_method`

### Optional Files

- `summary_out`
  Writes the summary as CSV or JSON.
- `details_dir`
  Writes one CSV per run with the detailed ranked pairs.
- `output_format`
  `csv` or `json` for the summary file.
- Numeric score fields are rounded to 4 decimal places in suite output artifacts.

## Python Example

```python
from matheel.comparison_suite import run_comparison_suite

runs = [
    {
        "run_name": "dense_baseline",
        "options": {
            "model_name": "huggingface/CodeBERTa-small-v1",
            "vector_backend": "sentence_transformers",
            "feature_weights": {"semantic": 0.7, "levenshtein": 0.3},
        },
    },
    {
        "run_name": "code_metric_blend",
        "options": {
            "model_name": "huggingface/CodeBERTa-small-v1",
            "vector_backend": "sentence_transformers",
            "chunking_method": "code",
            "chunk_language": "java",
            "code_metric": "codebleu",
            "code_language": "java",
            "code_metric_weight": 0.2,
            "feature_weights": {"semantic": 0.8, "code_metric": 0.2},
        },
    },
]

summary, details = run_comparison_suite("sample_pairs.zip", runs)
print(summary)
print(details["dense_baseline"].head())
```
