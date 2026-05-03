# Comparison Suite

The comparison suite runs multiple configurations against the same directory or ZIP archive and returns:

- a summary table
- one result DataFrame per run
- optional summary/detail files on disk

This is the right interface for repeatable ablations, backend comparisons, and parameter sweeps.
For calibrated decision thresholds, keep labels outside the suite output and use the score reports with the calibration helpers described in [Scoring and calibration](scoring.md).
Custom `score_pair` modules can also be included when you need to compare project-specific algorithms with Matheel's built-in scorers.

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
For custom runs, provide `algorithm_path` and optional `algorithm_options`; feature weights are not required for those runs.
Relative `algorithm_path` values are resolved from the config file's directory.
Custom runs reject unsupported option names so configuration typos fail early.

## Outputs

### Summary DataFrame

One row per run, including:

- `run_name`
- `pair_count`
- `elapsed_seconds`
- `feature_set`
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
- `algorithm_name`
- `algorithm_function`
- `algorithm_package_version`
- `algorithm_options`
- `algorithm_source_sha256`

`elapsed_seconds` is rounded to 4 decimal places. `feature_set` lists the active nonzero feature weights for the run.
For custom algorithm runs, `feature_set` is `custom`.

### Optional Files

- `summary_out`
  Writes the summary as CSV or JSON.
- `details_dir`
  Writes one CSV per run with the detailed ranked pairs.
- `output_format`
  `csv` or `json` for the summary file.
- `progress`
  Enables tqdm progress bars when set to `True`.
- `progress_callback`
  Receives structured progress event dictionaries for run-level and pair-level work.
- `reproducibility_out`
  Writes a JSON snapshot with package versions, source fingerprint, normalized run configs, and run metadata.
- Numeric score fields are rounded to 4 decimal places in suite output artifacts.

## Python Example

```python
from matheel.comparison_suite import run_comparison_suite
from examples.sample_data import write_sample_archive

sample_archive = write_sample_archive("sample_pairs.zip", overwrite=True)

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
        "run_name": "custom_exact_match",
        "options": {
            "algorithm_path": "./my_algorithm.py",
            "algorithm_options": {"bias": 0.2},
            "preprocess_mode": "basic",
            "number_results": 20,
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

summary, details = run_comparison_suite(
    sample_archive,
    runs,
    reproducibility_out="results/reproducibility.json",
    progress=True,
)
print(summary)
print(details["dense_baseline"].head())
```
