# Leaderboard

Matheel leaderboards rank algorithms across normalized pair-classification and retrieval datasets. The workflow is local and offline: users provide normalized datasets or source specs, Matheel scores them, then exports per-dataset and aggregate ranking tables.

## Manifest

Create a JSON manifest with datasets, algorithms, and metrics:

```json
{
  "name": "tiny_leaderboard",
  "seed": 7,
  "pair_metrics": ["f1", "auroc"],
  "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
  "datasets": [
    {
      "name": "tiny_pairs",
      "task": "pair",
      "path": "./normalized_pairs",
      "threshold": 0.5
    },
    {
      "name": "tiny_retrieval",
      "task": "retrieval",
      "path": "./normalized_retrieval",
      "k": 10
    }
  ],
  "algorithms": [
    {
      "name": "levenshtein",
      "feature_weights": {"levenshtein": 1.0}
    },
    {
      "name": "custom_exact",
      "algorithm_path": "./custom_algorithm.py"
    }
  ]
}
```

Relative dataset paths and custom algorithm paths are resolved from the manifest file location.

## CLI

```bash
matheel leaderboard leaderboard.json \
  --output-dir leaderboard_artifacts \
  --basename run_001
```

Artifacts:

- `run_001_per_dataset.csv`: one row per dataset, algorithm, and metric with per-dataset ranks.
- `run_001_aggregate.csv`: mean and median metric scores per algorithm, task, and metric.
- `run_001.json`: machine-readable metadata, manifest, and ranking rows.
- `run_001.html`: static aggregate and per-dataset tables.
- `run_001_reproducibility.json`: package versions and normalized manifest metadata.

The JSON artifact also includes dataset cards and algorithm cards. Cards keep concise metadata such as task family, counts, source type, license when present, sanitized source identifiers, algorithm options, package versions, and content fingerprints. Local absolute paths and credential-like fields are not included in card fields.

The HTML artifact is a static benchmark report. It bundles aggregate rankings, per-dataset rankings, dataset cards, algorithm cards, and sanitized links to the exported CSV/JSON metadata files.

## Python

```python
from matheel.leaderboard import load_leaderboard_manifest, run_leaderboard
from matheel.reports import write_benchmark_report

manifest = load_leaderboard_manifest("leaderboard.json")
report, artifacts = run_leaderboard(
    manifest,
    output_dir="leaderboard_artifacts",
)

write_benchmark_report(report, "leaderboard_artifacts/leaderboard_report.html")

print(report["aggregate"])
print(artifacts["json"])
```

The first implementation focuses on deterministic local tables. Richer static report pages and Gradio inspection views are tracked separately.
