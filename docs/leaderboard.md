# Leaderboard

Matheel leaderboards rank algorithms across normalized pair-classification and retrieval datasets. The workflow is local and offline: users provide normalized datasets or source specs, Matheel scores them, then exports per-dataset and aggregate ranking tables.

The Gradio app includes three leaderboard workflows:

- **Ready-made Leaderboard** immediately displays a committed public snapshot covering every registered dataset preset and every built-in algorithm preset.
- **Build Leaderboard** runs selected algorithm presets over uploaded normalized dataset ZIPs, shows all registered dataset presets, applies task-specific metric defaults, and exports standard leaderboard artifacts.
- **Inspect Artifacts** loads a leaderboard JSON or ZIP artifact and renders the aggregate table, per-dataset table, static report, and downloadable report bundle.

## Ready-made Public Snapshot

Open **Reports → Ready-made Leaderboard** to see rankings without uploading data or starting a benchmark run. The snapshot covers all six registered public dataset presets as seven task views: five pair-classification datasets plus SOCO14 and IRPlag retrieval views. All eight built-in algorithm presets are included.

This is a deterministic product demo, not a claim of full-corpus research performance. Each pair dataset uses a label-stratified sample capped at 128 pairs; each retrieval view is capped at 24 queries and 64 documents while retaining the selected queries' judged documents. It uses seed `7`, task-appropriate code languages, and the offline `static_hash` vector backend with 256 dimensions. The app displays these limits alongside the rankings.

Maintainers can regenerate the committed JSON artifact from the public source presets:

```bash
python scripts/build_ready_leaderboard.py
```

The builder downloads and normalizes the registered sources, creates deterministic samples, scores every algorithm preset, and writes `gradio_app/assets/ready_leaderboard.json`. Use **Build Leaderboard** or the CLI with a custom manifest when full-corpus evaluation, another embedding backend, or different sampling is required.

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
    "Lexical Only",
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

Algorithm entries can be full option objects or built-in preset names. Built-in presets include `Balanced`, `Lexical Only`, `Embedding Only`, `Code-Aware`, `Jaro-Winkler`, `Winnowing`, `GST`, and `CodeBLEU`.

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
- `run_001_details.html`: static dataset-card and algorithm-card detail page.
- `run_001_reproducibility.json`: package versions and normalized manifest metadata.

The JSON artifact also includes dataset cards and algorithm cards. Cards keep concise metadata such as task family, counts, source type, license when present, sanitized source identifiers, algorithm options, package versions, and content fingerprints. Local absolute paths and credential-like fields are not included in card fields.

The HTML artifact is a static benchmark report. It bundles aggregate rankings, per-dataset rankings, dataset cards, algorithm cards, and sanitized links to the exported CSV/JSON metadata files. The detail HTML artifact expands each dataset and algorithm card with the sanitized card JSON used by the benchmark report.

## Run Registry

Use a local benchmark registry to track exported leaderboard JSON artifacts across repeated runs:

```bash
matheel benchmark-registry add benchmark_registry.json leaderboard_artifacts/run_001.json \
  --run-name lexical_baseline

matheel benchmark-registry list benchmark_registry.json

matheel benchmark-registry compare benchmark_registry.json \
  --output benchmark_comparison.csv
```

The registry stores sanitized leaderboard payloads and artifact filenames. It is intended for local experiment tracking and does not store absolute dataset paths or credentials.

## Runnable Example

Run the synthetic local example to create one pair dataset, one retrieval dataset, a manifest, ranked tables, and HTML/JSON reproducibility artifacts:

```bash
python examples/evaluation/leaderboard_demo.py --overwrite
```

The example is deterministic and offline. It uses synthetic normalized datasets plus two algorithm entries: a lexical baseline and a tiny local custom scorer.

## Gradio Build Leaderboard

The Gradio **Reports → Build Leaderboard** workflow expects normalized dataset ZIP uploads. Unlike the ready-made sampled snapshot, custom runs do not fetch or bundle real datasets or credentials. The registered dataset table lists the current dataset presets, their task families, their source resolver, and the default evaluation plan:

- Pair-classification datasets use pair metrics such as `f1`, `accuracy`, `auroc`, and `average_precision`.
- Retrieval datasets use ranking metrics such as `mean_average_precision`, `mean_reciprocal_rank`, `ndcg_at_k`, `precision_at_k`, and `recall_at_k`.
- Pair datasets use the selected pair threshold.
- Retrieval datasets use the selected `k`.
- The seed is stored in the leaderboard manifest for reproducibility.

The ranked algorithm table is sorted by task, metric, rank, and algorithm name. The exported ZIP contains the same CSV, JSON, HTML, and reproducibility files as the Python API.

## Python

```python
from matheel import available_leaderboard_algorithm_presets
from matheel.leaderboard import load_leaderboard_manifest, run_leaderboard
from matheel.reports import write_benchmark_report

print(available_leaderboard_algorithm_presets())

manifest = load_leaderboard_manifest("leaderboard.json")
report, artifacts = run_leaderboard(
    manifest,
    output_dir="leaderboard_artifacts",
)

write_benchmark_report(report, "leaderboard_artifacts/leaderboard_report.html")

print(report["aggregate"].sort_values(["task_family", "metric", "rank", "algorithm_name"]))
print(artifacts["json"])
```
