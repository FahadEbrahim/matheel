# Reproducible Benchmark Demo

This demo creates a tiny synthetic pair-classification dataset, adapts it through the dataset manifest workflow, scores it with an offline lexical baseline, and writes auditable outputs.

It is a workflow example, not a benchmark claim. Real datasets are not bundled with Matheel, and users should provide or download datasets according to each source's terms.

## Run the Demo

From the repository root:

```bash
python examples/evaluation/reproducible_benchmark_demo.py \
  --output-dir benchmark_outputs/synthetic_pair_benchmark \
  --overwrite
```

The command writes:

```text
benchmark_outputs/synthetic_pair_benchmark/
  raw/pairs.csv
  dataset_manifest.json
  benchmark_config.json
  dataset/
  results/
    scored_pairs.csv
    pair_metrics.json
    resample_metrics.csv
    resample_summary.csv
    reproducibility.json
```

The generated files capture:

- synthetic raw inputs and the normalized Matheel dataset
- source, adapter, and destination choices in `dataset_manifest.json`
- threshold, feature weights, preprocessing, language, and resampling seed in `benchmark_config.json`
- scored rows and aggregate pair-classification metrics
- fold-level metrics and interval summaries
- package versions, Python metadata, platform metadata, source fingerprint, and run metadata

## CLI Equivalent

After generating the synthetic inputs, you can rerun the scoring step through the CLI:

```bash
matheel evaluate-pairs \
  --manifest benchmark_outputs/synthetic_pair_benchmark/dataset_manifest.json \
  --feature-weight levenshtein=1.0 \
  --preprocess-mode basic \
  --code-language python \
  --threshold 0.65 \
  --scores-out benchmark_outputs/synthetic_pair_benchmark/results/scored_pairs_cli.csv \
  --metrics-out benchmark_outputs/synthetic_pair_benchmark/results/pair_metrics_cli.json \
  --reproducibility-out benchmark_outputs/synthetic_pair_benchmark/results/reproducibility_cli.json
```

The example uses `levenshtein=1.0`, so it runs offline and does not download embedding models.

## Adapt the Workflow

For a custom pair dataset:

1. Replace `raw/pairs.csv` with your own table.
2. Update `dataset_manifest.json` so `adapter_options` point to your column names.
3. Keep `benchmark_config.json` with the run settings you want to audit.
4. Keep a fixed resampling seed while comparing configurations.
5. Store generated outputs outside source control unless the files are deliberately tiny examples.

For retrieval datasets, use `auto_retrieval_tabular` and the ranking metrics described in [Datasets and evaluation](datasets.md).

## Reproducibility Checklist

- Record the Matheel version and optional backend package versions.
- Record the dataset source, adapter, and normalized source fingerprint.
- Record scorer settings, preprocessing, language, threshold, and feature weights.
- Record split method, number of folds or rounds, confidence level, and random seed.
- Keep scored rows alongside summary metrics so threshold decisions can be audited.
