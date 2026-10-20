# Matheel Documentation

Matheel compares source code through a Python API, CLI, and Gradio app. Start with a lightweight lexical baseline, add embeddings or code-aware metrics when needed, then evaluate the configuration on labeled data.

## Getting Started

- [Install and run a first comparison](usage.md) with the base package and no model downloads.
- [Use the Gradio app](gradio.md) locally or in the hosted demo.
- [Choose similarity metrics](choosing_metrics.md) for whole files, partial matches, structural comparisons, or retrieval.

## Compare Code

| Task | Guide |
| --- | --- |
| Use edit distance, LCS, fuzzy matching, or token fingerprints | [Lexical metrics](lexical.md) |
| Select dense, static, or multi-vector embeddings | [Embeddings and routing](vectors.md) |
| Compare syntax, dataflow, or transformer token alignments | [Code-aware metrics](code_metrics.md) |
| Remove comments or normalize identifiers | [Preprocessing](preprocessing.md) |
| Understand what each scorer sees | [Tokenization and limits](tokenization.md) |
| Split long files before embedding | [Chunking](chunking.md) |
| Compare several configurations on one collection | [Comparison suites](comparison_suite.md) |

## Evaluate and Inspect Results

| Task | Guide |
| --- | --- |
| Load pair/retrieval data and measure quality | [Datasets and evaluation](datasets.md) |
| Bound expensive retrieval and audit candidate recall | [Exact scoring and scalable retrieval](scalable_retrieval.md) |
| Interpret score ranges and select thresholds | [Scoring and calibration](scoring.md) |
| Run a small auditable experiment | [Reproducible benchmark](reproducible_benchmark.md) |
| Freeze comparable plagiarism evaluations | [Versioned benchmark protocols](benchmark_protocol.md) |
| Inspect the declared public plagiarism benchmark | [Plagiarism benchmark results](plagiarism_results.md) |
| Rank methods and track exported runs | [Leaderboards and run registry](leaderboard.md) |
| Explore dataset maps and matched code regions | [Visualization and pair explanations](visualization.md) |

## Extend and Contribute

- [Custom algorithms](customization.md): prototype a scorer without changing Matheel internals.
- [Contribute algorithms](contributing_algorithms.md) or [datasets](contributing_datasets.md): add tested package support.
- [Development](development.md): install contributor tools, run tests, build docs, and check packages.

## Examples and Demos

Use the [examples catalog](https://github.com/FahadEbrahim/matheel/tree/main/examples) to find runnable scripts and the [notebook catalog](https://github.com/FahadEbrahim/matheel/tree/main/examples/notebooks) for Colab workflows. The [hosted Gradio demo](https://huggingface.co/spaces/buelfhood/matheel-framework) provides an interactive starting point.
