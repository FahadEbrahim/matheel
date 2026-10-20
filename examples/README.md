# Matheel Examples

For a complete offline error-inspection workflow, run
`python examples/evaluation/result_analysis_demo.py`. It writes synthetic scored
pairs, a paired reference and HTML/CSV/JSON reports. See the
[result analysis guide](../docs/result_analysis.md) and
[benchmark workflow recipes](../docs/benchmark_workflows.md).

Run these examples from the repository root after installing the checkout:

```bash
python -m pip install -e .
python examples/basic/lexical_demo.py
```

The default script runs use local sample data and do not download models or datasets. Optional dependencies are listed below. The samples demonstrate the APIs; use representative datasets to compare detection quality.

## Choose a workflow

| Goal | Examples | Installation beyond the base package |
| --- | --- | --- |
| Prepare code | [Preprocessing](basic/preprocessing_demo.py), [chunking](basic/chunking_demo.py) | `.[chunking_code]` for the optional code-chunking step |
| Compare text and source files | [Lexical metrics](basic/lexical_demo.py), [archive ranking](basic/sample_pairs_demo.py), [comparison suites](evaluation/comparison_suite_demo.py) | None |
| Explore vectors | [Vector options and static hash baseline](basic/vectors_demo.py) | `.[semantic]` for the optional model flags |
| Compare code metrics | [CodeBLEU, CrystalBLEU, RUBY, TSED, CodeBERTScore](basic/code_metrics_demo.py) | `.[metrics]` for TSED and the optional CodeBERTScore step |
| Evaluate datasets | [Dataset adapters](evaluation/datasets_demo.py), [validation reports](evaluation/dataset_validation_demo.py), [reproducible benchmark](evaluation/reproducible_benchmark_demo.py) | None |
| Tune and compare results | [Threshold tuning](evaluation/threshold_tuning_demo.py), [calibration](evaluation/calibration_report_demo.py), [leaderboard](evaluation/leaderboard_demo.py) | None |
| Inspect results visually | [Dataset maps and pair explanations](evaluation/visualization_demo.py) | None for PCA; `.[visualization]` enables UMAP elsewhere |
| Add an algorithm | [Custom algorithm](custom/custom_algorithm_demo.py) | None |
| Work interactively | [Colab notebooks](notebooks/README.md), [Gradio app guide](../docs/gradio.md) | See each guide |

## Start with lexical similarity

The lexical demo lists every supported RapidFuzz distance and `fuzz` scorer, including LCS, then combines LCS, RapidFuzz ratio, winnowing, and GST. `score_lexical_metric` compares raw strings; `calculate_similarity` also supports preprocessing and weighted feature blends. Scores use a 0–1 scale.

```bash
python examples/basic/lexical_demo.py
python examples/basic/sample_pairs_demo.py
python examples/evaluation/comparison_suite_demo.py
```

Archive examples generate their sample archive automatically. To create an archive yourself and use the same features through the CLI:

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare sample_pairs.zip \
  --feature-weight lcs=0.5 \
  --feature-weight rapidfuzz_ratio=0.5
```

## Opt into semantic models

The vector demo starts with a local static hash baseline. These flags load trained models and may download their weights on the first run:

```bash
python -m pip install -e ".[semantic]"
python examples/basic/vectors_demo.py --semantic
python examples/basic/vectors_demo.py --multivector
```

CodeBERTScore has its own opt-in flag:

```bash
python -m pip install -e ".[metrics]"
python examples/basic/code_metrics_demo.py --codebertscore
```

## Keep generated reports

Report scripts retain their outputs under `benchmark_outputs/` and print paths you can open after the script finishes. Use `--output-dir` to choose a destination; rerunning the report scripts refreshes their generated files.

```bash
python examples/evaluation/visualization_demo.py
python examples/evaluation/dataset_validation_demo.py
python examples/evaluation/threshold_tuning_demo.py --output-dir benchmark_outputs/my_thresholds
python examples/evaluation/calibration_report_demo.py
python examples/evaluation/leaderboard_demo.py
python examples/evaluation/reproducible_benchmark_demo.py
```

Leaderboard and reproducible-benchmark examples require `--overwrite` to replace a nonempty output directory. Generated archives and benchmark outputs are ignored by git.

## Extend or launch Matheel

Generate a starter custom algorithm module:

```bash
matheel init-custom-algorithm my_algorithm.py
python examples/custom/custom_algorithm_demo.py
```

Launch the Gradio app from a repository checkout:

```bash
python -m pip install -e ".[gradio]"
python gradio_app/app.py
```

Use `.[all]` when you want every optional backend. The [Gradio notebook](notebooks/04_gradio_app.ipynb) includes a guided lexical pairwise check. Notebooks install the tagged release recorded in their setup cells; the scripts in this checkout also demonstrate features added since that release.
