# Benchmark workflow recipes

## Reproduce published results

Install the repository in an isolated environment, then follow the [frozen
benchmark recipe](benchmark_protocol.md). Source archives, models, runtime
constraints and audited scored rows are documented with the [published
results](plagiarism_results.md). Keep local inputs and outputs outside the checkout.

Validate the result JSON and score CSVs before publishing. Preserve the original
protocol and result bundle; declare new settings or candidate policies as new
configurations or experiments rather than rewriting a previous release.

## Compare a new detector

Start with a [custom algorithm](contributing_algorithms.md) and a tiny local
fixture. Declare supported languages, score direction and scale, corpus scope,
timeout, tool version and configuration. Corpus tools should run once in
`prepare_dataset`, then return cached scores from `score_pair`.

Verify complete candidate coverage and preserve zero scores explicitly. Missing
output, unsupported syntax and process failures need distinct outcomes. A tool
with limited language support needs a frozen common language subset; its partial
coverage does not qualify it for the full multilingual ranking.

Run matching and nonmatching fixtures against the real executable, then test its
parser, identity metadata and failure handling independently. Keep tool
installation optional. See [versioned protocols](benchmark_protocol.md) for
configuration identity and artifact validation requirements.

## Select a threshold using validation data

Use the existing [grouped split and calibration workflow](datasets.md). Keep
related authors, assignments or source families in one partition, where those
identifiers are available. Fit a threshold on validation data, freeze it, and
evaluate a separate test partition. The protocol's `validation_f1` policy checks
exact content and declared group overlap.

```bash
matheel calibration-report validation_scores.csv --output-dir validation_calibration
```

Pass the selected threshold explicitly when evaluating held-out test data.
An F1 calculated on labels used to select the threshold is a calibration
diagnostic, not held-out performance. Public v1 F1 retains its fixed threshold.

## Investigate disagreements

Use [result analysis](result_analysis.md) to align score CSVs, inspect errors and
calculate paired grouped uncertainty. Inspect selected source regions with
`explain-pair`. Record failures and unsupported task views alongside quality
metrics so skipping difficult inputs cannot improve an aggregate.

## Check a pull request

```bash
python -m pytest
python -m ruff check .
python -m mkdocs build --strict
python scripts/check_benchmark_regressions.py --output /tmp/benchmark-regression.json
git diff --check
```

The regression fixture contains synthetic code and six existing lexical
configurations. It enforces score parity to `1e-12` absolute/relative tolerance
and records wall times, Python/platform and package versions. CI preserves the
report. Timing changes are informational because shared runners vary; compare
repeated measurements on equivalent environments.

For a deliberate scoring change, inspect differences and use
`--update-baseline` explicitly. Commit the reviewed fixture changes with the
implementation and describe their effect. Baseline regeneration must not hide an
unexpected change. This fixture is separate from the frozen research benchmark.
