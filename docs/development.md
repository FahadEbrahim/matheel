# Development

This page covers local development checks for contributors and release preparation.

## Editable Install

Install Matheel in editable mode with the development tools:

```bash
python -m pip install -e ".[dev]"
```

## Local Checks

Run the default offline-friendly test suite:

```bash
python -m pytest
```

Run the Ruff lint check:

```bash
python -m ruff check .
```

Real integration tests are opt-in because they need optional backends, model weights, or local detector runtimes:

```bash
python -m pytest -o addopts='' -m integration
```

Install the full optional stack before running them locally:

```bash
python -m pip install -e ".[all,dev]"
```

The `Real Integration Tests` GitHub Actions workflow runs every Monday at 05:00 UTC on Python 3.12. Manual runs can select any supported Python version and either `all`, `models`, or `detectors`. The jobs run independently on Ubuntu 24.04, so model downloads do not delay detector checks. Normal pull-request CI remains offline.

The model job installs `matheel[semantic,chunking,dev]` and CPU PyTorch. It first downloads three public models at immutable revisions listed in [`prepare_real_models.py`](https://github.com/FahadEbrahim/matheel/blob/main/scripts/ci/prepare_real_models.py), reusing `~/.cache/huggingface`. Tests then use those local snapshot paths with Hugging Face offline mode enabled. This covers the dense, static and multivector backends plus the real MiniLM long-file test. A missing model or backend fails CI rather than silently producing a skipped green run.

The detector job installs only `matheel[dev]` and provisions JPlag 6.3.0, Temurin 25.0.4.1+1, Node 22.23.3 and Dolos 2.10.0 under the runner's temporary directory. [`setup_external_detectors.sh`](https://github.com/FahadEbrahim/matheel/blob/main/scripts/ci/setup_external_detectors.sh) checks pinned SHA256 values before extracting or executing downloads. Dolos uses the checked-in npm lockfile and its integrity entries; native parsers build from source with the pinned Node headers. Java and Node are not installed globally. Adapter calls retain their wall-time, heap, input, comparison, log and result limits described in [External detectors](external_detectors.md#reproducibility-and-failures).

The model job has a 45-minute ceiling and the detector job a 25-minute ceiling, with shorter install, download and test-step limits. Each publishes JUnit results and source-free model/detector identities as artifacts for 14 days, including the installed Dolos/native-parser tree and runtime hashes. Exactly one detector skip is expected: Dolos has no directional JPlag score mode. Any other skip fails the job.

To run the pinned detector setup on a Linux x86_64 development machine with Python, npm native build tools, curl and tar available:

```bash
python -m pip install -e ".[dev]"
bash scripts/ci/setup_external_detectors.sh /tmp/matheel-ci-detectors /tmp/matheel-ci-detectors.env
set -a
. /tmp/matheel-ci-detectors.env
set +a
python -m pytest -o addopts='' -m integration tests/test_external_detectors_integration.py
```

Use a new destination directory for each setup. The script does not remove existing installations. The setup performs network downloads; detector execution itself stays local and does not upload source code.

Pull-request CI also runs selected compatibility checks against the external `codebleu` package and builds the wheel and source distribution. The package job validates metadata, installs the wheel into a clean environment, imports Matheel, and exercises the CLI help command.

## Documentation Checks

Install the docs dependencies separately from the package development extra, then build with the same strict checks as CI:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
```

Use `python -m mkdocs serve` for a local preview. Navigation is grouped by user task in `mkdocs.yml`; the [documentation index](index.md) provides the same entry points. Keep page filenames stable when changing categories, and update links and examples when changing public options.

## Gradio Checks

The default development install does not include Gradio, so Gradio-specific tests skip when its optional dependencies are unavailable. Install the Gradio and development extras to run the same focused checks as pull-request CI:

```bash
python -m pip install -e ".[dev,gradio]"
python -m pytest tests/test_gradio_app.py tests/test_gradio_html_utils.py
```

These tests cover workflow helpers, the stable tab and primary-action structure, and a live server smoke check of the root page and `/config` endpoint. They intentionally avoid model downloads.

The real-browser workflows use Chromium and deterministic local ZIP fixtures. They exercise the lexical pair and collection paths, dataset validation and evaluation, artifact downloads, and leaderboard inspection without model or network access. Install the browser extra and Chromium, then override the default marker filter explicitly:

```bash
python -m pip install -e ".[dev,gradio,browser]"
python -m playwright install chromium
export MATHEEL_BROWSER_ARTIFACTS_DIR=test-results/browser
python -m pytest -o addopts='' -m browser tests/browser --collect-only -q
python -m pytest -o addopts='' -m browser tests/browser \
  --browser chromium \
  --tracing retain-on-failure \
  --video retain-on-failure \
  --screenshot only-on-failure \
  --output test-results/browser
```

The collection guard prevents an accidentally filtered zero-test run. When a browser workflow fails, screenshots, traces, video, browser console events, failed requests, and the Gradio server log are retained under `test-results/browser`.

## Package Checks

When preparing release or packaging changes, build the package and check the distribution metadata:

```bash
python -m build
python -m twine check dist/*
```

## Deferred Dependency Migrations

Some upper bounds are deliberate: [Gradio 6 support](https://github.com/FahadEbrahim/matheel/issues/284) requires replacing or upgrading the model-search component, and the [parser/Chonkie migration](https://github.com/FahadEbrahim/matheel/issues/285) changes parser loading and first-use downloads. Keep these migrations separate from routine compatible dependency upgrades.

## Cache Safety

Matheel uses module-level caches for Hugging Face model metadata, detected tokenizer limits, CodeBLEU keyword sets, tree-sitter parsers, and CodeBERTScore scorer objects.

Request-path cache writes are protected with locks so concurrent API or Gradio calls do not mutate the same cache at the same time. Keyword cache entries are stored as immutable values. Pair-level caches inside prepared run contexts are scoped to that run context and should not be reused across independent requests.
