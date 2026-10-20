# Matheel 0.7.1 benchmark editions

The 0.7.1 results use the existing plagiarism datasets and methods with the released scorer. They are separate from the immutable [original v1 publication](plagiarism_results.md). Select an edition under **Reports → Official Benchmark → Published Benchmark Edition**. Scores and ranks belong to the selected protocol; switching an edition never combines its rows with another protocol.

## Scope and runtime budgets

| Edition | Existing views | Methods | Method budget | Workers |
| --- | ---: | --- | ---: | ---: |
| All methods | 16 | Original 29 configurations | 180 seconds | 3 |
| Java detectors | 8 | JPlag, Dolos, Levenshtein, Winnowing | 600 seconds | 1 |
| C++ detectors | 6 | JPlag, Dolos, Levenshtein, Winnowing | 600 seconds | 1 |
| Python detectors | 1 | JPlag, Dolos, Levenshtein, Winnowing | 600 seconds | 1 |

The all-method edition retains every prepared row, source label, candidate pool, metric, fixed threshold (0.5) and retrieval cutoff (10) from v1. Its 180-second budget applies to a complete method across all 16 views. The original edition allowed 1,800 seconds. This deliberate resource change creates a different protocol ID, so changes in eligibility cannot be interpreted solely as scorer improvements. Concurrent workers each use one CPU thread; observed times include preparation and contend for hardware.

Detector tracks use the same complete prepared view for every compared method. All prepared files enter each native detector's corpus, including files that are not judged queries. A detector prepares the corpus once per view and must supply every required score. These are language-specific plagiarism comparisons, separate from the all-method multilingual ranking.

The common detector scope was declared before scoring: at most 500 files and 125,000 native unordered file comparisons. ConPlag's 1,822 files require 1,658,931 native comparisons and exceed these limits, so **the entire ConPlag view is excluded for all methods in the Java track**. ConPlag remains present in the all-method edition. No file was sampled or removed from a retained view. Native processes have a 120-second limit per corpus and a 1,024 MiB runtime heap limit; heap limits do not bound total process memory.

JPlag uses pinned 6.3.0, average similarity and 9 matching tokens. Dolos uses pinned 2.10.0, kgrams 23/window 17 and its symmetric fingerprint-coverage score. Both consume raw code and verify the installed tool/runtime hashes. The existing [adapter guide](external_detectors.md) explains their parser and single-file submission contracts. These settings were fixed without tuning on test labels.

## Missing results and uncertainty

A worker timeout produces null scores and an unranked method, with its timeout visible in the published result. A native parser rejection, missing pair, empty fingerprint set or other adapter error is a failed evaluation. It remains a failure even when another method can score the same view. No missing detector pair becomes a zero. No post-run error is relabeled as N/A.

The publication retains full-precision metrics and source-free scored-row indexes. Per-view uncertainty companions use the same grouped bootstrap as the result-analysis workflow: 500 rounds, seed 7, 95% intervals. Pair resampling preserves connected source groups; retrieval resamples complete queries. Unavailable intervals retain their explicit reason. These conditional intervals leave published ranks unchanged.

## Reproduction and artifacts

The [edition directory](https://github.com/FahadEbrahim/matheel/tree/main/gradio_app/benchmark_data/plagiarism-071) records each portable frozen protocol, complete method identities, audited score hashes, runtime metadata, confidence companion and publication manifest. `edition.json` binds the exact source commit, every Matheel Python module, release wheel SHA256, original protocol, predeclared scopes, limits and worker counts. `edition-manifest.json` hashes the complete public directory.

The [0.7.1 release assets](https://github.com/FahadEbrahim/matheel/releases/tag/v0.7.1) hold one scored-row ZIP per edition. They contain identifiers, supplied labels/relevance and scores, without submission text or private runtime paths. The four ZIPs are `all-methods-scores.zip`, `detectors-java-scores.zip`, `detectors-cpp-scores.zip` and `detectors-python-scores.zip`.

Prepare the pinned sources through the unchanged [v1 preparation recipe](benchmark_protocol.md), install the optional local detector runtimes, and use the exact released source checkout and wheel:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_release_benchmarks.py \
  --prepared-manifest /data/prepared/prepared-manifest.json \
  --output-dir /data/results-071 \
  --destination /data/public-071 \
  --expected-source-commit RELEASE_COMMIT \
  --release-wheel /data/matheel-0.7.1-py3-none-any.whl \
  --jar-path /tools/jplag-6.3.0-jar-with-dependencies.jar \
  --java-path /tools/jdk-25/bin/java \
  --dolos-path /tools/dolos \
  --node-path /tools/node-22/bin/node
```

The runner is a POSIX release-maintenance script and uses an exclusive file lock to serialize controlled benchmark measurements. It verifies that the prepared dataset protocol exactly matches v1 and that all scorer modules match the supplied wheel before scoring. Outputs belong outside the source checkout until validated for publication. Existing model snapshots must be cached at their declared revisions; the run does not download replacement models.
