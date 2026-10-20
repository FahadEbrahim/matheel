# Local detector diagnostic: full prepared IRPlag Java views

This is a **single-source Java diagnostic**, not the official multilingual Matheel-Plagiarism-v1 leaderboard. It measures the optional local adapters on the existing full prepared IRPlag views. No test-label tuning, sampling, new labels or dataset changes were made. The original dataset includes non-plagiarized submissions whose labels are disputed in its README; those supplied labels remain unchanged.

| Local configuration | Pair AP | Pair AUROC | F1 at 0.5 | Retrieval MAP/MRR | Pair seconds | Retrieval seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dolos 2.10.0 | 0.8782 | 0.6702 | 0.7154 | 0.9564 | 4.80 | 4.73 |
| JPlag 6.3.0 average, min_tokens=9 | 0.8161 | 0.5559 | 0.5709 | 0.7071 | 30.89 | 30.44 |

Both methods cover all 460 labeled pairs and 355 positive-judged queries against all seven originals: 2,485 query/document candidates. All 467 prepared source files belong to each native detector corpus, including 105 submissions that are not retrieval queries because they lack positive qrels. Each preparation compared 108,811 unordered native pairs. Only the 460/2,485 required Matheel scores are published. Missing native pairs were never replaced with zero. Pair threshold 0.5 was fixed; runtime is per complete corpus preparation/evaluation, not a per-pair latency. Total diagnostic wall time was 71.76 seconds on macOS ARM64.

JPlag uses its pinned release JAR, Java 25, AVG similarity, 9 matching tokens, no normalization/base code/frequency weighting/merging, no report censoring, and no network version check. Dolos uses its pinned local npm package/API with Node 22.23.3, kgrams 23/window 17, comments excluded, no fingerprint-frequency filtering for this corpus, and no report censoring. Resource limits are 300 seconds, 1,024 MiB heap, 500 files and 125,000 native unordered pairs. Exact effective configurations, adapter/runtime/installed-artifact hashes and the Matheel Python-source hash manifest are in [results.json](results.json).

The immutable source revision is [`f07e95120680b25f78659ed63cd6130fd05b2df8`](https://github.com/oscarkarnalim/sourcecodeplagiarismdataset/tree/f07e95120680b25f78659ed63cd6130fd05b2df8). Source archive SHA256: `288efbe08cc60837ae6509e49fd0285e7d22cb9b6e4dd6740d86229969fe9b8d`.

Normalized input fingerprints, verified unchanged before and after the run:

- Pair: `0460f56a3a07245b2598ac10db8374b610a1d54d9b7912473a53cbb8f6a6e889` (467 files, 460 pairs).
- Retrieval: `73d8d190a278898d18647419d8db135d943cbaa6ff41813f91db01600b7c5fea` (467 files, 355 queries, 7 candidates, 355 qrels).

The JPlag JAR SHA256 is `5f2c21e8b88ed77134effcb3a5a3ab13d188f6a3e16d401387f7479e92db9aa2`. The Dolos npm lockfile SHA256 is `29f32c0af8eb8e824a9d22b5fd1c0335a8545ba275c76920e675a59a24d804bc`; actual installed code/native-library SHA256 is `20c136bd5f23d9b0f901f7a08b52202571bfc5a06f034ec7d080a94cd252e89d`. Native builds may differ across platforms; those differences are retained in method identity.

Public artifacts contain IDs, supplied labels/relevance and numeric scores; they contain no source-code text, private local paths or native reports. [manifest.json](manifest.json) hashes every data artifact; [scores/index.json](scores/index.json) describes each complete Matheel score CSV. [aggregate.csv](aggregate.csv) and [per_dataset.csv](per_dataset.csv) preserve full-precision metrics. This narrow diagnostic supports local integration feasibility; it does not establish relative quality across other languages, datasets, code-clone tasks or parameter settings.

To reproduce, prepare the existing IRPlag pair/retrieval views from the pinned source using `scripts/prepare_plagiarism_benchmark.py`, install the separately pinned tools described in [the adapter guide](../../../docs/external_detectors.md), and run a custom leaderboard with the two configurations recorded in `results.json`. Supply your local tool paths privately, use fixed threshold 0.5 and retrieval k=10, enable strict coverage, and use `score_artifact_dir` to retain complete required scores. `results.json` records portable artifact identity rather than machine-specific executable locations.
