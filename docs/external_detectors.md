# Local plagiarism detectors

Matheel can evaluate **JPlag 6.3.0** and **Dolos 2.10.0** through its custom algorithm interface. Both run locally, once per corpus, using explicit tool paths. Importing Matheel does not install tools, start services, or upload source code. These adapters add no Python dependencies.

These are optional detector integrations. The original [Matheel-Plagiarism-v1 results](plagiarism_results.md) retain their 29 configurations. Separate [0.7.1 supported-language comparisons](plagiarism_release_071.md) evaluate the detectors and existing lexical references on identical bounded scopes.

## Scope and score meaning

Each normalized `files.csv` entry is exported as one single-file submission under an opaque name. The corpus consists of **all prepared files**, including unjudged files. Labels, qrels, original file names and original directory paths are not passed to either tool. Keep `preprocess_mode="none"` for a native detector baseline; changing preprocessing defines a different method.

The language must be explicit and the corpus must use that language. Mixed-language and multi-file project submissions are outside this adapter contract. Java is covered by real local integration tests. Other exposed language identifiers use the same pinned tool interfaces and must pass parser/coverage checks on your corpus:

| Adapter | Language identifiers |
| --- | --- |
| JPlag | `java`, `python3`, `cpp`, `c`, `javascript`, `typescript`, `csharp`, `go`, `rust` |
| Dolos | `java`, `python`, `cpp`, `c`, `javascript`, `typescript` |

JPlag `score_mode="average"` uses the tool's `AVG` score. `maximum` uses `MAX`. `query` uses `firstSimilarity` when the requested left/query file is first in the report and `secondSimilarity` in the reverse direction. This preserves the tool's directional coverage rather than averaging it. Dolos uses its symmetric selected-fingerprint coverage score. Scores are in `[0, 1]`; they do not establish plagiarism by themselves.

Self comparisons are unsupported. Set `exclude_self_matches=True` for retrieval when queries also occur in the candidate corpus. Every other query/document candidate is scored, including unjudged candidates; qrels are used only by Matheel's evaluator.

Both pinned tools produce all unordered corpus pairs. The adapters validate the complete matrix before exposing any score. They reject omitted/duplicate/unknown pairs, rejected submissions, empty inputs, parser failures and inputs with no Dolos fingerprints. **An omitted pair never becomes a zero.** A zero is accepted only when the tool explicitly reports it. Dolos syntax-recovery errors are rejected using its own Tree-sitter parser's `hasError` flag.

## Install tools separately

Use trusted local runtimes. The adapters require Java 25 for JPlag and Node 22+ for Dolos. Tools and source processing stay outside Matheel's Python dependency installation. The tools retain their own licenses: [JPlag GPLv3](https://github.com/jplag/JPlag/blob/v6.3.0/LICENSE) and [Dolos MIT](https://github.com/dodona-edu/dolos).

Download the [official JPlag 6.3.0 standalone JAR](https://github.com/jplag/JPlag/releases/tag/v6.3.0). The required asset is `jplag-6.3.0-jar-with-dependencies.jar`, SHA256:

```text
5f2c21e8b88ed77134effcb3a5a3ab13d188f6a3e16d401387f7479e92db9aa2
```

Every JPlag run verifies this exact hash. The adapter disables JPlag's remote version check, runs in `RUN` mode, disables clustering, uses UTF-8, and explicitly requests all comparisons at threshold zero. No report viewer is launched.

For Dolos, copy `examples/external_detectors/package.json` and `package-lock.json` into a separate local directory, then install with your Node 22+ toolchain:

```bash
mkdir -p /tmp/matheel-dolos
cp examples/external_detectors/package*.json /tmp/matheel-dolos/
cd /tmp/matheel-dolos
npm ci --no-audit --no-fund
```

This is a local prefix, not a global installation. Dolos's Tree-sitter dependencies compile native bindings and may need your platform's C/C++ build tools and Python. The checked-in lockfile pins the complete npm dependency graph. Its SHA256 is `29f32c0af8eb8e824a9d22b5fd1c0335a8545ba275c76920e675a59a24d804bc`. The adapter verifies the upstream `@dodona/dolos@2.10.0` npm integrity entry and records the actual installed JavaScript/native-binary tree hash, lockfile hash and Node binary hash. It invokes the pinned local JavaScript API in a bounded Node child process; no web UI or server starts.

By default Dolos uses `kgram_length=23`, `kgrams_in_window=17`, `include_comments=False` and `max_fingerprint_count=2147483647`. The explicit high fingerprint cap keeps all corpus fingerprints for ordinary bounded runs. Lowering it enables corpus-dependent filtering and changes the method identity. The adapter does not use report limits or similarity censoring.

## Python evaluation

```python
from matheel.evaluation import score_pair_dataset, score_retrieval_dataset

jplag_options = {
    "language": "java",
    "jar_path": "/tools/jplag-6.3.0-jar-with-dependencies.jar",
    "java_path": "/tools/jdk-25/bin/java",
    "min_tokens": 9,
    "score_mode": "average",
}
scored = score_pair_dataset(
    "data/java-pairs",
    algorithm="matheel.external_detectors.jplag",
    algorithm_options=jplag_options,
)

retrieval = score_retrieval_dataset(
    "data/java-retrieval",
    algorithm="matheel.external_detectors.dolos",
    algorithm_options={
        "language": "java",
        "dolos_path": "/tmp/matheel-dolos",
        "node_path": "/tools/node-22/bin/node",
    },
    exclude_self_matches=True,
)
print(scored.attrs["algorithm"])  # portable verified tool identity
```

A prepared corpus is shared by every pair in that evaluation. Calling `score_pair` directly without the prepared context is rejected. Changing pair labels does not alter source processing or detector input.

## CLI and leaderboards

The `--algorithm` alias and existing `--algorithm-path` accept an importable module name or an existing Python file:

```bash
matheel evaluate-pairs data/java-pairs \
  --algorithm matheel.external_detectors.jplag \
  --algorithm-option language=java \
  --algorithm-option jar_path=/tools/jplag-6.3.0-jar-with-dependencies.jar \
  --algorithm-option java_path=/tools/jdk-25/bin/java \
  --scores-out results/jplag-scores.csv \
  --metrics-out results/jplag-metrics.json \
  --reproducibility-out results/jplag-reproducibility.json
```

For a custom leaderboard, use the same module in `algorithm_path` and options in `algorithm_options`. A retrieval dataset entry can set `exclude_self_matches: true`. Keep machine paths in your private runtime manifest; exported artifacts replace them with verified tool identity. Existing [frozen benchmark protocols](benchmark_protocol.md) still enforce their source selection, candidate coverage and threshold policies. Language-specific detector experiments need an explicit appropriate corpus and should not silently reinterpret an existing frozen protocol.

## Reproducibility and failures

`portable_configuration(algorithm_options)` verifies the actual local artifacts and returns their versions, SHA256 fingerprints, effective options, corpus policy and adapter-code fingerprint. It is used for method IDs, algorithm cards, result metadata and exported leaderboard manifests. Identical installations copied to a different directory retain the same method ID; changed tool bytes, runtime, preprocessing or detector configuration change it. Published identities can be compared without reinstalling the tools; executing them still requires explicit local runtime paths.

To lock a verified Dolos installation for a subsequent run, set `expected_installation_sha256` to the `installation_sha256` value returned by its portable configuration. A mismatch fails before source execution. Build output object files and generated Makefiles are excluded from the installed-tree hash; executable `.node` parser libraries are included.

The default per-corpus limits are 300 seconds, 1,024 MiB runtime heap, 500 files, 50 MB UTF-8 source, 125,000 unordered comparisons, and 2 MB process log. Results are limited to 250 MB for JPlag and 100 MB for Dolos. These are resource limits, not quality thresholds. Explicit `timeout_seconds`, `heap_mb`, `max_files`, `max_source_bytes`, `max_comparisons`, `max_output_bytes` and `max_result_bytes` options can change them; all enter the method identity. A corpus can require much more runtime/memory than its file count suggests. The heap limit controls the runtime heap rather than total process memory.

Commands use argument arrays without a shell, isolated temporary working directories and bounded output. On POSIX, timeout/error cleanup kills the process group; on Windows the direct process is killed. Temporary sources and reports are deleted after parsing. Failures raise `DetectorError`/`DetectorTimeoutError`; published error messages omit tool logs, private paths and source content. A leaderboard with `continue_on_error=True` records missing metrics, and strict coverage keeps the failed method unranked.

## Verified local smoke

The standalone [synthetic example](https://github.com/FahadEbrahim/matheel/blob/main/examples/external_detectors/smoke.py) runs both pair and retrieval scoring without downloading data:

```bash
python examples/external_detectors/smoke.py --detector jplag \
  --jar-path /tools/jplag-6.3.0-jar-with-dependencies.jar \
  --java-path /tools/jdk-25/bin/java
python examples/external_detectors/smoke.py --detector dolos \
  --dolos-path /tmp/matheel-dolos --node-path /tools/node-22/bin/node
```

Real local verification used JPlag 6.3.0 with Temurin 25.0.4.1 and Dolos 2.10.0 with Node 22.23.3 on macOS ARM64. Three synthetic Java files produced all three pair comparisons, explicit identical/nonmatching scores, and all four requested self-excluded retrieval candidates. Invalid Java was rejected by both adapters. These checks verify execution, report parsing and candidate mapping; they are not effectiveness measurements.

To repeat the optional integration tests, set `MATHEEL_JPLAG_JAR`, `MATHEEL_JAVA`, `MATHEEL_DOLOS_PREFIX` and `MATHEEL_NODE`, then run:

```bash
python -m pytest -o addopts='' -m integration tests/test_external_detectors_integration.py
```

Ordinary unit tests use offline fixtures. Integration tests skip when local tool paths are absent and never install or download tools.


## Existing IRPlag Java diagnostic

A separate [source-free diagnostic bundle](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/external-detectors/irplag-java-20261020)
evaluates the full existing prepared IRPlag views: 460 labeled pairs and 355
positive-judged queries against seven originals (2,485 candidates). Both local
adapters completed every required score. With the recorded untuned configurations,
Dolos scored pair AP 0.8782 and retrieval MAP 0.9564; JPlag average/min_tokens=9
scored pair AP 0.8161 and retrieval MAP 0.7071. Pair threshold 0.5 was fixed.
The bundle records full scores, immutable source/content/tool/runtime hashes and
per-corpus timings. This common Java scope is separate from the official
multilingual leaderboard and does not establish results on other datasets.
