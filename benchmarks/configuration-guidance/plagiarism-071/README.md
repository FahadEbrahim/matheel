# Existing-configuration diagnostics

This source-free experiment uses the existing Student Code Similarity Python
pair view. Its predeclared plan fixes the four lexical configurations, threshold,
bootstrap seed/rounds, byte-based input selection, pinned MiniLM snapshot and
120-second deadline for each workflow.

1. All 293 original pair observations are scored with raw/basic Levenshtein, raw
   LCS and raw Winnowing. Paired error analysis uses raw Levenshtein as the
   reference on the same candidates and source labels. Source-sharing connected
   components supply bootstrap groups; the threshold stays fixed at 0.5.
2. The existing six controlled robustness fixtures and AST-checked
   formatting/comment variants of a declared source are scored. No supplied
   submission is executed and no new plagiarism labels are created.
3. One Python pair, selected by the recorded label-blind byte rule, is scored by
   pinned cached MiniLM with raw/mean/max/first/capped chunk settings and full
   Winnowing. Real tokenizer offsets measure actual retained input.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/compare_existing_configurations.py \
  --manifest prepared/prepared-manifest.json \
  --plan benchmarks/configuration-guidance/plagiarism-071/plan.json \
  --source-root ../matheel-0.7.1 --release-wheel dist/matheel-0.7.1-py3-none-any.whl \
  --cpu-lock /private/tmp/matheel-071-benchmark-cpu.lock \
  --output-dir benchmark_outputs/configuration-guidance-071
```

Use the exact `v0.7.1` source checkout and published wheel described in the
[retrieval replay instructions](../../retrieval-presets/plagiarism-071/README.md).
The existing `sentence_transformers` and `chunking` extras and the exact model
snapshot in `plan.json` must already be installed/cached. No datasets or models
are downloaded. Use a new output directory for each run.

`provenance.json` binds released package code, runner/plan/manifest and environment.
The pair workflow exports full source-free scored rows, errors and paired metric
intervals. Robustness outputs retain source hashes, transformation validation and
score deltas without source strings. Long-file outputs retain input hashes,
source IDs/row order, model-file hashes, measured tokenizer/chunk coverage and
scores. Failures and timeouts remain explicit, with no invented zero scores.

This supports practical inspection of the declared configurations and inputs.
The threshold is diagnostic, not a fitted operating point; controlled fixtures
and one long pair do not establish population-level performance. Existing labels
and the official benchmark protocol are preserved.

## Recorded comparisons

Released source `64ca1364a08f76904559e19504524f80671b0479` matched all 46
Python modules in the actual published 0.7.1 wheel. Plan commit `92cb4a4` and
runner commit `aab90b7` precede measurement. All three workflows completed
(pairs 1.20 s, robustness 0.63 s, long files 5.49 s), with every declared scorer
row successful. These wall times include setup and reporting. Full environment,
model-file hashes and exclusive-lock timestamps are in the artifacts.
`artifact-manifest.json` binds all 34 generated public files by size/SHA-256.

### Existing Python pair observations

All 293 original observations and labels were retained. The point values below
use the predeclared threshold 0.5 and 40 source-ID connected-component bootstrap
groups. The paired 500-round intervals, disagreement rows and length slices are
in `pairs/`; no threshold was fitted and no significance claim follows from
these point values.

| Configuration | Precision | Recall | F1 at 0.5 | Average precision | AUROC |
| --- | --- | --- | --- | --- | --- |
| raw_levenshtein | 0.4460 | 0.9500 | 0.6070 | 0.5600 | 0.7141 |
| basic_levenshtein | 0.4091 | 0.8100 | 0.5436 | 0.5695 | 0.6980 |
| raw_lcs | 0.3984 | 0.9800 | 0.5665 | 0.5467 | 0.7018 |
| raw_winnowing | 0.6000 | 0.1200 | 0.2000 | 0.4470 | 0.5407 |

Basic preprocessing slightly increased AP's point estimate while reducing F1 at
the unchanged threshold. Raw Winnowing's recall at that threshold was 0.12.
These observations show why a score threshold cannot be transferred between
representations without evaluating the intended operating point. They do not
establish a universal choice of preprocessing or detection method.

### Controlled and supplied-source robustness

The selected source is the left endpoint of row 102, ID
`right_8007c9aa2f279a9c`, SHA-256
`6cc2e782830d5e7d906cc9aa72a34cdc3aa16f2a9ad67d4e35795772a157ed74`.
Its original size is 2,390 UTF-8 bytes. Formatting and comments were AST-checked
without executing this submission. Every baseline identity score is 1.0.

| Configuration | Score after formatting | Score after comments |
| --- | --- | --- |
| raw_levenshtein | 0.9816 | 0.9759 |
| basic_levenshtein | 0.9854 | 1.0000 |
| raw_lcs | 0.9833 | 0.9759 |
| raw_winnowing | 0.5709 | 0.9698 |

For this source, basic Levenshtein was invariant to the added comments; raw
Winnowing fell to 0.570850 after AST-preserving formatting. Do not assume raw
fingerprints are formatting invariant. The six fixed built-in transformations
add 24 successful diagnostic comparisons in `robustness/result.json`; they do
not supply new plagiarism labels or population-level robustness estimates.

### Real-model input coverage

The same declared row's right endpoint has 2,167 UTF-8 bytes. Its two inputs
were selected by byte counts and original row order, without labels or scores.
The pinned MiniLM snapshot uses a 256-token limit; chunking uses 512 characters
and 64-character overlap. Percentages below measure the union of tokenizer-mapped
**prepared-byte spans**, excluding whitespace without token offsets. They are
not percentages of semantic importance or detection quality.

| Configuration | Similarity | Selected chunks left/right | Tokenizer-mapped prepared-byte spans left/right | Truncated chunks left/right |
| --- | --- | --- | --- | --- |
| raw_minilm | 0.9014 | 1 / 1 | 30.8% / 30.1% | 1 / 1 |
| chunked_mean | 0.8995 | 6 / 5 | 78.2% / 75.9% | 0 / 0 |
| chunked_max | 0.8986 | 6 / 5 | 78.2% / 75.9% | 0 / 0 |
| chunked_first | 0.9186 | 6 / 5 | 15.0% / 16.5% | 0 / 0 |
| chunked_mean_cap2 | 0.9202 | 2 / 2 | 29.0% / 31.3% | 0 / 0 |
| full_winnowing | 0.1161 | 1 / 1 | N/A: lexical full-source scoring | 0 / 0 |

Unlimited mean/max chunks cover all prepared-source bytes at the chunk-selection
level and have zero model-truncated chunks. Their lower tokenizer-mapped byte
fraction reflects token offsets that exclude whitespace; it does not mean the
remaining bytes were truncated. Raw input truncated both files. `first` encoded
six/five chunks but used only the first; the two-chunk cap deliberately omitted
input. The higher similarities of `first`/capped runs do not establish better
detection. Evaluate metrics on the intended workload after checking coverage.

Full-source Winnowing uses no transformer token limit. Its semantic-token
coverage is N/A because semantic scoring is inactive; the scorer uses the full
prepared text. Its score scale is not a calibrated substitute for MiniLM's.
