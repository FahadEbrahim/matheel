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
