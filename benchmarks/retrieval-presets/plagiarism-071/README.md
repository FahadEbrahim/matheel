# Retrieval presets on the existing plagiarism views

This separate diagnostic compares seven existing configurations on all eleven
prepared retrieval views. The plan fixes N=1/5/10, k=10 and a 45-second
configuration/view deadline before scoring. It preserves every query, corpus
document and qrel from those views, including the existing self-exclusion rules.
It does not change the frozen plagiarism-v1 artifacts or any labels.

Run from a checkout containing the plan's exact source commit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validate_retrieval_presets.py \
  --manifest prepared/prepared-manifest.json \
  --plan benchmarks/retrieval-presets/plagiarism-071/plan.json \
  --source-root ../matheel-0.7.1 --release-wheel dist/matheel-0.7.1-py3-none-any.whl \
  --output-dir benchmark_outputs/retrieval-071
```

Use `git worktree add --detach ../matheel-0.7.1 v0.7.1` for the exact release
checkout and `python -m pip download --no-deps matheel==0.7.1 --dest dist` for the
published wheel. The runner recursively verifies that all wheel Python modules
match the execution checkout. Use the existing versioned preparation recipe to obtain the local normalized
inputs; this runner downloads nothing. Public artifacts contain source IDs,
hashes, configurations and numerical evidence. `provenance.json` binds the runner,
plan, package Python files, manifest, normalized tables and source inputs.
`results.json` preserves complete/timeout/failed rows, while `summary.csv` is the
compact per-view comparison. Each completed configuration has its selected-score
CSV and metrics; two-stage results include per-query candidate recall. The
exhaustive control also includes source-free error analysis and 100-round grouped
bootstrap intervals. Original source code and local source locators are omitted.

`scoring_seconds` covers scoring plus metrics; `wall_seconds` additionally includes
process startup, validation and reporting. Every timed cell is a single
observation on the recorded machine. Timeouts have null quality metrics. There is
no partial-scope aggregate ranking or quality-dependent method selection.

See [practical preset guidance](../../../docs/scalable_retrieval.md) for the
candidate-recall ceiling, fixed top-ten denominator and judged-pool limitations.
