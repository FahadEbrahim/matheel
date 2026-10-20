# Versioned plagiarism benchmarks

Matheel uses named, frozen benchmarks to compare methods on the same tasks and
publish their scores. The design follows [MTEB's separation of a benchmark from
its selected tasks and leaderboard results](https://docs.mteb.org/contributing/adding_a_benchmark/).
Matheel evaluates source-code similarity methods; its benchmark results are
independent of MTEB's text-embedding results.

`Matheel-Plagiarism-v1` is the first benchmark name. A **protocol ID** identifies
its precise dataset content and evaluation rules. A name alone never establishes
that two runs are comparable. The dataset category is **Plagiarism**
(`task_type`/`domain = plagiarism`); pair classification and retrieval are separate
evaluation tasks within that category. Later code-clone datasets will need their
own category, ground-truth definitions, protocols, and rankings.

## Freeze data before evaluating methods

First download and normalize datasets using the dataset adapters. Freeze explicit
local datasets rather than resolving a moving upstream branch during evaluation.
Each dataset needs an immutable source revision, a source URL, an explicit split,
and its normalized content. Only `code_language` may be supplied as a dataset-specific similarity option. Model, backend, weights, preprocessing, tokenizer, and chunking settings belong to each method configuration. Real semantic methods must declare an explicit vector backend so resolving a pinned local snapshot cannot change model routing.

The freezer hashes the normalized labels, queries,
corpus, relevance judgments, file metadata, and resolved source-code text. Machine
storage paths are excluded from the hash.

A minimal freeze manifest is:

```json
{
  "domain": "plagiarism",
  "seed": 7,
  "pair_metrics": ["average_precision", "auroc", "f1"],
  "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
  "split_policy": {
    "type": "provided_test",
    "description": "Evaluate the complete supplied test split without threshold tuning.",
    "group_fields": []
  },
  "threshold_policy": {"mode": "fixed", "value": 0.5},
  "datasets": [
    {
      "name": "my_plagiarism_pairs",
      "task": "pair",
      "path": "normalized/my_pairs",
      "source_url": "https://example.org/versioned-dataset",
      "source_revision": "REPLACE_WITH_IMMUTABLE_COMMIT_OR_VERSIONED_RECORD",
      "source_checksum": "REPLACE_WITH_ARCHIVE_CHECKSUM_IF_AVAILABLE",
      "split": "test",
      "selection": {"mode": "complete_split"},
      "similarity_options": {"code_language": "python"}
    }
  ]
}
```

The example placeholders describe required provenance; they are not a published
benchmark definition. Maintainers freeze actual downloaded revisions and record
any query selection or exclusions explicitly. A sampled or bounded pilot needs
its own name and selection description. It must not be presented as a complete
upstream corpus.

```bash
matheel benchmark freeze datasets.json \
  --benchmark-id Matheel-Plagiarism-v1 --output protocol.local.json
```

The local protocol includes paths needed to rerun evaluation. The protocol
embedded in exported results removes those paths. Keep the local file out of
published artifacts when it contains machine-specific paths. Copying a normalized
dataset to another root preserves its content and protocol identity. Changing
source text, labels, policies, selection, source revisions, or dataset-specific
scoring options changes the protocol identity. Evaluation verifies the content
fingerprints before scoring.

## Thresholds and splits

The default policy fixes the pair threshold at `0.5` before evaluation. This makes
F1, accuracy, precision, and recall diagnostic operating-point metrics. Average
precision and AUROC describe discrimination without selecting a test threshold.
An absent/undefined metric does not become a zero or disappear from an average.

For a benchmark with a legitimate validation split, use
`"threshold_policy": {"mode": "validation_f1"}` and give every pair dataset a
`validation_path` and `validation_split`. Matheel selects a separate threshold for
each method/dataset by validation F1 using its deterministic calibration rule,
then evaluates the frozen test data. Validation and test source content must be
disjoint. To enforce assignment/code-family separation too, set
`split_policy.type` to `group_disjoint` and provide the corresponding
`group_fields` from `files.csv`; missing or overlapping group values are rejected.
Without supplied group annotations, the protocol cannot establish assignment-level
independence and must not claim it. Test labels are never used to choose a
threshold. Methods cannot override protocol thresholds or retrieval `k`.

Retrieval evaluates every frozen query against every frozen corpus document.
Unjudged pairs have relevance zero. Score ties use ascending document IDs. MAP
and MRR use the full ranking; nDCG, precision, and recall use the dataset's frozen
`k` (default `10`). Query-selection rules, including treatment of queries without
positive judgments, belong in the frozen dataset's `selection` field. The runner
evaluates the supplied data as frozen and does not silently filter or sample it.

## Evaluate identified methods

A methods file is an `algorithms` list using the same built-in presets and custom
algorithm interface as custom leaderboards:

```json
{
  "algorithms": [
    "Levenshtein",
    "Longest Common Subsequence (LCS)",
    {
      "name": "MiniLM pinned",
      "feature_weights": {"semantic": 1.0},
      "vector_backend": "sentence_transformers",
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "model_revision": "REPLACE_WITH_40_CHARACTER_HUB_COMMIT_SHA"
    }
  ]
}
```

Every result records the full effective configuration and a SHA-256 method ID,
including scorer defaults, feature weights, preprocessing, tokenizer, chunking,
model/backend settings, package version, and the custom module's content hash.
Display names are not identities. A real embedding method requires an immutable
Hub model revision; the runner resolves that snapshot for execution and preserves
the canonical model name/revision in shared results. A `static_hash` result is a
hashing baseline, not a trained embedding model.

```bash
matheel benchmark evaluate protocol.local.json methods.json \
  --output-dir benchmark_artifacts --basename plagiarism_v1
matheel benchmark validate benchmark_artifacts/plagiarism_v1.json
```

Evaluation exports the usual leaderboard CSV, JSON, HTML, detail report, and
reproducibility snapshot. Each method/dataset run records elapsed wall-clock
seconds, including dataset loading and scorer preparation during the evaluation.
Model snapshot acquisition and validation-threshold fitting occur before this
scoring timer. Runtime depends on hardware, cache state, and concurrency; report
those conditions when making performance claims.

## Coverage and ranking

Scores aggregate with equal weight per dataset, separately for each category,
evaluation task, and metric. Pair F1 and retrieval MAP never share a ranking.
Every method must have a finite score on **every declared dataset** for that
ranking to be eligible. A failed method retains null-score rows and an error;
missing or nonfinite metrics make the affected aggregate unranked. The exported
`dataset_count`, `expected_dataset_count`, `coverage`, `complete_coverage`, and
`eligible` fields make the gate visible. Partial results are not silently averaged
into official rankings.

The result importer verifies protocol and dataset fingerprints, method
configuration hashes, required metric rows, runtime fields, aggregate values,
coverage eligibility, and ranks. It recomputes the aggregate from per-dataset
scores. Validation checks artifact consistency; it cannot prove that an external
submitter executed the claimed model or followed the stated experiment.

## Review and compare results

For a proposed public result:

1. Submit the frozen portable protocol, complete result bundle, source revisions,
   method configuration, reproducibility snapshot, and execution instructions in a
   pull request. Follow the dataset licenses; publish references rather than code
   or archives that cannot be redistributed.
2. Run `matheel benchmark validate` against the reviewed protocol. Confirm complete
   coverage, source provenance, split/selection rules, model revisions, and hardware
   details. Rerun the method or an agreed reproducible subset before accepting the
   submission's execution claims.
3. Compare only results with identical protocol IDs. A protocol change creates a
   separate benchmark version or explicitly labeled track.

The local registry validates versioned artifacts when importing and refuses
cross-protocol or versioned-versus-legacy comparisons. Legacy custom runs remain
supported, but differing dataset coverage, task metrics, or dataset fingerprints
cannot produce a comparison that implies equivalent evaluation.

```python
from matheel.benchmark_protocol import freeze_benchmark_protocol, run_benchmark_protocol

protocol = freeze_benchmark_protocol(manifest, benchmark_id="Matheel-Plagiarism-v1")
report, artifacts = run_benchmark_protocol(protocol, ["Levenshtein", "Longest Common Subsequence (LCS)"], output_dir="results")
print(report["aggregate"])
```

## Resource budgets and score audit artifacts

The frozen `execution_policy.max_method_seconds` sets a positive finite wall-clock
budget per complete method run (default `1800` seconds). It is part of protocol
identity. Official runs use a supervisor with an isolated process per method so
arbitrary scorers can be terminated when the budget expires. The in-process
`run_benchmark_protocol` API and direct CLI evaluation do not interrupt running
Python/native scorer code; callers enforcing the budget must supervise execution.
Resource limits describe feasibility under the declared run conditions and do not
imply that a timed-out method has poor detection quality.

A supervisor can preserve completed method checkpoints and create honest timeout
artifacts with the same required dataset/metric rows:

```python
from matheel.benchmark_protocol import failed_benchmark_result, merge_benchmark_results

failed, _ = failed_benchmark_result(
    protocol, [method_config], error="TimeoutError", elapsed_seconds=1800,
)
combined, artifacts = merge_benchmark_results(
    protocol, [completed_method_report, failed], output_dir="results",
)
```

Failed runs contain null scores and ranks, zero coverage, and `eligible = false`.
The total supervised wall time is recorded in `metadata.execution_failures`.
Uncompleted dataset rows use `elapsed_seconds = 0`, meaning no completed scoring
measurement; the method timeout is not repeated as if it were measured separately
on every dataset.

For auditable per-pair or query/document scores, pass
`score_artifact_dir="score_artifacts"` to `run_benchmark_protocol` or
`run_leaderboard`. Each already-computed score frame is written once to a CSV;
scoring is not repeated. The files contain IDs, labels/relevance, approved group
metadata, and scores, without source-code text or absolute paths. The result's
`metadata.score_artifacts` index records filenames, SHA-256 fingerprints, row
counts, columns, datasets, and complete method IDs. Keep licensed source data
separate from these result files. Validate a received score bundle with
`validate_benchmark_result(payload, score_artifact_dir="score_artifacts")`.
Score files are optional for local experiments; reviewed public results should
include them or explain any distribution restriction.
