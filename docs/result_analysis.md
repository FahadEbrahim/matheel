# Inspect errors and uncertainty

Use scored CSVs from dataset evaluation to inspect false positives, false
negatives, retrieval misses and disagreements between methods. This workflow
reuses scores; it does not rerun a model, fit a threshold, or change a published
leaderboard rank.

## Run an offline example

From an editable repository installation:

```bash
python examples/evaluation/result_analysis_demo.py --output-dir /tmp/matheel-analysis
matheel analyze-results /tmp/matheel-analysis/scores.csv \
  --reference /tmp/matheel-analysis/reference.csv \
  --group-column assignment --rounds 500 --output-dir /tmp/matheel-analysis/cli
```

The example uses synthetic diagnostic scores, not benchmark evidence.

Pair CSVs require `left_id`, `right_id`, `label` and `similarity_score`.
Retrieval CSVs require `query_id`, `document_id`, `relevance` and
`similarity_score`. Use the complete evaluated candidate universe. Optional
`language`, `assignment` and `length_bucket` columns produce slice tables when
present; they are not inferred from filenames or source content.

Pass `--dataset normalized_pairs` (or a normalized retrieval dataset) to add
language and file-length slices locally. Pair length is the larger endpoint's
character count; retrieval length is the query's character count. Buckets are
at most 1,000, at most 10,000, and over 10,000 characters. Languages use explicit
file metadata, recognized source suffixes, or a single dataset language. Unknown
languages remain explicit. Source text is read for these slices but is not
included in exported analysis artifacts. Retrieval slices always retain complete
queries and their candidate rows; inconsistent per-document slice labels are
skipped with a recorded note.

```bash
matheel analyze-results scored_pairs.csv --threshold 0.5 --output-dir analysis
matheel analyze-results retrieval_scores.csv --task retrieval --k 10 --output-dir retrieval_analysis
```

Outputs include JSON, an escaped HTML preview, and complete CSVs for metrics,
errors, slices and per-query metrics. HTML previews show up to 200 rows per table.
The Gradio **Reports → Analyze Results** tab offers the same workflow, with an
optional reference-method upload, table filtering and downloads. Use the CLI
for files larger than the interface's 50 MB or 100,000-row input limits.

## Inspect individual errors

Pair rows have `outcome` values `false_positive`, `false_negative`,
`true_positive` or `true_negative`, using the supplied threshold. Retrieval rows
record rank and distinguish relevant submissions inside and below top-k.
Ground truth comes from the supplied score file; analysis cannot recover
candidates or relevance judgments the evaluation did not record.

A reference comparison requires exactly the same candidates and labels. Pair
observations repeated in a source dataset retain their original weight and
occurrence order. Different candidate coverage is not a paired comparison.

The `score_row_index` points to the original CSV row, even when the inspection
table is sorted. Use it with existing pair explanations:

```bash
matheel explain-pair --dataset normalized_pairs --scores scored_pairs.csv \
  --score-row-index 12 --output-dir pair_explanation
```

The Explain tab also accepts your source files. Local matching explanations help
inspection; they do not reconstruct an external detector's reasoning.

## Interpret uncertainty

The default is a seeded, 95% percentile bootstrap over independent groups:

- Pair analysis keeps connected source IDs together. Supply
  `--group-column assignment` when assignment/author grouping is the appropriate
  declared unit.
- Retrieval resamples complete queries rather than query/document comparisons.
- Paired comparisons use identical resamples and report metric-difference bounds.
- Fewer than two groups produces N/A. AUROC and AP are not fabricated for
  one-class samples. Too few valid bootstrap rounds also produces N/A bounds.

Intervals are conditional on the supplied datasets and grouping policy. Queries
may share assignments, and distinct source IDs may refer to related code; neither
is automatically an independent sample. Overlapping intervals are not a
significance test. A small gap between rounded scores does not establish a
reliable difference.

## Published leaderboard diagnostics

The official view retains original scores, ranks, coverage and method IDs. It
adds visible wall times, failure status and a speed/quality Pareto flag. A
Pareto-efficient method has no eligible method with both an equal-or-better score
and equal-or-lower observed time, with at least one strict improvement. The flag
is calculated before method filtering. Runtime includes preparation and depends
on hardware and concurrent execution.

Generate a per-view uncertainty companion from audited score CSVs:

```bash
python scripts/analyze_plagiarism_benchmark.py \
  --result gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1.json \
  --scores /path/to/extracted/scores --rounds 500 \
  --output gradio_app/benchmark_data/plagiarism-v1-analysis.json
```

The companion records the exact result hash, protocol ID and score hashes. The
viewer rejects mismatches and never changes official rankings. Aggregate
confidence bounds are not inferred from rounded aggregate scores.
