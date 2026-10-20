# Exact lexical scoring and bounded retrieval

Dataset evaluation prepares lexical token streams and Winnowing fingerprints
once per distinct prepared text and configuration. GST indexes exact token prefixes
up to its effective minimum match length (capped at eight tokens for bounded
index storage), verifies extensions against the full minimum, and skips
starts that cannot tie the current longest match. Longest-tile selection, overlap
handling, tie order, short-input behavior, and normalization retain their existing
semantics. The optimization does not truncate inputs or apply approximate cutoffs.

## Two-stage experiments

Use an existing cheap scorer to select candidates, then an existing expensive
scorer to rerank them. This reduces expensive comparisons; it does **not** turn
the first stage into an approximate index. First-stage work remains O(Q*D), while
only O(Q*N) candidate rows and a size-N heap per query are retained. Multivector
stages preserve query/document roles without constructing a full Q*D score matrix.

```python
from matheel import evaluate_two_stage_retrieval_dataset

scored, metrics = evaluate_two_stage_retrieval_dataset(
    "normalized_retrieval", candidate_count=5, k=10,
    candidate_similarity_options={"feature_weights": {"winnowing": 1.0}},
    similarity_options={"feature_weights": {"gst": 1.0}},
    exclude_self_matches=True,
)
print(metrics["candidate_recall"], metrics["mean_average_precision"])
```

The default first stage is LCS and the default reranker is Levenshtein; both run
offline. Options files or dictionaries can choose other existing scorers. Python
also accepts custom `candidate_scorer` and reranking `scorer` callbacks, and a
custom reranking `algorithm` with the existing preparation hook. Candidate callbacks
and reranker callbacks receive identifiers, without relevance labels. Custom reranker preparation retains
the existing trusted algorithm contract, including access to the dataset.

Every declared query and **all original qrels** contribute to end-to-end metrics.
Missed relevant documents lower AP/MAP and recall. Omitted candidates remain absent
from the ranking; they are never inserted as zero-score ties. Candidate ties use
ascending document IDs. Self exclusion follows file identity and rejects positive
self judgments. A query with no remaining candidates still contributes zero
retrieval metrics. Queries without positive judgments remain in evaluation;
candidate recall is undefined for them and reports their count separately.

The report includes macro candidate recall over queries with positives, micro
recall over all positive judgments, per-query diagnostics, selected/full candidate
counts, both stage comparison counts and times, and scorer configurations. Candidate
recall measures the supplied judgments, whose annotation limits still apply.

```bash
matheel evaluate-two-stage-retrieval normalized_retrieval \
  --candidate-count 5 --candidate-feature-weight winnowing=1 \
  --feature-weight gst=1 --k 10 --exclude-self-matches \
  --scores-out experiment/scores.csv --metrics-out experiment/metrics.json
```

`--candidate-options` and `--reranker-options` accept JSON scorer-options files,
including language, tokenizer, preparation, and model settings. Feature-weight
flags override those files' feature weights. The scores CSV contains both candidate
score/rank and final similarity; the JSON contains metrics and full diagnostics.
Run `examples/evaluation/two_stage_retrieval_demo.py` for a complete synthetic
example that needs neither downloads nor models.

Custom leaderboard methods can declare `retrieval_options`:

```json
{
  "name": "Winnowing top5 then GST",
  "feature_weights": {"gst": 1.0},
  "retrieval_options": {
    "candidate_count": 5,
    "candidate_similarity_options": {"feature_weights": {"winnowing": 1.0}}
  }
}
```

The complete pipeline policy participates in method identity. Pipeline reports
are separate experiments. The frozen exhaustive v1 runner rejects shortlist
options; existing v1 protocols, scores, timeouts, and published artifacts remain
unchanged. Use a new artifact directory for every new experiment.

## Predeclare applicability

Methods can declare task exclusions before scoring:

```json
{
  "name": "GST for pairs",
  "feature_weights": {"gst": 1.0},
  "task_eligibility": {
    "retrieval": {"status": "not_applicable", "reason": "Exhaustive retrieval is outside this experiment's runtime scope."}
  },
  "dataset_exclusions": {"large_pairs": "This experiment includes small pair tasks only."}
}
```

Task keys are `pair` or `retrieval`; status is `eligible` or `not_applicable`, with
a nonempty reason required for exclusions. Dataset exclusions use declared dataset
names. Omitted fields retain existing applicability and method identities. The
declarations themselves participate in new method identities.

Excluded rows have status `not_applicable`, the declared reason, zero comparisons,
and null score/rank. Runtime errors remain `failed`. Any exclusion in an aggregate's
declared scope suppresses its mean and rank, even when legacy partial averaging is
enabled. Other applicable task families can still rank normally. Frozen-result
validation checks exclusions against the recorded method configuration; a failed
run cannot be relabeled N/A after execution. Older published rows without status
remain valid through their existing score/error/eligibility fields.

`examples/evaluation/retrieval_methods.example.json` provides a conservative new
experiment policy for the five methods that exceeded v1's 1,800-second method
budget: GST, Lexical Only, Partial Ratio, Partial Token Sort Ratio, and WRatio.
It is an example scope declaration, not a global restriction or a claim that pair
scoring will always fit the budget. Review runtime evidence before widening scope.

## Measured feasibility example

Bounded reference runs used the complete SOCO14 C++ A2 judged-endpoint view:
145 queries and 145 corpus documents, with 20,880 allowed comparisons after
self exclusion. Inputs came from the frozen v1 preparation recipe. The reference
environment was CPython 3.12 on macOS arm64, with numerical-library threads limited
to one. These are separate experiments with their own output directories.

| Configuration | Budget | Outcome | First-stage comparisons / seconds | Completed reranks / seconds | Macro candidate recall |
| --- | --- | --- | --- | --- | --- |
| Exhaustive GST | 60 s | Timed out | — | Not fully measured | — |
| Exhaustive Lexical Only | 60 s | Timed out | — | Not fully measured | — |
| Exhaustive Partial Ratio | 60 s | Timed out | — | Not fully measured | — |
| LCS top 10 → GST | 90 s | Timed out | Not fully reported | Not fully measured | Not reported |
| LCS top 1 → GST | 60 s | Completed in 31.75 s | 20,880 / 0.901 s | 145 / 29.30 s | 0.7713 |
| LCS top 10 → Partial Ratio | 90 s | Timed out | Not fully reported | Not fully measured | Not reported |
| LCS top 1 → Partial Ratio | 60 s | Completed in 10.47 s | 20,880 / 0.924 s | 145 / 7.93 s | 0.7713 |
| Winnowing top 5 → GST | 90 s | Completed in 34.67 s | 20,880 / 0.417 s | 725 / 32.66 s | 1.0 |
| Winnowing top 5 → Partial Ratio | 90 s | Completed in 26.75 s | 20,880 / 0.424 s | 725 / 24.72 s | 1.0 |

The completed top-1 experiments retained 122 of 172 positive judgments (micro
candidate recall 0.7093). All 145 queries and 172 positives remained in the final
metric denominators; MAP was 0.7713. A top-1 shortlist demonstrates a runtime/recall
tradeoff and supplies fewer than the requested `k=10` results. Choose and record N
and the time budget before evaluation. Candidate recall is diagnostic and does not
control admission, thresholds, retained methods, or ranks. Runtime choices were
bounded in advance; these observations do not alter the original 1,800-second v1
run or establish feasibility outside the tested configurations. The predeclared
Winnowing top-5 experiments retained all 172 positive judgments in this view;
MAP was 0.9671 with GST and 0.9327 with Partial Ratio. The candidate count and
budget were chosen before observing those recall values. Both top-1 and top-5
shortlists supply fewer than ten results at `k=10`; precision retains its declared
denominator. Inspect query-level diagnostics and repeat on other data before
making broader performance or quality claims.

## Reproduce a bounded GST profile

From a checkout containing the trusted `v0.7.0` tag:

```bash
python scripts/profile_lexical_scoring.py --baseline-ref v0.7.0 --seconds 5
python scripts/profile_lexical_scoring.py --pair-dataset normalized_pairs --seconds 5
```

Each before/after case runs in a bounded subprocess and records scores, token
counts, input hashes, wall time, and cProfile call counts. Optional real inputs
select the smallest, middle, and largest supplied pair by character length.
Source text is not exported. A deadline is recorded as a timeout, and parity is
asserted only when both implementations finish. Exactness is also covered by
an independent exhaustive GST oracle and deterministic adversarial/random cases.

An initial synthetic profile with 750 punctuation-heavy tokens retained score 0
and reduced cProfile calls from 630,017 to 4,489 (0.115 s to 0.00128 s in one
observation). An identical 1,200-token case retained score 1 and measured 0.125 s
before versus 0.00110 s after. These fixtures exercise exact pruning; the real
feasibility runs above show why a synthetic speedup alone does not establish
corpus-scale suitability. A separate cached-score check compared 20 full-source
SOCO14 C++ A2 observations each for Levenshtein, LCS, Winnowing, and Partial Token
Ratio: all 80 floats matched the published CSV values exactly.
