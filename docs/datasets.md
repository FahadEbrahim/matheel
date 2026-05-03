# Datasets and Evaluation

Matheel supports local labeled pair and retrieval datasets as the first step toward benchmark workflows.

No external datasets are bundled or downloaded by default. Add public datasets only after checking licensing, provenance, access requirements, and expected task meaning.
Use `task_type: plagiarism` for plagiarism-oriented datasets so they remain easy to separate from future dataset families.

## Dataset Registry

Dataset tracking uses these fields:

| Field | Meaning |
| --- | --- |
| `name` | Stable dataset identifier. |
| `task_type` | High-level task label. Use `plagiarism` for plagiarism-oriented datasets. |
| `dataset_kind` | Dataset layout, such as `pair_classification` or `retrieval`. |
| `languages` | Languages covered by the dataset. |
| `license` | Dataset license or `unknown`. |
| `source_url` | Public source page. |
| `access` | `bundled`, `download`, `manual`, or `external`. |
| `citation` | Citation text or DOI when available. |
| `notes` | Short caveats about labels, splits, or preprocessing. |

The built-in registry starts empty. Register datasets in code only after deciding they should be tracked by Matheel:

```python
from matheel.datasets import register_dataset_entry

register_dataset_entry(
    "example_plagiarism_dataset",
    task_type="plagiarism",
    dataset_kind="pair_classification",
    languages=("python",),
    license="unknown",
    source_url="https://example.org/dataset",
    access="manual",
    notes="Example registry entry only.",
)
```

## Pair Dataset Format

A pair-classification dataset directory contains:

```text
metadata.json
files.csv
pairs.csv
files/
```

`metadata.json` should include:

```json
{
  "task_type": "plagiarism",
  "dataset_kind": "pair_classification",
  "name": "tiny_plagiarism_fixture"
}
```

`files.csv` must include:

| Column | Meaning |
| --- | --- |
| `file_id` | Stable file identifier with no path separators. |
| `file_path` | Relative path under the dataset directory. |

`pairs.csv` must include:

| Column | Meaning |
| --- | --- |
| `left_id` | File id for the first submission. |
| `right_id` | File id for the second submission. |
| `label` | Binary label, where `1` means positive/plagiarism match and `0` means negative. |

You can write a small dataset programmatically:

```python
import pandas as pd
from matheel.datasets import write_pair_dataset

write_pair_dataset(
    "tiny_pairs",
    files=pd.DataFrame(
        [
            {"file_id": "a", "text": "print(1)", "suffix": ".py"},
            {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            {"file_id": "c", "text": "print(2)", "suffix": ".py"},
        ]
    ),
    pairs=pd.DataFrame(
        [
            {"left_id": "a", "right_id": "b", "label": 1},
            {"left_id": "a", "right_id": "c", "label": 0},
        ]
    ),
    metadata={"name": "tiny_plagiarism_fixture"},
)
```

## Pair Evaluation

Use the CLI to score local pair datasets and write both scored rows and metrics:

```bash
matheel evaluate-pairs tiny_pairs \
  --feature-weight levenshtein=1.0 \
  --threshold 0.8 \
  --scores-out scored_pairs.csv \
  --metrics-out pair_metrics.json
```

The command defaults to `levenshtein=1.0` so small local evaluations are offline-friendly. Add semantic features explicitly when you want model-backed scoring.

## Retrieval Dataset Format

A retrieval dataset directory contains:

```text
metadata.json
files.csv
queries.csv
corpus.csv
qrels.csv
files/
```

`metadata.json` should include:

```json
{
  "task_type": "plagiarism",
  "dataset_kind": "retrieval",
  "name": "tiny_plagiarism_retrieval_fixture"
}
```

`files.csv` uses the same columns as pair datasets. `queries.csv` maps query ids to files:

| Column | Meaning |
| --- | --- |
| `query_id` | Stable query identifier with no path separators. |
| `file_id` | File id from `files.csv`. |

`corpus.csv` maps candidate document ids to files:

| Column | Meaning |
| --- | --- |
| `document_id` | Stable document identifier with no path separators. |
| `file_id` | File id from `files.csv`. |

`qrels.csv` stores relevance judgments:

| Column | Meaning |
| --- | --- |
| `query_id` | Query id from `queries.csv`. |
| `document_id` | Candidate document id from `corpus.csv`. |
| `relevance` | Non-negative relevance score. Values greater than `0` are treated as relevant. |

You can write a small retrieval dataset programmatically:

```python
import pandas as pd
from matheel.datasets import write_retrieval_dataset

write_retrieval_dataset(
    "tiny_retrieval",
    files=pd.DataFrame(
        [
            {"file_id": "query_a", "text": "print(1)", "suffix": ".py"},
            {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
            {"file_id": "doc_b", "text": "print(2)", "suffix": ".py"},
        ]
    ),
    queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
    corpus=pd.DataFrame(
        [
            {"document_id": "d1", "file_id": "doc_a"},
            {"document_id": "d2", "file_id": "doc_b"},
        ]
    ),
    qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
    metadata={"name": "tiny_plagiarism_retrieval_fixture"},
)
```

## Retrieval Evaluation

Use the CLI to score each query against every corpus document and write ranking metrics:

```bash
matheel evaluate-retrieval tiny_retrieval \
  --feature-weight levenshtein=1.0 \
  --k 10 \
  --scores-out scored_retrieval.csv \
  --metrics-out retrieval_metrics.json
```

The metrics include mean average precision, mean reciprocal rank, precision at `k`, recall at `k`, and nDCG at `k`.

## Resampling and Uncertainty

Use resampling when you want uncertainty summaries instead of only one point metric. Matheel provides split generators for single splits, k-fold, repeated k-fold, and bootstrap:

```python
from matheel.resampling import bootstrap_resamples, kfold_splits, single_split

single = single_split(100, train_size=0.7, validation_size=0.1, seed=7)
folds = kfold_splits(100, n_splits=5, seed=7)
bootstraps = bootstrap_resamples(100, n_rounds=100, seed=7)
```

For pair-classification results, splits are applied to scored pair rows:

```python
from matheel.evaluation import evaluate_pair_resamples
from matheel.resampling import kfold_splits

splits = kfold_splits(len(scored_pairs), n_splits=5, seed=7)
fold_metrics, fold_summary = evaluate_pair_resamples(
    scored_pairs,
    splits,
    threshold=0.8,
)
```

For retrieval results, splits are applied to query ids, so every selected query keeps its candidate documents:

```python
from matheel.evaluation import evaluate_retrieval_resamples
from matheel.resampling import kfold_splits

query_ids = sorted(scored_retrieval["query_id"].unique())
splits = kfold_splits(query_ids, n_splits=5, seed=7)
fold_metrics, fold_summary = evaluate_retrieval_resamples(
    scored_retrieval,
    splits,
    k=10,
)
```

`fold_summary` contains percentile intervals for each metric across the selected resamples. For paired comparisons between two configurations, use `compare_metric_samples(...)` on matching split-level metric values:

```python
from matheel.resampling import compare_metric_samples

comparison = compare_metric_samples(
    baseline_fold_metrics["f1"],
    candidate_fold_metrics["f1"],
    metric_name="f1",
)
```

The comparison report includes mean difference, interval bounds, win/loss/tie counts, and a two-sided sign-test p-value.
