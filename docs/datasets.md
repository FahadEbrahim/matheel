# Datasets and Evaluation

Matheel supports local labeled pair datasets as the first step toward benchmark workflows.

No external datasets are bundled or downloaded by default. Add public datasets only after checking licensing, provenance, access requirements, and expected task meaning.

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
