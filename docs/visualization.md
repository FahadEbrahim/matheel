# Visualization

Matheel can export dataset-level embedding maps for normalized pair and retrieval datasets. The first visualization path uses dependency-free static hash embeddings by default, then projects documents to two dimensions with UMAP when `umap-learn` is installed. If UMAP is unavailable and `method=auto`, Matheel uses deterministic PCA.

Install the optional UMAP backend when you want UMAP projections:

```bash
pip install "matheel[visualization]"
```

## CLI

Generate CSV, JSON, and HTML artifacts:

```bash
matheel visualize-dataset ./normalized_pairs \
  --kind pair \
  --method auto \
  --seed 7 \
  --output-dir visualization_artifacts
```

The output directory contains:

- `dataset_map.csv`: projected coordinates and document metadata.
- `dataset_map.json`: machine-readable points plus projection metadata.
- `dataset_map.html`: static scatter plot for local inspection or report bundles.

Use `--method pca` for dependency-free deterministic projections, or `--method umap` to require UMAP explicitly.

## Python

```python
from matheel.visualization import build_dataset_embedding_map, write_dataset_map_artifacts

projection = build_dataset_embedding_map(
    "./normalized_pairs",
    kind="pair",
    method="pca",
    seed=7,
)

artifacts = write_dataset_map_artifacts(
    projection,
    "visualization_artifacts",
)

print(artifacts["html"])
```

Projection metadata is stored in `projection.attrs`, including the requested method, actual method, seed, dataset kind, dataset name, and embedding source.

Dataset maps preserve normalized `files.csv` metadata except raw code text, and Python callers can merge extra per-document metadata keyed by `document_id`. Use this for color columns such as split, cluster, label, metric score, or algorithm family:

```python
projection = build_dataset_embedding_map(
    "./normalized_pairs",
    kind="pair",
    method="pca",
    document_metadata=[
        {"document_id": "a", "metric_score": 0.91, "algorithm": "baseline"},
        {"document_id": "b", "metric_score": 0.88, "algorithm": "baseline"},
    ],
)

artifacts = write_dataset_map_artifacts(
    projection,
    "visualization_artifacts",
    color_column="metric_score",
)
```

## Pair Explanations

Pair explanations export side-by-side HTML and machine-readable JSON for a selected code pair. Matheel segments each submission by line, token, or fixed-size line chunks, then marks non-overlapping local matches as high, medium, low, or no match.

```bash
matheel explain-pair left.py right.py \
  --segment-mode line \
  --output-dir pair_explanations
```

For normalized pair datasets, select a row or pair ids:

```bash
matheel explain-pair \
  --dataset ./normalized_pairs \
  --left-id a \
  --right-id b \
  --output-dir pair_explanations
```

For compare results from a directory or ZIP archive, select the relative file names:

```bash
matheel explain-pair \
  --source ./submissions.zip \
  --left-name a.py \
  --right-name b.py \
  --output-dir pair_explanations
```

The JSON artifact stores segment offsets, line numbers, match ids, scores, thresholds, and segmentation metadata. The HTML artifact uses the same data for local inspection.

```python
from matheel.visualization import build_pair_explanation, write_pair_explanation_artifacts

explanation = build_pair_explanation(
    "def add(a, b):\n    return a + b",
    "def add(x, y):\n    return x + y",
    segment_mode="line",
)

artifacts = write_pair_explanation_artifacts(
    explanation,
    "pair_explanations",
)

print(artifacts["html"])
```
