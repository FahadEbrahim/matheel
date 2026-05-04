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
