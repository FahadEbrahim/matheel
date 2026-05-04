import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import PairDataset, RetrievalDataset, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .vectors import build_static_hash_vectors


_PROJECTION_METHODS = ("auto", "umap", "pca")
_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#9333ea",
    "#0891b2",
    "#db2777",
    "#4b5563",
)


def available_projection_methods():
    return _PROJECTION_METHODS


def project_embeddings(embeddings, method="auto", seed=7):
    array = _coerce_embedding_matrix(embeddings)
    selected = _normalize_projection_method(method)
    if array.shape[0] == 0:
        raise ValueError("embeddings must contain at least one vector.")

    actual_method = selected
    if selected == "auto":
        try:
            coordinates = _project_umap(array, seed=seed)
            actual_method = "umap"
        except ImportError:
            coordinates = _project_pca(array)
            actual_method = "pca"
    elif selected == "umap":
        coordinates = _project_umap(array, seed=seed)
    else:
        coordinates = _project_pca(array)

    frame = pd.DataFrame(coordinates, columns=["x", "y"])
    frame.attrs["projection_method"] = actual_method
    frame.attrs["requested_projection_method"] = selected
    frame.attrs["seed"] = int(seed)
    frame.attrs["embedding_count"] = int(array.shape[0])
    frame.attrs["embedding_dim"] = int(array.shape[1])
    return frame


def build_embedding_projection(embeddings, ids=None, metadata=None, method="auto", seed=7):
    array = _coerce_embedding_matrix(embeddings)
    document_ids = _document_ids_for_embeddings(array, ids)
    projection = project_embeddings(array, method=method, seed=seed)
    projection.insert(0, "document_id", document_ids)
    attrs = dict(projection.attrs)
    if metadata is not None:
        metadata_frame = pd.DataFrame(metadata).copy()
        if "document_id" not in metadata_frame.columns:
            raise ValueError("metadata must include a document_id column.")
        metadata_frame["document_id"] = metadata_frame["document_id"].astype(str)
        projection = projection.merge(metadata_frame, on="document_id", how="left")
        projection.attrs.update(attrs)
    return projection


def build_dataset_embedding_map(dataset, kind="auto", method="auto", seed=7, static_vector_dim=256):
    loaded, dataset_kind = _load_visualization_dataset(dataset, kind=kind)
    texts = load_code_texts(loaded)
    document_ids = tuple(sorted(texts))
    vectors = build_static_hash_vectors(
        [texts[document_id] for document_id in document_ids],
        dim=int(static_vector_dim),
        lowercase=True,
    )
    metadata = _dataset_document_metadata(loaded, dataset_kind, document_ids)
    projection = build_embedding_projection(
        vectors,
        ids=document_ids,
        metadata=metadata,
        method=method,
        seed=seed,
    )
    projection.attrs["dataset_kind"] = dataset_kind
    projection.attrs["dataset_name"] = str(loaded.metadata.get("name") or loaded.root.name)
    projection.attrs["embedding_source"] = "static_hash"
    projection.attrs["static_vector_dim"] = int(static_vector_dim)
    return projection


def dataset_map_payload(projection):
    frame = projection.copy() if isinstance(projection, pd.DataFrame) else pd.DataFrame(projection)
    required = {"document_id", "x", "y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"projection is missing required columns: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "metadata": _json_safe_dict(getattr(frame, "attrs", {})),
        "points": [_json_safe_dict(row) for row in frame.to_dict(orient="records")],
    }


def dataset_map_html(projection, title="Matheel Dataset Map", color_column=None):
    frame = projection.copy() if isinstance(projection, pd.DataFrame) else pd.DataFrame(projection)
    if frame.empty:
        raise ValueError("projection must contain at least one point.")
    color_column = color_column or _default_color_column(frame)
    points = _svg_points(frame, color_column=color_column)
    legend = _legend_html(points)
    rows = "\n".join(_point_table_row(point) for point in points)
    escaped_title = html.escape(str(title))
    svg_body = "\n".join(_svg_circle(point) for point in points)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 16px; }}
    .layout {{ display: grid; grid-template-columns: minmax(360px, 2fr) minmax(260px, 1fr); gap: 20px; align-items: start; }}
    svg {{ width: 100%; height: auto; border: 1px solid #d1d5db; background: #f9fafb; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 12px; font-size: 0.9rem; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="legend">{legend}</div>
  <div class="layout">
    <svg viewBox="0 0 960 620" role="img" aria-label="{escaped_title}">
      <rect x="48" y="32" width="864" height="520" fill="#ffffff" stroke="#e5e7eb"/>
      {svg_body}
    </svg>
    <table>
      <thead><tr><th>Document</th><th>{html.escape(str(color_column or "Group"))}</th><th>x</th><th>y</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def write_dataset_map_artifacts(projection, output_dir, basename="dataset_map", title=None, color_column=None):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    frame = projection.copy() if isinstance(projection, pd.DataFrame) else pd.DataFrame(projection)
    csv_path = target / f"{basename}.csv"
    json_path = target / f"{basename}.json"
    html_path = target / f"{basename}.html"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(dataset_map_payload(frame), indent=2, sort_keys=True), encoding="utf-8")
    html_title = title or str(frame.attrs.get("dataset_name") or "Matheel Dataset Map")
    html_path.write_text(dataset_map_html(frame, title=html_title, color_column=color_column), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "html": html_path}


def write_dataset_embedding_map(dataset, output_dir, kind="auto", method="auto", seed=7, static_vector_dim=256):
    projection = build_dataset_embedding_map(
        dataset,
        kind=kind,
        method=method,
        seed=seed,
        static_vector_dim=static_vector_dim,
    )
    artifacts = write_dataset_map_artifacts(
        projection,
        output_dir,
        title=str(projection.attrs.get("dataset_name") or "Matheel Dataset Map"),
        color_column="role" if "role" in projection.columns else None,
    )
    return projection, artifacts


def _normalize_projection_method(method):
    selected = str(method or "auto").strip().lower().replace("-", "_")
    if selected not in _PROJECTION_METHODS:
        supported = ", ".join(_PROJECTION_METHODS)
        raise ValueError(f"Projection method must be one of: {supported}. Got: {method}")
    return selected


def _coerce_embedding_matrix(embeddings):
    if isinstance(embeddings, pd.DataFrame):
        array = embeddings.to_numpy(dtype=float)
    else:
        array = np.asarray(list(embeddings), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    if array.shape[1] == 0:
        raise ValueError("embeddings must contain at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("embeddings must contain only finite values.")
    return array


def _document_ids_for_embeddings(array, ids):
    if ids is None:
        return [f"doc_{index + 1}" for index in range(array.shape[0])]
    document_ids = [str(value) for value in ids]
    if len(document_ids) != array.shape[0]:
        raise ValueError("ids must have the same length as embeddings.")
    return document_ids


def _project_pca(array):
    if array.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    centered = array - array.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    coordinates = centered @ components
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(array.shape[0], dtype=float)])
    return coordinates[:, :2]


def _project_umap(array, seed=7):
    if array.shape[0] < 3:
        return _project_pca(array)
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP projection requires the optional 'umap-learn' dependency. "
            "Install `matheel[visualization]` or use method='pca'."
        ) from exc
    n_neighbors = min(15, max(2, array.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=int(seed))
    return np.asarray(reducer.fit_transform(array), dtype=float)


def _load_visualization_dataset(dataset, kind="auto"):
    if isinstance(dataset, PairDataset):
        return dataset, "pair_classification"
    if isinstance(dataset, RetrievalDataset):
        return dataset, "retrieval"
    path = Path(dataset)
    selected = str(kind or "auto").strip().lower()
    if selected in ("pair", "pair_classification"):
        return load_pair_dataset(path), "pair_classification"
    if selected == "retrieval":
        return load_retrieval_dataset(path), "retrieval"
    if (path / "pairs.csv").exists():
        return load_pair_dataset(path), "pair_classification"
    if (path / "queries.csv").exists() and (path / "corpus.csv").exists():
        return load_retrieval_dataset(path), "retrieval"
    raise ValueError("Could not detect dataset kind. Use kind='pair' or kind='retrieval'.")


def _dataset_document_metadata(dataset, dataset_kind, document_ids):
    files = dataset.files.copy()
    files["document_id"] = files["file_id"].astype(str)
    selected = files[files["document_id"].isin(document_ids)].copy()
    if dataset_kind == "retrieval":
        selected["role"] = selected["document_id"].map(_retrieval_file_roles(dataset))
        selected["role"] = selected["role"].fillna("other")
    else:
        selected["role"] = "submission"
        pair_counts = _pair_file_counts(dataset)
        selected["pair_count"] = selected["document_id"].map(pair_counts).fillna(0).astype(int)
    metadata_columns = [column for column in ("document_id", "file_path", "role", "pair_count") if column in selected.columns]
    return selected[metadata_columns].to_dict(orient="records")


def _pair_file_counts(dataset):
    counts = {}
    for row in dataset.pairs.to_dict(orient="records"):
        for key in ("left_id", "right_id"):
            file_id = str(row[key])
            counts[file_id] = counts.get(file_id, 0) + 1
    return counts


def _retrieval_file_roles(dataset):
    roles = {}
    for row in dataset.queries.to_dict(orient="records"):
        roles.setdefault(str(row["file_id"]), set()).add("query")
    for row in dataset.corpus.to_dict(orient="records"):
        roles.setdefault(str(row["file_id"]), set()).add("document")
    return {
        file_id: "both" if len(role_set) > 1 else next(iter(role_set))
        for file_id, role_set in roles.items()
    }


def _default_color_column(frame):
    for column in ("role", "label", "dataset", "split", "cluster", "algorithm"):
        if column in frame.columns:
            return column
    return None


def _svg_points(frame, color_column=None):
    x_values = frame["x"].astype(float).to_numpy()
    y_values = frame["y"].astype(float).to_numpy()
    groups = _color_groups(frame, color_column)
    colors = {group: _PALETTE[index % len(_PALETTE)] for index, group in enumerate(sorted(set(groups)))}
    points = []
    for index, row in enumerate(frame.to_dict(orient="records")):
        group = groups[index]
        points.append(
            {
                **row,
                "_screen_x": _scale_value(float(row["x"]), x_values, 72, 888),
                "_screen_y": _scale_value(float(row["y"]), y_values, 528, 56),
                "_group": group,
                "_color": colors[group],
            }
        )
    return points


def _color_groups(frame, color_column):
    if color_column is None or color_column not in frame.columns:
        return ["documents"] * len(frame)
    return [str(value) if pd.notna(value) else "unknown" for value in frame[color_column].tolist()]


def _scale_value(value, values, output_min, output_max):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        return (output_min + output_max) / 2.0
    ratio = (value - minimum) / (maximum - minimum)
    return output_min + ratio * (output_max - output_min)


def _svg_circle(point):
    document_id = html.escape(str(point["document_id"]))
    group = html.escape(str(point["_group"]))
    return (
        f'<circle cx="{point["_screen_x"]:.2f}" cy="{point["_screen_y"]:.2f}" r="7" '
        f'fill="{point["_color"]}" stroke="#111827" stroke-width="1">'
        f"<title>{document_id} ({group})</title></circle>"
    )


def _legend_html(points):
    seen = {}
    for point in points:
        seen.setdefault(point["_group"], point["_color"])
    return " ".join(
        f'<span><span class="swatch" style="background:{color}"></span>{html.escape(str(group))}</span>'
        for group, color in sorted(seen.items())
    )


def _point_table_row(point):
    document_id = html.escape(str(point["document_id"]))
    group = html.escape(str(point["_group"]))
    return (
        f"<tr><td>{document_id}</td><td>{group}</td>"
        f"<td>{float(point['x']):.4f}</td><td>{float(point['y']):.4f}</td></tr>"
    )


def _json_safe_dict(values):
    payload = {}
    for key, value in dict(values).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (np.integer,)):
            payload[str(key)] = int(value)
        elif isinstance(value, (np.floating,)):
            payload[str(key)] = float(value)
        elif pd.isna(value):
            payload[str(key)] = None
        else:
            payload[str(key)] = value
    return payload
