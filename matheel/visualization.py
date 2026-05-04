import html
import json
import re
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .datasets import PairDataset, RetrievalDataset, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .vectors import build_static_hash_vectors


_PROJECTION_METHODS = ("auto", "umap", "pca")
_PAIR_SEGMENT_MODES = ("line", "token", "chunk")
_TOKEN_PATTERN = re.compile(r"\w+|[^\s\w]", re.UNICODE)
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


def available_pair_explanation_segment_modes():
    return _PAIR_SEGMENT_MODES


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


def build_dataset_embedding_map(
    dataset,
    kind="auto",
    method="auto",
    seed=7,
    static_vector_dim=256,
    document_metadata=None,
):
    loaded, dataset_kind = _load_visualization_dataset(dataset, kind=kind)
    texts = load_code_texts(loaded)
    document_ids = tuple(sorted(texts))
    vectors = build_static_hash_vectors(
        [texts[document_id] for document_id in document_ids],
        dim=int(static_vector_dim),
        lowercase=True,
    )
    metadata = _dataset_document_metadata(
        loaded,
        dataset_kind,
        document_ids,
        document_metadata=document_metadata,
    )
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


def write_dataset_embedding_map(
    dataset,
    output_dir,
    kind="auto",
    method="auto",
    seed=7,
    static_vector_dim=256,
    document_metadata=None,
):
    projection = build_dataset_embedding_map(
        dataset,
        kind=kind,
        method=method,
        seed=seed,
        static_vector_dim=static_vector_dim,
        document_metadata=document_metadata,
    )
    artifacts = write_dataset_map_artifacts(
        projection,
        output_dir,
        title=str(projection.attrs.get("dataset_name") or "Matheel Dataset Map"),
        color_column="role" if "role" in projection.columns else None,
    )
    return projection, artifacts


def build_pair_explanation(
    left_code,
    right_code,
    left_id="left",
    right_id="right",
    segment_mode="line",
    high_threshold=0.85,
    medium_threshold=0.6,
    low_threshold=0.3,
    chunk_size=5,
):
    selected_mode = _normalize_pair_segment_mode(segment_mode)
    thresholds = _normalize_pair_thresholds(
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
    )
    left_segments = _segment_code(left_code, selected_mode, chunk_size=chunk_size)
    right_segments = _segment_code(right_code, selected_mode, chunk_size=chunk_size)
    matches = _match_segments(left_segments, right_segments, thresholds=thresholds)
    _apply_pair_matches(left_segments, right_segments, matches)
    return {
        "schema_version": 1,
        "metadata": {
            "left_id": str(left_id),
            "right_id": str(right_id),
            "segment_mode": selected_mode,
            "similarity_metric": "levenshtein_normalized_similarity",
            "thresholds": thresholds,
            "chunk_size": int(chunk_size or 0) if selected_mode == "chunk" else None,
            "tokenizer": "regex_code_tokens" if selected_mode == "token" else None,
        },
        "left": {
            "document_id": str(left_id),
            "segments": left_segments,
        },
        "right": {
            "document_id": str(right_id),
            "segments": right_segments,
        },
        "matches": matches,
    }


def pair_explanation_payload(explanation):
    return _json_safe_pair_explanation(explanation)


def pair_explanation_html(explanation, title="Matheel Pair Explanation"):
    payload = pair_explanation_payload(explanation)
    metadata = payload["metadata"]
    escaped_title = html.escape(str(title))
    left_id = html.escape(str(payload["left"]["document_id"]))
    right_id = html.escape(str(payload["right"]["document_id"]))
    left_rows = "\n".join(_pair_segment_html(segment) for segment in payload["left"]["segments"])
    right_rows = "\n".join(_pair_segment_html(segment) for segment in payload["right"]["segments"])
    thresholds = metadata.get("thresholds", {})
    match_rows = "\n".join(_pair_match_table_row(match) for match in payload["matches"])
    if not match_rows:
        match_rows = '<tr><td colspan="4">No matching regions above the low threshold.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 8px; }}
    .meta {{ color: #4b5563; margin: 0 0 16px; }}
    .layout {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr); gap: 16px; align-items: start; }}
    .panel {{ border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; background: #ffffff; }}
    .panel h2 {{ font-size: 1rem; margin: 0; padding: 10px 12px; background: #f3f4f6; border-bottom: 1px solid #d1d5db; }}
    pre {{ margin: 0; padding: 8px; overflow-x: auto; background: #f9fafb; }}
    .segment {{ display: grid; grid-template-columns: 4.5rem minmax(0, 1fr); gap: 8px; padding: 3px 6px; border-left: 4px solid transparent; white-space: pre-wrap; }}
    .line-number {{ color: #6b7280; user-select: none; text-align: right; }}
    .level-high {{ background: #fee2e2; border-left-color: #dc2626; }}
    .level-medium {{ background: #fef3c7; border-left-color: #d97706; }}
    .level-low {{ background: #dbeafe; border-left-color: #2563eb; }}
    .level-none {{ color: #374151; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 16px; font-size: 0.9rem; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: -1px; border: 1px solid #9ca3af; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <p class="meta">mode={html.escape(str(metadata.get("segment_mode")))}; high>={float(thresholds.get("high", 0.0)):.2f}; medium>={float(thresholds.get("medium", 0.0)):.2f}; low>={float(thresholds.get("low", 0.0)):.2f}</p>
  <div class="legend">
    <span><span class="swatch level-high"></span>high</span>
    <span><span class="swatch level-medium"></span>medium</span>
    <span><span class="swatch level-low"></span>low</span>
    <span><span class="swatch level-none"></span>no match</span>
  </div>
  <div class="layout">
    <section class="panel">
      <h2>{left_id}</h2>
      <pre>{left_rows}</pre>
    </section>
    <section class="panel">
      <h2>{right_id}</h2>
      <pre>{right_rows}</pre>
    </section>
  </div>
  <table>
    <thead><tr><th>Match</th><th>Level</th><th>Score</th><th>Segments</th></tr></thead>
    <tbody>
{match_rows}
    </tbody>
  </table>
</body>
</html>
"""


def write_pair_explanation_artifacts(explanation, output_dir, basename="pair_explanation", title=None):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    safe_basename = _safe_artifact_basename(basename or "pair_explanation")
    json_path = target / f"{safe_basename}.json"
    html_path = target / f"{safe_basename}.html"
    payload = pair_explanation_payload(explanation)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_title = title or f"{payload['metadata']['left_id']} vs {payload['metadata']['right_id']}"
    html_path.write_text(pair_explanation_html(payload, title=html_title), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def write_pair_explanation(
    left_code,
    right_code,
    output_dir,
    left_id="left",
    right_id="right",
    segment_mode="line",
    high_threshold=0.85,
    medium_threshold=0.6,
    low_threshold=0.3,
    chunk_size=5,
    basename=None,
    title=None,
):
    explanation = build_pair_explanation(
        left_code,
        right_code,
        left_id=left_id,
        right_id=right_id,
        segment_mode=segment_mode,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
        chunk_size=chunk_size,
    )
    resolved_basename = basename or f"{left_id}_vs_{right_id}"
    artifacts = write_pair_explanation_artifacts(
        explanation,
        output_dir,
        basename=resolved_basename,
        title=title,
    )
    return explanation, artifacts


def build_pair_dataset_explanation(
    dataset,
    pair_index=0,
    left_id=None,
    right_id=None,
    segment_mode="line",
    high_threshold=0.85,
    medium_threshold=0.6,
    low_threshold=0.3,
    chunk_size=5,
):
    loaded = dataset if isinstance(dataset, PairDataset) else load_pair_dataset(dataset)
    pair = _select_pair_row(loaded, pair_index=pair_index, left_id=left_id, right_id=right_id)
    texts = load_code_texts(loaded)
    resolved_left_id = str(pair["left_id"])
    resolved_right_id = str(pair["right_id"])
    explanation = build_pair_explanation(
        texts[resolved_left_id],
        texts[resolved_right_id],
        left_id=resolved_left_id,
        right_id=resolved_right_id,
        segment_mode=segment_mode,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
        chunk_size=chunk_size,
    )
    explanation["metadata"]["dataset_name"] = str(loaded.metadata.get("name") or loaded.root.name)
    explanation["metadata"]["pair_index"] = int(pair.name) if getattr(pair, "name", None) is not None else int(pair_index)
    explanation["metadata"]["label"] = int(pair["label"])
    return explanation


def write_pair_dataset_explanation(
    dataset,
    output_dir,
    pair_index=0,
    left_id=None,
    right_id=None,
    segment_mode="line",
    high_threshold=0.85,
    medium_threshold=0.6,
    low_threshold=0.3,
    chunk_size=5,
    basename=None,
    title=None,
):
    explanation = build_pair_dataset_explanation(
        dataset,
        pair_index=pair_index,
        left_id=left_id,
        right_id=right_id,
        segment_mode=segment_mode,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        low_threshold=low_threshold,
        chunk_size=chunk_size,
    )
    metadata = explanation["metadata"]
    resolved_basename = basename or f"{metadata['left_id']}_vs_{metadata['right_id']}"
    artifacts = write_pair_explanation_artifacts(
        explanation,
        output_dir,
        basename=resolved_basename,
        title=title,
    )
    return explanation, artifacts


def _normalize_pair_segment_mode(segment_mode):
    selected = str(segment_mode or "line").strip().lower().replace("-", "_")
    if selected not in _PAIR_SEGMENT_MODES:
        supported = ", ".join(_PAIR_SEGMENT_MODES)
        raise ValueError(f"segment_mode must be one of: {supported}. Got: {segment_mode}")
    return selected


def _normalize_pair_thresholds(high_threshold=0.85, medium_threshold=0.6, low_threshold=0.3):
    thresholds = {
        "high": float(high_threshold),
        "medium": float(medium_threshold),
        "low": float(low_threshold),
    }
    if any(not np.isfinite(value) for value in thresholds.values()):
        raise ValueError("Pair explanation thresholds must be finite.")
    if not 0.0 <= thresholds["low"] <= thresholds["medium"] <= thresholds["high"] <= 1.0:
        raise ValueError("Pair explanation thresholds must satisfy 0 <= low <= medium <= high <= 1.")
    return thresholds


def _segment_code(code, segment_mode, chunk_size=5):
    text = str(code or "")
    if segment_mode == "line":
        return _line_segments(text)
    if segment_mode == "token":
        return _token_segments(text)
    return _chunk_segments(text, chunk_size=chunk_size)


def _line_segments(text):
    raw_lines = text.splitlines(keepends=True)
    if not raw_lines:
        raw_lines = [""]
    segments = []
    offset = 0
    for index, raw_line in enumerate(raw_lines):
        clean_line = raw_line.rstrip("\r\n")
        end_offset = offset + len(raw_line)
        segments.append(
            _base_segment(
                index=index,
                text=clean_line,
                start_char=offset,
                end_char=end_offset,
                start_line=index + 1,
                end_line=index + 1,
            )
        )
        offset = end_offset
    return segments


def _chunk_segments(text, chunk_size=5):
    lines = _line_segments(text)
    resolved_chunk_size = max(1, int(chunk_size or 1))
    chunks = []
    for start in range(0, len(lines), resolved_chunk_size):
        group = lines[start : start + resolved_chunk_size]
        chunks.append(
            _base_segment(
                index=len(chunks),
                text="\n".join(segment["text"] for segment in group),
                start_char=group[0]["start_char"],
                end_char=group[-1]["end_char"],
                start_line=group[0]["start_line"],
                end_line=group[-1]["end_line"],
            )
        )
    return chunks


def _token_segments(text):
    matches = list(_TOKEN_PATTERN.finditer(text))
    if not matches:
        return [
            _base_segment(
                index=0,
                text="",
                start_char=0,
                end_char=0,
                start_line=1,
                end_line=1,
            )
        ]
    line_starts = _line_start_offsets(text)
    segments = []
    for index, match in enumerate(matches):
        start_line = _line_number_for_offset(line_starts, match.start())
        end_line = _line_number_for_offset(line_starts, max(match.end() - 1, match.start()))
        segments.append(
            _base_segment(
                index=index,
                text=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start_line=start_line,
                end_line=end_line,
            )
        )
    return segments


def _base_segment(index, text, start_char, end_char, start_line, end_line):
    return {
        "index": int(index),
        "text": str(text),
        "start_char": int(start_char),
        "end_char": int(end_char),
        "start_line": int(start_line),
        "end_line": int(end_line),
        "match_id": None,
        "matched_index": None,
        "score": 0.0,
        "level": "none",
    }


def _line_start_offsets(text):
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def _line_number_for_offset(line_starts, offset):
    return max(1, bisect_right(line_starts, int(offset)))


def _match_segments(left_segments, right_segments, thresholds):
    candidates = []
    low_threshold = float(thresholds["low"])
    for left in left_segments:
        for right in right_segments:
            score = _segment_similarity(left["text"], right["text"])
            if score >= low_threshold:
                candidates.append((score, left["index"], right["index"]))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_left = set()
    used_right = set()
    accepted = []
    for score, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        accepted.append((left_index, right_index, score))

    matches = []
    for match_number, (left_index, right_index, score) in enumerate(sorted(accepted), start=1):
        matches.append(
            {
                "match_id": f"m{match_number}",
                "left_index": int(left_index),
                "right_index": int(right_index),
                "score": _rounded_score(score),
                "level": _score_level(score, thresholds),
            }
        )
    return matches


def _segment_similarity(left_text, right_text):
    left = str(left_text or "").strip()
    right = str(right_text or "").strip()
    if not left or not right:
        return 0.0
    return float(Levenshtein.normalized_similarity(left, right))


def _score_level(score, thresholds):
    value = float(score)
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    if value >= float(thresholds["low"]):
        return "low"
    return "none"


def _rounded_score(score):
    return round(float(score), 6)


def _apply_pair_matches(left_segments, right_segments, matches):
    left_by_index = {segment["index"]: segment for segment in left_segments}
    right_by_index = {segment["index"]: segment for segment in right_segments}
    for match in matches:
        left = left_by_index[match["left_index"]]
        right = right_by_index[match["right_index"]]
        for segment, matched_index in ((left, right["index"]), (right, left["index"])):
            segment["match_id"] = match["match_id"]
            segment["matched_index"] = int(matched_index)
            segment["score"] = match["score"]
            segment["level"] = match["level"]


def _select_pair_row(dataset, pair_index=0, left_id=None, right_id=None):
    pairs = dataset.pairs.copy()
    if left_id is not None or right_id is not None:
        if left_id is None or right_id is None:
            raise ValueError("Both left_id and right_id are required when selecting a pair by id.")
        mask = (pairs["left_id"].astype(str) == str(left_id)) & (pairs["right_id"].astype(str) == str(right_id))
        if not mask.any():
            reverse_mask = (pairs["left_id"].astype(str) == str(right_id)) & (
                pairs["right_id"].astype(str) == str(left_id)
            )
            if reverse_mask.any():
                mask = reverse_mask
        matches = pairs[mask]
        if matches.empty:
            raise ValueError(f"Pair dataset does not contain pair: {left_id} vs {right_id}")
        return matches.iloc[0]
    resolved_index = int(pair_index or 0)
    if resolved_index < 0 or resolved_index >= len(pairs):
        raise ValueError(f"pair_index must be between 0 and {len(pairs) - 1}. Got: {pair_index}")
    return pairs.iloc[resolved_index]


def _pair_segment_html(segment):
    level = _safe_css_level(segment.get("level"))
    line_label = _line_label(segment)
    text = html.escape(str(segment.get("text") or ""))
    if text == "":
        text = '<span class="empty">empty</span>'
    match_id = html.escape(str(segment.get("match_id") or ""))
    score = float(segment.get("score") or 0.0)
    title = f'{match_id} score={score:.4f}' if match_id else "no match"
    return (
        f'<span class="segment level-{level}" data-match-id="{match_id}" title="{html.escape(title)}">'
        f'<span class="line-number">{html.escape(line_label)}</span>'
        f"<code>{text}</code></span>"
    )


def _line_label(segment):
    start_line = int(segment.get("start_line") or 1)
    end_line = int(segment.get("end_line") or start_line)
    if start_line == end_line:
        return str(start_line)
    return f"{start_line}-{end_line}"


def _safe_css_level(level):
    value = str(level or "none").strip().lower()
    if value in ("high", "medium", "low", "none"):
        return value
    return "none"


def _pair_match_table_row(match):
    match_id = html.escape(str(match["match_id"]))
    level = html.escape(str(match["level"]))
    score = float(match["score"])
    segments = f"{int(match['left_index']) + 1} -> {int(match['right_index']) + 1}"
    return f"<tr><td>{match_id}</td><td>{level}</td><td>{score:.4f}</td><td>{segments}</td></tr>"


def _json_safe_pair_explanation(explanation):
    payload = json.loads(json.dumps(explanation))
    for side in ("left", "right"):
        for segment in payload[side]["segments"]:
            segment["level"] = _safe_css_level(segment.get("level"))
            segment["score"] = _rounded_score(segment.get("score", 0.0))
    for match in payload.get("matches", []):
        match["level"] = _safe_css_level(match.get("level"))
        match["score"] = _rounded_score(match.get("score", 0.0))
    return payload


def _safe_artifact_basename(value):
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    basename = basename.strip("._")
    return basename or "pair_explanation"


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


def _dataset_document_metadata(dataset, dataset_kind, document_ids, document_metadata=None):
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
    excluded_columns = {"file_id", "text"}
    metadata_columns = [
        "document_id",
        *[
            column
            for column in selected.columns
            if column != "document_id" and column not in excluded_columns
        ],
    ]
    records = selected[metadata_columns].to_dict(orient="records")
    return _merge_document_metadata(records, document_metadata)


def _merge_document_metadata(records, document_metadata):
    if document_metadata is None:
        return records
    base = pd.DataFrame(records)
    extra = _coerce_document_metadata(document_metadata)
    if "document_id" not in extra.columns:
        raise ValueError("document_metadata must include a document_id column.")
    extra = extra.copy()
    extra["document_id"] = extra["document_id"].astype(str)
    override_columns = [
        column
        for column in extra.columns
        if column != "document_id" and column in base.columns
    ]
    if override_columns:
        base = base.drop(columns=override_columns)
    merged = base.merge(extra, on="document_id", how="left")
    return merged.to_dict(orient="records")


def _coerce_document_metadata(document_metadata):
    if isinstance(document_metadata, pd.DataFrame):
        return document_metadata.copy()
    if isinstance(document_metadata, dict):
        if "document_id" in document_metadata:
            try:
                return pd.DataFrame(document_metadata)
            except ValueError:
                return pd.DataFrame([document_metadata])
        rows = []
        for document_id, value in document_metadata.items():
            row = {"document_id": document_id}
            if isinstance(value, dict):
                row.update(value)
            else:
                row["value"] = value
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame(document_metadata)


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
