import csv
import json
from html import escape
from pathlib import Path

import pandas as pd

from ._path_utils import (
    is_unsafe_relative_path,
    resolve_relative_path_within_root,
)


PAIR_KIND = "pair_classification"
RETRIEVAL_KIND = "retrieval"


def validate_dataset_report(dataset_root, kind="auto"):
    """Inspect a normalized Matheel dataset without mutating it."""
    root = Path(dataset_root).expanduser().resolve()
    issues = []
    counts = {}
    metadata = _read_metadata(root, issues)
    dataset_kind = _resolve_dataset_kind(root, kind, metadata, issues)

    if dataset_kind == PAIR_KIND:
        _inspect_pair_dataset(root, metadata, counts, issues)
    elif dataset_kind == RETRIEVAL_KIND:
        _inspect_retrieval_dataset(root, metadata, counts, issues)
    else:
        _add_issue(
            issues,
            "error",
            "unknown_dataset_kind",
            "Could not determine dataset kind from manifests. Use kind='pair' or kind='retrieval'.",
        )

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "error" if error_count else "warning" if warning_count else "pass"
    return {
        "schema_version": 1,
        "dataset_kind": dataset_kind or "unknown",
        "dataset_name": str(metadata.get("name") or root.name),
        "root_name": root.name,
        "status": status,
        "error_count": error_count,
        "warning_count": warning_count,
        "counts": counts,
        "metadata": metadata,
        "issues": issues,
    }


def dataset_validation_report_payload(report):
    return _json_safe(report)


def dataset_validation_report_html(report):
    payload = dataset_validation_report_payload(report)
    title = escape(str(payload.get("dataset_name") or "dataset"))
    status = escape(str(payload.get("status") or "unknown"))
    counts_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(escape(str(key)), escape(str(value)))
        for key, value in sorted((payload.get("counts") or {}).items())
    )
    if not counts_rows:
        counts_rows = '<tr><td colspan="2">No counts available</td></tr>'

    issue_rows = []
    for issue in payload.get("issues") or []:
        issue_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(issue.get("severity", ""))),
                escape(str(issue.get("code", ""))),
                escape(str(issue.get("count", ""))),
                escape(str(issue.get("message", ""))),
            )
        )
    if not issue_rows:
        issue_rows.append('<tr><td colspan="4">No validation issues</td></tr>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Matheel Dataset Validation</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2328; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 0.4rem; }}
    h2 {{ font-size: 1.05rem; margin-top: 1.4rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 0.6rem; }}
    th, td {{ border: 1px solid #d6d9df; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .status {{ display: inline-block; border: 1px solid #d6d9df; border-radius: 6px; padding: 2px 8px; background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Dataset kind: {escape(str(payload.get("dataset_kind", "unknown")))}. Status: <span class="status">{status}</span>.</p>
  <h2>Counts</h2>
  <table><tbody>{counts_rows}</tbody></table>
  <h2>Issues</h2>
  <table>
    <thead><tr><th>Severity</th><th>Code</th><th>Count</th><th>Message</th></tr></thead>
    <tbody>{''.join(issue_rows)}</tbody>
  </table>
</body>
</html>
"""


def write_dataset_validation_report(dataset_root, output_dir, kind="auto", basename="dataset_validation"):
    report = validate_dataset_report(dataset_root, kind=kind)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_basename(basename or "dataset_validation")
    artifacts = {
        "summary_json": target / f"{stem}_summary.json",
        "issues_csv": target / f"{stem}_issues.csv",
        "report_html": target / f"{stem}_report.html",
        "report_json": target / f"{stem}_report.json",
    }
    payload = dataset_validation_report_payload(report)
    artifacts["summary_json"].write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"issues", "metadata"}
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_issues_csv(payload.get("issues") or [], artifacts["issues_csv"])
    artifacts["report_html"].write_text(dataset_validation_report_html(payload), encoding="utf-8")
    artifacts["report_json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report, artifacts


def _inspect_pair_dataset(root, metadata, counts, issues):
    _inspect_metadata(metadata, PAIR_KIND, issues)
    files = _read_csv_manifest(root, "files.csv", ("file_id", "file_path"), issues)
    pairs = _read_csv_manifest(root, "pairs.csv", ("left_id", "right_id", "label"), issues)
    counts["files"] = int(len(files)) if files is not None else 0
    counts["pairs"] = int(len(pairs)) if pairs is not None else 0

    file_ids = _inspect_files_manifest(root, files, counts, issues)
    if pairs is None:
        return

    _add_duplicate_issue(pairs, ("left_id", "right_id"), issues, "duplicate_pairs", "pairs.csv contains duplicate pair rows.")
    if "label" in pairs.columns:
        labels = pairs["label"]
        missing_labels = int(labels.isna().sum())
        if missing_labels:
            _add_issue(issues, "error", "missing_pair_labels", "pairs.csv contains missing label values.", missing_labels)
        normalized_labels = labels.dropna().map(_coerce_binary_label)
        invalid_count = int(normalized_labels.isna().sum())
        if invalid_count:
            _add_issue(issues, "error", "invalid_pair_labels", "pairs.csv labels must be binary values.", invalid_count)
        valid_labels = normalized_labels.dropna().astype(int)
        positive_count = int(valid_labels.sum())
        negative_count = int(len(valid_labels) - positive_count)
        counts["positive_pairs"] = positive_count
        counts["negative_pairs"] = negative_count
        if len(valid_labels) and (positive_count == 0 or negative_count == 0):
            _add_issue(
                issues,
                "warning",
                "single_class_labels",
                "pairs.csv contains only one label class; threshold tuning and resampling will be limited.",
            )

    if file_ids is None or not {"left_id", "right_id"}.issubset(pairs.columns):
        return
    missing = sorted(
        {
            str(value)
            for value in pd.concat([pairs["left_id"], pairs["right_id"]], ignore_index=True).dropna()
            if str(value) not in file_ids
        }
    )
    if missing:
        _add_issue(
            issues,
            "error",
            "unknown_pair_file_ids",
            f"pairs.csv references unknown file ids: {', '.join(missing[:8])}.",
            len(missing),
        )


def _inspect_retrieval_dataset(root, metadata, counts, issues):
    _inspect_metadata(metadata, RETRIEVAL_KIND, issues)
    files = _read_csv_manifest(root, "files.csv", ("file_id", "file_path"), issues)
    queries = _read_csv_manifest(root, "queries.csv", ("query_id", "file_id"), issues)
    corpus = _read_csv_manifest(root, "corpus.csv", ("document_id", "file_id"), issues)
    qrels = _read_csv_manifest(root, "qrels.csv", ("query_id", "document_id", "relevance"), issues)
    counts.update(
        {
            "files": int(len(files)) if files is not None else 0,
            "queries": int(len(queries)) if queries is not None else 0,
            "documents": int(len(corpus)) if corpus is not None else 0,
            "qrels": int(len(qrels)) if qrels is not None else 0,
        }
    )

    file_ids = _inspect_files_manifest(root, files, counts, issues)
    query_ids = _inspect_id_manifest(queries, "query_id", "queries.csv", issues)
    document_ids = _inspect_id_manifest(corpus, "document_id", "corpus.csv", issues)
    _add_duplicate_issue(qrels, ("query_id", "document_id"), issues, "duplicate_qrels", "qrels.csv contains duplicate query/document judgments.")

    if file_ids is not None and queries is not None and "file_id" in queries:
        missing = sorted({str(value) for value in queries["file_id"].dropna() if str(value) not in file_ids})
        if missing:
            _add_issue(
                issues,
                "error",
                "unknown_query_file_ids",
                f"queries.csv references unknown file ids: {', '.join(missing[:8])}.",
                len(missing),
            )
    if file_ids is not None and corpus is not None and "file_id" in corpus:
        missing = sorted({str(value) for value in corpus["file_id"].dropna() if str(value) not in file_ids})
        if missing:
            _add_issue(
                issues,
                "error",
                "unknown_corpus_file_ids",
                f"corpus.csv references unknown file ids: {', '.join(missing[:8])}.",
                len(missing),
            )
    if qrels is not None:
        if query_ids is not None and "query_id" in qrels:
            missing = sorted({str(value) for value in qrels["query_id"].dropna() if str(value) not in query_ids})
            if missing:
                _add_issue(
                    issues,
                    "error",
                    "unknown_qrel_query_ids",
                    f"qrels.csv references unknown query ids: {', '.join(missing[:8])}.",
                    len(missing),
                )
        if document_ids is not None and "document_id" in qrels:
            missing = sorted({str(value) for value in qrels["document_id"].dropna() if str(value) not in document_ids})
            if missing:
                _add_issue(
                    issues,
                    "error",
                    "unknown_qrel_document_ids",
                    f"qrels.csv references unknown document ids: {', '.join(missing[:8])}.",
                    len(missing),
                )
        if "relevance" in qrels:
            relevance = pd.to_numeric(qrels["relevance"], errors="coerce")
            invalid_count = int(relevance.isna().sum() + (relevance < 0).sum())
            if invalid_count:
                _add_issue(
                    issues,
                    "error",
                    "invalid_qrel_relevance",
                    "qrels.csv relevance values must be non-negative numbers.",
                    invalid_count,
                )
            counts["positive_qrels"] = int((relevance > 0).sum())
            if len(qrels) and int((relevance > 0).sum()) == 0:
                _add_issue(
                    issues,
                    "warning",
                    "no_positive_qrels",
                    "qrels.csv does not contain any positive relevance judgments.",
                )


def _read_metadata(root, issues):
    path = root / "metadata.json"
    if not path.exists():
        _add_issue(issues, "error", "missing_metadata", "metadata.json is missing.")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _add_issue(issues, "error", "invalid_metadata_json", f"metadata.json is not valid JSON: {exc.msg}.")
        return {}
    if not isinstance(payload, dict):
        _add_issue(issues, "error", "invalid_metadata", "metadata.json must contain a JSON object.")
        return {}
    return payload


def _resolve_dataset_kind(root, requested_kind, metadata, issues):
    requested = str(requested_kind or "auto").strip().lower()
    if requested in {"pair", "pairs", PAIR_KIND, "pair-classification"}:
        return PAIR_KIND
    if requested in {"retrieval", "ranking"}:
        return RETRIEVAL_KIND
    if requested not in {"auto", ""}:
        _add_issue(issues, "error", "invalid_requested_kind", f"Unsupported dataset kind: {requested_kind}.")
        return None

    metadata_kind = str(metadata.get("dataset_kind") or "").strip().lower()
    has_pair_files = (root / "files.csv").exists() and (root / "pairs.csv").exists()
    has_retrieval_files = all((root / name).exists() for name in ("files.csv", "queries.csv", "corpus.csv", "qrels.csv"))
    if metadata_kind == PAIR_KIND:
        return PAIR_KIND
    if metadata_kind == RETRIEVAL_KIND:
        return RETRIEVAL_KIND
    if has_pair_files and has_retrieval_files:
        _add_issue(issues, "error", "ambiguous_dataset_kind", "Dataset contains both pair and retrieval manifests.")
        return None
    if has_pair_files:
        return PAIR_KIND
    if has_retrieval_files:
        return RETRIEVAL_KIND
    return None


def _inspect_metadata(metadata, expected_kind, issues):
    if not metadata:
        return
    dataset_kind = str(metadata.get("dataset_kind") or "").strip()
    if dataset_kind and dataset_kind != expected_kind:
        _add_issue(
            issues,
            "error",
            "metadata_kind_mismatch",
            f"metadata.json dataset_kind is {dataset_kind!r}, expected {expected_kind!r}.",
        )
    if not str(metadata.get("name") or "").strip():
        _add_issue(issues, "warning", "missing_dataset_name", "metadata.json does not include a dataset name.")
    if not str(metadata.get("task_type") or "").strip():
        _add_issue(issues, "warning", "missing_task_type", "metadata.json does not include task_type.")


def _read_csv_manifest(root, filename, required_columns, issues):
    path = root / filename
    if not path.exists():
        _add_issue(issues, "error", "missing_manifest", f"Required manifest is missing: {filename}.")
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        expected = ", ".join(required_columns)
        _add_issue(
            issues,
            "error",
            "empty_csv",
            f"{filename} is empty; expected columns: {expected}.",
        )
        return None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        _add_issue(issues, "error", "invalid_csv", f"{filename} could not be read as CSV: {exc}.")
        return None
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        _add_issue(
            issues,
            "error",
            "missing_columns",
            f"{filename} is missing required columns: {', '.join(missing)}.",
            len(missing),
        )
        return frame
    if frame.empty:
        _add_issue(issues, "warning", "empty_manifest", f"{filename} has no rows.")
    return frame


def _inspect_files_manifest(root, files, counts, issues):
    if files is None or not {"file_id", "file_path"}.issubset(files.columns):
        return None
    file_ids = _inspect_id_manifest(files, "file_id", "files.csv", issues)
    missing_paths = int(files["file_path"].isna().sum())
    if missing_paths:
        _add_issue(issues, "error", "missing_file_paths", "files.csv contains missing file_path values.", missing_paths)

    missing_files = []
    unsafe_paths = []
    non_file_paths = []
    empty_files = []
    for row in files.to_dict(orient="records"):
        raw_path = row.get("file_path")
        if pd.isna(raw_path):
            continue
        if is_unsafe_relative_path(raw_path):
            unsafe_paths.append(str(raw_path))
            continue
        try:
            target = resolve_relative_path_within_root(root, raw_path)
        except ValueError:
            unsafe_paths.append(str(raw_path))
            continue
        if not target.exists():
            missing_files.append(str(raw_path))
            continue
        if not target.is_file():
            non_file_paths.append(str(raw_path))
            continue
        try:
            if not target.read_text(encoding="utf-8", errors="ignore").strip():
                empty_files.append(str(raw_path))
        except OSError:
            missing_files.append(str(raw_path))

    if unsafe_paths:
        _add_issue(
            issues,
            "error",
            "unsafe_file_paths",
            f"files.csv contains unsafe relative paths: {', '.join(unsafe_paths[:8])}.",
            len(unsafe_paths),
        )
    if missing_files:
        _add_issue(
            issues,
            "error",
            "missing_files",
            f"files.csv references missing files: {', '.join(missing_files[:8])}.",
            len(missing_files),
        )
    if non_file_paths:
        _add_issue(
            issues,
            "error",
            "non_file_paths",
            f"files.csv references paths that are not files: {', '.join(non_file_paths[:8])}.",
            len(non_file_paths),
        )
    if empty_files:
        _add_issue(
            issues,
            "warning",
            "empty_files",
            f"files.csv references empty text files: {', '.join(empty_files[:8])}.",
            len(empty_files),
        )
    counts["empty_files"] = len(empty_files)
    return file_ids


def _inspect_id_manifest(frame, id_column, filename, issues):
    if frame is None or id_column not in frame.columns:
        return None
    ids = frame[id_column]
    missing = int(ids.isna().sum())
    if missing:
        _add_issue(issues, "error", "missing_ids", f"{filename} contains missing {id_column} values.", missing)
    string_ids = ids.dropna().map(str)
    blank = int((string_ids.str.strip() == "").sum())
    if blank:
        _add_issue(issues, "error", "blank_ids", f"{filename} contains blank {id_column} values.", blank)
    duplicates = string_ids[string_ids.duplicated()].tolist()
    if duplicates:
        _add_issue(
            issues,
            "error",
            "duplicate_ids",
            f"{filename} contains duplicate {id_column} values: {', '.join(sorted(set(duplicates))[:8])}.",
            len(set(duplicates)),
        )
    return set(value for value in string_ids.tolist() if value.strip())


def _add_duplicate_issue(frame, columns, issues, code, message):
    if frame is None or not set(columns).issubset(frame.columns):
        return
    duplicate_count = int(frame.duplicated(subset=list(columns)).sum())
    if duplicate_count:
        _add_issue(issues, "warning", code, message, duplicate_count)


def _coerce_binary_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "p",
        "positive",
        "cheating",
        "clone",
        "plagiarized",
        "plagiarised",
        "plagiarism",
        "match",
        "same",
    }:
        return 1
    if text in {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "negative",
        "original",
        "nonplagiarized",
        "nonplagiarised",
        "non_plagiarized",
        "non_plagiarised",
        "not plagiarized",
        "not plagiarised",
        "not_plagiarized",
        "not_plagiarised",
        "non_plagiarism",
        "non-plagiarism",
        "nonmatch",
        "non-match",
        "different",
        "np",
    }:
        return 0
    return None


def _add_issue(issues, severity, code, message, count=1):
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "count": int(count),
        }
    )


def _write_issues_csv(issues, path):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("severity", "code", "message", "count"))
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "severity": issue.get("severity", ""),
                    "code": issue.get("code", ""),
                    "message": issue.get("message", ""),
                    "count": issue.get("count", 1),
                }
            )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_artifact_basename(value):
    text = str(value or "").strip()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    return safe.strip("._") or "dataset_validation"
