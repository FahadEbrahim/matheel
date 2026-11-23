"""Import saved SourcererCC pair IDs with a separately recorded source map."""

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping

from ..clone_discovery import CloneDetectorReport, CloneFinding, SourceSpan
from ._native import MAX_REPORT_BYTES, read_report
from .common import DetectorError, checked_score, positive_number

_MAPPING_COLUMNS = ("native_id", "file_id", "start_line", "end_line")
_PROJECT_MAPPING_COLUMNS = (*_MAPPING_COLUMNS, "native_project_id")


def _identifier(value):
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise DetectorError("SourcererCC IDs must be non-empty strings or integers.")
    key = str(value).strip()
    if not key or any(ord(char) < 32 for char in key):
        raise DetectorError("SourcererCC IDs must be non-empty strings or integers.")
    return key


def _project_block_id(values):
    if len(values) != 2:
        raise DetectorError("SourcererCC project and block identifiers require two values.")
    identifiers = tuple(_identifier(value) for value in values)
    if any(not (value.isascii() and value.isdecimal()) for value in identifiers):
        raise DetectorError("SourcererCC project and block identifiers must be decimal IDs.")
    return identifiers


def _mapping_rows(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True)
    columns = tuple(reader.fieldnames or ())
    valid_columns = (_MAPPING_COLUMNS, _PROJECT_MAPPING_COLUMNS)
    if not any(len(columns) == len(expected) and set(columns) == set(expected) for expected in valid_columns):
        raise DetectorError(
            "SourcererCC source mapping needs native_id, file_id, start_line, and end_line, "
            "with native_project_id for project/block reports."
        )
    for row in reader:
        if None in row or None in row.values():
            raise DetectorError("SourcererCC source mapping has an incomplete or extra field.")
        native_id = row["native_id"]
        if "native_project_id" in row:
            native_id = (row["native_project_id"], native_id)
        yield native_id, SourceSpan(row["file_id"], row["start_line"], row["end_line"])


def _source_mapping(source_map, max_report_bytes):
    if isinstance(source_map, Mapping):
        rows = source_map.items()
    else:
        data, _ = read_report(source_map, max_report_bytes)
        rows = _mapping_rows(data)
    output = {}
    key_size = None
    for key, value in rows:
        key = _project_block_id(key) if isinstance(key, tuple) else (_identifier(key),)
        if key_size is not None and len(key) != key_size:
            raise DetectorError("Use one SourcererCC identifier layout per source mapping.")
        key_size = len(key)
        if key in output:
            raise DetectorError("Duplicate or ambiguous SourcererCC native ID in source mapping.")
        if not isinstance(value, SourceSpan):
            raise DetectorError("SourcererCC source mapping values must be SourceSpan instances.")
        output[key] = value
    if not output:
        raise DetectorError("SourcererCC source mapping must contain at least one source.")
    return output, key_size


def parse_report(
    path, source_map, *, max_report_bytes=MAX_REPORT_BYTES, version=None, threshold=None,
    granularity="file", minimum_tokens=None, corpus_sha256=None,
):
    """Read native pair IDs with their original mapped coordinates.

    Two-column reports use scalar IDs. Four-column reports use project ID and
    block ID at each endpoint. Supply tuple keys or a native_project_id column
    for that layout. The normalized native IDs then use project_id:block_id.
    Map file-level reports to complete files. Supply verified original source
    ranges for function or block reports.
    """
    if granularity not in {"file", "function", "block"}:
        raise DetectorError("SourcererCC granularity must be file, function, or block.")
    if threshold is not None:
        threshold = checked_score(threshold)
    provenance = {}
    if minimum_tokens is not None:
        provenance["minimum_tokens"] = positive_number(minimum_tokens, "minimum_tokens", integer=True)
    if corpus_sha256 is not None:
        if not isinstance(corpus_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_sha256):
            raise DetectorError("SourcererCC corpus_sha256 must be a lowercase SHA-256 hex digest.")
        provenance["corpus_sha256"] = corpus_sha256
    try:
        sources, key_size = _source_mapping(source_map, max_report_bytes)
        data, report_hash = read_report(path, max_report_bytes)
        findings = []
        seen = set()
        duplicates = 0
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True)
        for row in reader:
            if not row:
                continue
            if key_size == 1 and len(row) != 2:
                raise DetectorError("SourcererCC pair rows must contain exactly two native IDs.")
            if key_size == 2 and len(row) != 4:
                raise DetectorError("SourcererCC project/block rows must contain exactly four native IDs.")
            if key_size == 1:
                left_id, right_id = ((_identifier(value),) for value in row)
            else:
                left_id, right_id = _project_block_id(row[:2]), _project_block_id(row[2:])
            try:
                left, right = sources[left_id], sources[right_id]
            except KeyError as exc:
                raise DetectorError("SourcererCC pair references a native ID absent from the source map.") from exc
            pair = tuple(sorted((left, right)))
            if pair not in seen:
                findings.append(CloneFinding(left, right, group_id=f"sourcerercc-{reader.line_num}",
                                            left_native_id=":".join(left_id), right_native_id=":".join(right_id)))
                seen.add(pair)
            else:
                duplicates += 1
    except (csv.Error, UnicodeError, ValueError) as exc:
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError("Invalid SourcererCC report or source-span mapping.") from exc
    mapping = [[":".join(key), span.file_id, span.start_line, span.end_line] for key, span in sorted(sources.items())]
    return CloneDetectorReport(tuple(findings), {
        "detector": "sourcerercc", "format": "native_project_block_ids" if key_size == 2 else "native_pair_ids",
        "version": version,
        "granularity": granularity, "threshold": threshold, "mapped_sources": len(sources),
        "source_mapping_sha256": hashlib.sha256(json.dumps(mapping).encode("utf-8")).hexdigest(),
        "report_sha256": report_hash, "reported_pairs": len(findings), "duplicate_pair_rows": duplicates,
        **provenance,
    })
