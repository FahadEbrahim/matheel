"""Import saved SourcererCC pair IDs with a separately recorded source map."""

import csv
import hashlib
import io
import json
from collections.abc import Mapping

from ..clone_discovery import CloneDetectorReport, CloneFinding, SourceSpan
from ._native import MAX_REPORT_BYTES, read_report
from .common import DetectorError, checked_score

_MAPPING_COLUMNS = ("native_id", "file_id", "start_line", "end_line")


def _identifier(value):
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise DetectorError("SourcererCC IDs must be non-empty strings or integers.")
    key = str(value).strip()
    if not key or any(ord(char) < 32 for char in key):
        raise DetectorError("SourcererCC IDs must be non-empty strings or integers.")
    return key


def _source_mapping(source_map, max_report_bytes):
    if isinstance(source_map, Mapping):
        rows = source_map.items()
    else:
        data, _ = read_report(source_map, max_report_bytes)
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if set(reader.fieldnames or ()) != set(_MAPPING_COLUMNS) or len(reader.fieldnames) != len(_MAPPING_COLUMNS):
            raise DetectorError("SourcererCC source mapping needs native_id, file_id, start_line, and end_line.")
        rows = (
            (row["native_id"], SourceSpan(row["file_id"], row["start_line"], row["end_line"]))
            for row in reader
        )
    output = {}
    for key, value in rows:
        key = _identifier(key)
        if key in output:
            raise DetectorError("Duplicate or ambiguous SourcererCC native ID in source mapping.")
        if not isinstance(value, SourceSpan):
            raise DetectorError("SourcererCC source mapping values must be SourceSpan instances.")
        output[key] = value
    if not output:
        raise DetectorError("SourcererCC source mapping must contain at least one source.")
    return output


def parse_report(
    path, source_map, *, max_report_bytes=MAX_REPORT_BYTES, version=None, threshold=None,
    granularity="file",
):
    """Read native two-column pair IDs, preserving their original mapped coordinates.

    File-level native reports should map each ID to the complete original file.
    Use function or block granularity only when the supplied map records actual
    coordinate ranges from that native tokenizer.
    """
    if granularity not in {"file", "function", "block"}:
        raise DetectorError("SourcererCC granularity must be file, function, or block.")
    if threshold is not None:
        threshold = checked_score(threshold)
    try:
        sources = _source_mapping(source_map, max_report_bytes)
        data, report_hash = read_report(path, max_report_bytes)
        findings = []
        seen = set()
        duplicates = 0
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise DetectorError("SourcererCC pair rows must contain exactly two native IDs.")
            left_id, right_id = map(_identifier, row)
            try:
                left, right = sources[left_id], sources[right_id]
            except KeyError as exc:
                raise DetectorError("SourcererCC pair references a native ID absent from the source map.") from exc
            pair = tuple(sorted((left, right)))
            if pair not in seen:
                findings.append(CloneFinding(left, right, group_id=f"sourcerercc-{reader.line_num}",
                                            left_native_id=left_id, right_native_id=right_id))
                seen.add(pair)
            else:
                duplicates += 1
    except (csv.Error, UnicodeError, ValueError) as exc:
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError("Invalid SourcererCC report or source-span mapping.") from exc
    mapping = [[key, span.file_id, span.start_line, span.end_line] for key, span in sorted(sources.items())]
    return CloneDetectorReport(tuple(findings), {
        "detector": "sourcerercc", "format": "native_pair_ids", "version": version,
        "granularity": granularity, "threshold": threshold, "mapped_sources": len(sources),
        "source_mapping_sha256": hashlib.sha256(json.dumps(mapping).encode("utf-8")).hexdigest(),
        "report_sha256": report_hash, "reported_pairs": len(findings), "duplicate_pair_rows": duplicates,
    })
