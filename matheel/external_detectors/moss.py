"""Read saved MOSS HTML reports locally, keeping unreported pairs sparse."""

import csv
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from ..clone_discovery import SourceSpan
from ._native import MAX_REPORT_BYTES, native_id, read_report
from .common import DetectorError

_PERCENT = re.compile(r"(.+?)\s*\(([0-9]{1,3})\s*%\)\s*", re.DOTALL)
_DETAIL = re.compile(r"(?:^|/)match([0-9]+)\.html$", re.IGNORECASE)
_RANGE = re.compile(r"([0-9]+)\s*(?:[-–]\s*([0-9]+))?")


@dataclass(frozen=True)
class MossMatch:
    """One reported pair with native directional percentages and optional passages."""

    left_id: str
    right_id: str
    left_percent: int
    right_percent: int
    matched_lines: int
    passages: tuple[tuple[SourceSpan, SourceSpan], ...] = ()
    detail_status: str = "unavailable"

    @property
    def mean_similarity(self):
        """Mean native percentage expressed as a fraction in the interval [0, 1]."""
        return (self.left_percent + self.right_percent) / 200


@dataclass(frozen=True)
class MossReport:
    matches: tuple[MossMatch, ...]
    metadata: dict


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.title = ""
        self._in_title = False
        self._depth = 0
        self._row = None
        self._cell = None

    def _finish_row(self):
        if self._row:
            self.rows.append(tuple((" ".join("".join(parts).split()), href) for parts, href in self._row))
        self._row = self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "table":
            self._depth += 1
        elif self._depth == 1:
            if tag == "tr":
                self._finish_row()
                self._row = []
            elif tag in {"td", "th"} and self._row is not None:
                self._cell = [[], None]
                self._row.append(self._cell)
            elif tag == "a" and self._cell is not None and self._cell[1] is None:
                self._cell[1] = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "table":
            if self._depth == 1:
                self._finish_row()
            self._depth = max(0, self._depth - 1)
        elif self._depth == 1:
            if tag == "tr":
                self._finish_row()
            elif tag in {"td", "th"}:
                self._cell = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._depth == 1 and self._cell is not None:
            self._cell[0].append(data)

    def close(self):
        super().close()
        self._finish_row()


def _parse_html(path, max_report_bytes):
    content, digest = read_report(path, max_report_bytes)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise DetectorError("Saved MOSS HTML must use UTF-8 encoding.") from exc
    parser = _TableParser()
    parser.feed(text)
    parser.close()
    return parser, digest


def _match_id(href):
    if not href:
        return None
    try:
        match = _DETAIL.search(urlsplit(href).path)
    except ValueError as exc:
        raise DetectorError("Malformed pair link in saved MOSS report.") from exc
    return match[1] if match else None


def _percentage(cell, source_map):
    found = _PERCENT.fullmatch(cell)
    if found is None or int(found[2]) > 100:
        raise DetectorError("MOSS row is missing a valid directional percentage.")
    label = found[1].strip().replace("\\", "/")
    identifier = native_id(label, source_map)
    # A trailing slash is the native directory-mode marker, not part of an ID.
    return SourceSpan(identifier.rstrip("/"), 1, 1).file_id, int(found[2])


def _passages(path, left, right, max_report_bytes):
    parser, digest = _parse_html(path, max_report_bytes)
    passages = []
    for row in parser.rows:
        ranges = [found for text, _ in row if (found := _RANGE.fullmatch(text)) is not None]
        if not ranges:
            continue
        if len(ranges) != 2:
            raise DetectorError("Malformed line ranges in saved MOSS match details.")
        spans = tuple(SourceSpan(file_id, match[1], match[2] or match[1])
                      for file_id, match in zip((left, right), ranges))
        passages.append(spans)
    if not passages:
        raise DetectorError("Saved MOSS match details do not contain any paired line ranges.")
    return tuple(passages), digest


def parse_report(path, source_map=None, *, match_directory=None, max_report_bytes=MAX_REPORT_BYTES):
    """Import a saved MOSS index and optional local ``matchN-top.html`` tables.

    Source mapping keys are the native labels as printed, including any trailing
    slash for a submission directory. IDs are portable corpus-relative values.
    Report links are only used to identify local passage tables, never fetched.
    """
    parser, report_hash = _parse_html(path, max_report_bytes)
    if "moss results" not in " ".join(parser.title.lower().split()):
        raise DetectorError("Expected a saved MOSS results index.")
    directory = Path(match_directory).expanduser().resolve() if match_directory is not None else None
    if directory is not None and not directory.is_dir():
        raise DetectorError("MOSS match_directory must be an existing local directory.")
    matches, detail_hashes, seen = [], {}, set()
    has_header = False
    try:
        for row in parser.rows:
            labels = [text for text, _ in row]
            if tuple(text.lower() for text in labels) == ("file 1", "file 2", "lines matched"):
                has_header = True
                continue
            if not has_header:
                continue
            if len(row) != 3 or re.fullmatch(r"[0-9]+", labels[-1]) is None:
                raise DetectorError("MOSS index contains a malformed match row.")
            left, left_percent = _percentage(labels[0], source_map)
            right, right_percent = _percentage(labels[1], source_map)
            pair = tuple(sorted((left, right)))
            if left == right or pair in seen:
                raise DetectorError("MOSS index contains a self pair or repeated reported pair.")
            seen.add(pair)
            left_page, right_page = _match_id(row[0][1]), _match_id(row[1][1])
            if left_page != right_page:
                raise DetectorError("MOSS index row refers to different match details for each side.")
            passages, status = (), "unavailable"
            if directory is not None and left_page is not None:
                detail = directory / f"match{left_page}-top.html"
                if detail.is_file():
                    if detail.is_symlink():
                        raise DetectorError("Saved MOSS match details must not be symbolic links.")
                    passages, detail_hashes[detail.name] = _passages(detail, left, right, max_report_bytes)
                    status = "parsed"
            matches.append(MossMatch(left, right, left_percent, right_percent, int(labels[-1]), passages, status))
    except ValueError as exc:
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError("Saved MOSS report contains an invalid mapped source or passage.") from exc
    if not has_header:
        raise DetectorError("Saved MOSS index is missing its match table header.")
    return MossReport(tuple(matches), {
        "detector": "moss", "format": "saved_html", "report_sha256": report_hash,
        "reported_pairs": len(matches), "pair_coverage": "reported_only", "detail_sha256": detail_hashes,
        "pairs_with_details": len(detail_hashes),
    })


def compare_coverage(report, expected_pairs):
    """Count matches within an explicitly declared pair universe."""
    expected = {tuple(sorted((left, right))) for left, right in expected_pairs}
    if any(left == right for left, right in expected):
        raise ValueError("The expected MOSS pair universe must exclude self pairs.")
    reported = {tuple(sorted((pair.left_id, pair.right_id))) for pair in report.matches}
    covered = len(expected & reported)
    return {
        "expected_pairs": len(expected), "reported_expected_pairs": covered,
        "missing_expected_pairs": len(expected) - covered, "reported_other_pairs": len(reported - expected),
    }


def write_moss_matches(path, report):
    """Write reported pairs only, with native percentages and optional line passages."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("left_id", "right_id", "left_percent", "right_percent", "matched_lines", "detail_status", "passages"))
        for match in report.matches:
            passages = [[left.start_line, left.end_line, right.start_line, right.end_line] for left, right in match.passages]
            writer.writerow((match.left_id, match.right_id, match.left_percent, match.right_percent,
                             match.matched_lines, match.detail_status, json.dumps(passages, separators=(",", ":"))))
    return path
