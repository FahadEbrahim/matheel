"""Source-span pair manifests and recall for annotated clone references."""

import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


CLONE_TYPES = (
    "type1",
    "type2",
    "type2_consistent",
    "type2_blind",
    "type3",
    "type3_very_strong",
    "type3_strong",
    "type3_moderate",
    "weak_type3_type4",
    "type4",
)
PROJECT_SCOPES = ("intra_project", "inter_project", "unknown")
SPAN_COLUMNS = (
    "left_file_id", "left_start_line", "left_end_line",
    "right_file_id", "right_start_line", "right_end_line",
)
FINDING_COLUMNS = (*SPAN_COLUMNS, "score", "matched_tokens", "group_id", "left_native_id", "right_native_id")
REFERENCE_COLUMNS = (
    *SPAN_COLUMNS, "clone_type", "project_scope", "pretty_lines", "tokens",
    "line_similarity", "token_similarity", "reference_id", "original_lines",
)


def _file_id(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("file_id must be a non-empty relative identifier.")
    path = PurePosixPath(value)
    if (
        path.as_posix() != value or path.is_absolute() or path.name in {"", ".", ".."}
        or ".." in path.parts or "\\" in value or ":" in path.parts[0]
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("file_id must be a portable corpus-relative identifier.")
    return value


def _integer(value, name):
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _fraction(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a number between zero and one.") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{name} must be a number between zero and one.")
    return parsed


def _clone_type(value):
    if not isinstance(value, str):
        raise ValueError("clone_type must declare a registered reference clone type.")
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if key not in CLONE_TYPES:
        raise ValueError("clone_type must declare a registered reference clone type.")
    return key


@dataclass(frozen=True, order=True)
class SourceSpan:
    """An original source range using positive, one-based inclusive lines."""

    file_id: str
    start_line: int
    end_line: int

    def __post_init__(self):
        _file_id(self.file_id)
        object.__setattr__(self, "start_line", _integer(self.start_line, "start_line"))
        object.__setattr__(self, "end_line", _integer(self.end_line, "end_line"))
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line.")

    @property
    def line_count(self):
        return self.end_line - self.start_line + 1


def _check_span_pair(left, right):
    if not isinstance(left, SourceSpan) or not isinstance(right, SourceSpan):
        raise TypeError("Both sides must be SourceSpan instances.")
    if left == right:
        raise ValueError("A clone pair needs two distinct source spans.")


@dataclass(frozen=True)
class CloneFinding:
    """A reported clone pair whose optional evidence retains the tool's meaning."""

    left: SourceSpan
    right: SourceSpan
    score: float | None = None
    matched_tokens: int | None = None
    group_id: str | None = None
    left_native_id: str | None = None
    right_native_id: str | None = None

    def __post_init__(self):
        _check_span_pair(self.left, self.right)
        if self.score is not None:
            object.__setattr__(self, "score", _fraction(self.score, "score"))
        if self.matched_tokens is not None:
            object.__setattr__(self, "matched_tokens", _integer(self.matched_tokens, "matched_tokens"))
        for field in ("group_id", "left_native_id", "right_native_id"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be a non-empty string when supplied.")


@dataclass(frozen=True)
class CloneReference:
    """A judged clone with a type and optional measurements from its reference corpus."""

    left: SourceSpan
    right: SourceSpan
    clone_type: str
    project_scope: str = "unknown"
    pretty_lines: int | None = None
    tokens: int | None = None
    line_similarity: float | None = None
    token_similarity: float | None = None
    reference_id: str | None = None
    original_lines: int | None = None

    def __post_init__(self):
        _check_span_pair(self.left, self.right)
        object.__setattr__(self, "clone_type", _clone_type(self.clone_type))
        if self.project_scope not in PROJECT_SCOPES:
            raise ValueError("project_scope must be intra_project, inter_project, or unknown.")
        if self.reference_id is not None and (not isinstance(self.reference_id, str) or not self.reference_id.strip()):
            raise ValueError("reference_id must be a non-empty string when supplied.")
        for field in ("pretty_lines", "tokens", "original_lines"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _integer(value, field))
        for field in ("line_similarity", "token_similarity"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _fraction(value, field))


@dataclass(frozen=True)
class CloneDetectorReport:
    """Native detector findings with portable information about report scope."""

    findings: tuple[CloneFinding, ...]
    metadata: dict


def _csv_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(required).issubset(fields):
            raise ValueError("Unexpected clone discovery CSV columns.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed clone discovery CSV row {reader.line_num}.")
            yield row


def _row_span(row, side):
    return SourceSpan(row[f"{side}_file_id"], row[f"{side}_start_line"], row[f"{side}_end_line"])


def load_clone_findings(path) -> Iterator[CloneFinding]:
    """Read normalized findings without materializing the CSV in memory."""
    for row in _csv_rows(path, SPAN_COLUMNS):
        yield CloneFinding(
            _row_span(row, "left"), _row_span(row, "right"),
            score=row.get("score") or None, matched_tokens=row.get("matched_tokens") or None,
            group_id=row.get("group_id") or None, left_native_id=row.get("left_native_id") or None,
            right_native_id=row.get("right_native_id") or None,
        )


def load_clone_references(path) -> Iterator[CloneReference]:
    """Read typed clone references from the normalized manifest."""
    for row in _csv_rows(path, (*SPAN_COLUMNS, "clone_type")):
        yield CloneReference(
            _row_span(row, "left"), _row_span(row, "right"), row["clone_type"],
            project_scope=row.get("project_scope") or "unknown", pretty_lines=row.get("pretty_lines") or None,
            tokens=row.get("tokens") or None, line_similarity=row.get("line_similarity") or None,
            token_similarity=row.get("token_similarity") or None, reference_id=row.get("reference_id") or None,
            original_lines=row.get("original_lines") or None,
        )


def _span_columns(span):
    return span.file_id, span.start_line, span.end_line


def write_clone_findings(path, findings: Iterable[CloneFinding]):
    """Write a normalized source-span CSV without source text."""
    file = Path(path)
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FINDING_COLUMNS)
        for finding in findings:
            writer.writerow((
                *_span_columns(finding.left), *_span_columns(finding.right),
                finding.score, finding.matched_tokens, finding.group_id, finding.left_native_id, finding.right_native_id,
            ))
    return file


def write_clone_references(path, references: Iterable[CloneReference]):
    """Write a typed reference manifest with its optional original measurements."""
    file = Path(path)
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(REFERENCE_COLUMNS)
        for reference in references:
            writer.writerow((
                *_span_columns(reference.left), *_span_columns(reference.right), reference.clone_type,
                reference.project_scope, reference.pretty_lines, reference.tokens,
                reference.line_similarity, reference.token_similarity, reference.reference_id, reference.original_lines,
            ))
    return file


def _bce_span(span, source_map):
    source = span.file_id if source_map is None else source_map.get(span.file_id)
    if source is None:
        raise ValueError("BigCloneEval source mapping does not cover a finding.")
    parts = _file_id(source).split("/")
    if len(parts) != 2 or parts[0] not in {"default", "sample", "selected"} or "," in parts[1]:
        raise ValueError("BigCloneEval sources require an IJaDataset subdirectory and filename.")
    return *parts, span.start_line, span.end_line


def write_bigcloneeval_findings(path, findings: Iterable[CloneFinding], source_map: Mapping | None = None):
    """Export headerless eight-column CSV accepted by BigCloneEval importClones."""
    file = Path(path)
    seen = set()
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for finding in findings:
            pair = tuple(sorted((_bce_span(finding.left, source_map), _bce_span(finding.right, source_map))))
            if pair[0] == pair[1]:
                raise ValueError("The BigCloneEval source mapping collapses distinct finding spans.")
            if pair not in seen:
                writer.writerow((*pair[0], *pair[1]))
                seen.add(pair)
    return file


def _overlap(reference, candidate, threshold):
    overlap = min(reference.end_line, candidate.end_line) - max(reference.start_line, candidate.start_line) + 1
    return overlap / reference.line_count >= threshold


def _detected(left, right, candidates, threshold):
    for candidate_left, candidate_right in candidates:
        if _overlap(left, candidate_left, threshold) and _overlap(right, candidate_right, threshold):
            return True
        if left.file_id == right.file_id and _overlap(left, candidate_right, threshold) and _overlap(
            right, candidate_left, threshold
        ):
            return True
    return False


def _eligible(reference, settings):
    original_lines = reference.original_lines or min(reference.left.line_count, reference.right.line_count)
    if original_lines < settings["minimum_lines"]:
        return False
    for field, minimum in (("pretty_lines", settings["minimum_pretty_lines"]), ("tokens", settings["minimum_tokens"])):
        value = getattr(reference, field)
        if minimum is not None:
            if value is None:
                raise ValueError(f"A clone reference is missing {field} required by the size filter.")
            if value < minimum:
                return False
    minimum_similarity = settings["minimum_syntax_similarity"]
    if minimum_similarity is None:
        return True
    mode = settings["similarity_type"]
    if mode == "both":
        if reference.line_similarity is None or reference.token_similarity is None:
            raise ValueError("The BOTH similarity filter requires line and token measurements.")
        value = min(reference.line_similarity, reference.token_similarity)
    else:
        value = getattr(reference, f"{mode}_similarity")
        if value is None:
            raise ValueError(f"The similarity filter requires a {mode} measurement.")
    return value >= minimum_similarity


def _metric(matched, eligible):
    return {"matched": matched, "eligible": eligible, "recall": matched / eligible if eligible else None}


def evaluate_clone_discovery(
    references: Iterable[CloneReference],
    findings: Iterable[CloneFinding],
    *,
    coverage_threshold=0.7,
    minimum_lines=1,
    minimum_pretty_lines=None,
    minimum_tokens=None,
    similarity_type="both",
    minimum_syntax_similarity=None,
):
    """Measure recall using per-fragment inclusive coverage against known positives.

    Findings are matched as unordered span pairs. Duplicate findings count once.
    Project and type strata come from source reference annotations. A zero-size
    stratum has no recall. This evaluator never scores unrestricted precision.
    """
    coverage_threshold = _fraction(coverage_threshold, "coverage_threshold")
    if coverage_threshold == 0:
        raise ValueError("coverage_threshold must be greater than zero.")
    if similarity_type not in {"both", "line", "token"}:
        raise ValueError("similarity_type must be both, line, or token.")
    settings = {"coverage_threshold": coverage_threshold, "similarity_type": similarity_type}
    for name, value in (
        ("minimum_lines", minimum_lines), ("minimum_pretty_lines", minimum_pretty_lines),
        ("minimum_tokens", minimum_tokens),
    ):
        settings[name] = _integer(value, name) if value is not None or name == "minimum_lines" else None
    settings["minimum_syntax_similarity"] = (
        None if minimum_syntax_similarity is None
        else _fraction(minimum_syntax_similarity, "minimum_syntax_similarity")
    )

    index = defaultdict(set)
    for finding in findings:
        left, right = sorted((finding.left, finding.right))
        index[(left.file_id, right.file_id)].add((left, right))
    eligible, matched = Counter(), Counter()
    seen = set()
    reference_count = 0
    for reference in references:
        left, right = sorted((reference.left, reference.right))
        pair = reference.reference_id if reference.reference_id is not None else (left, right)
        if pair in seen:
            raise ValueError("Clone reference manifest contains a repeated reference ID or unidentified span pair.")
        seen.add(pair)
        reference_count += 1
        if not _eligible(reference, settings):
            continue
        keys = (
            ("all", "all"), (reference.clone_type, "all"), ("all", reference.project_scope),
            (reference.clone_type, reference.project_scope),
        )
        if reference.clone_type in {"type2_consistent", "type2_blind"}:
            keys += (("type2", "all"), ("type2", reference.project_scope))
        eligible.update(keys)
        if _detected(left, right, index.get((left.file_id, right.file_id), ()), coverage_threshold):
            matched.update(keys)

    def row(clone_type, scope):
        return {"clone_type": clone_type, "project_scope": scope, **_metric(
            matched[(clone_type, scope)], eligible[(clone_type, scope)],
        )}

    return {
        "schema_version": 1,
        "protocol": {"name": "clone_span_recall", **settings},
        "reported_pairs": sum(map(len, index.values())),
        "reference_pairs": reference_count,
        "overall": _metric(matched[("all", "all")], eligible[("all", "all")]),
        "strata": [
            row(clone_type, scope) for clone_type in ("all", *CLONE_TYPES) for scope in ("all", *PROJECT_SCOPES)
            if (clone_type, scope) != ("all", "all")
        ],
    }
