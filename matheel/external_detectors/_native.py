"""Local artifact reading and staging for span-based detector adapters."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..clone_discovery import SourceSpan
from .common import DetectorError, positive_number

MAX_REPORT_BYTES = 100_000_000


def read_report(path, max_report_bytes):
    limit = positive_number(max_report_bytes, "max_report_bytes", integer=True)
    try:
        with Path(path).open("rb") as stream:
            content = stream.read(limit + 1)
    except OSError as exc:
        raise DetectorError("Could not read the specified local detector report.") from exc
    if len(content) > limit:
        raise DetectorError("Detector report exceeds max_report_bytes.")
    return content, hashlib.sha256(content).hexdigest()


def read_xml_report(path, max_report_bytes):
    content, fingerprint = read_report(path, max_report_bytes)
    # Decode before checking XML declarations so UTF-16 cannot bypass the check.
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise DetectorError("Detector XML report must use UTF-8 encoding.") from exc
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise DetectorError("Detector XML reports must not contain DTD or entity definitions.")
    try:
        return ET.fromstring(text), fingerprint
    except ET.ParseError as exc:
        raise DetectorError("Invalid detector XML report.") from exc


def native_id(value, source_map):
    if not isinstance(value, str) or not value.strip():
        raise DetectorError("A detector report occurrence is missing a source identifier.")
    name = value.strip().replace("\\", "/")
    if source_map is None:
        return name.removeprefix("./")
    try:
        return source_map[name]
    except KeyError as exc:
        raise DetectorError("Detector report references a source missing from the supplied mapping.") from exc


def xml_span(element, source_map, *, path_field, start_field, end_field):
    try:
        return SourceSpan(
            native_id(element.get(path_field), source_map), element.get(start_field), element.get(end_field),
        )
    except ValueError as exc:
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError("Detector report contains an invalid source span.") from exc


def stage_corpus(corpus_dir, destination, extension, *, max_files, max_source_bytes):
    """Copy regular source files into controlled opaque filenames and hash their bytes."""
    corpus = Path(corpus_dir).expanduser().resolve()
    if not corpus.is_dir():
        raise DetectorError("Detector corpus must be an existing local directory.")
    file_limit = positive_number(max_files, "max_files", integer=True)
    byte_limit = positive_number(max_source_bytes, "max_source_bytes", integer=True)
    destination.mkdir()
    paths = []
    for source in corpus.rglob("*"):
        if source.suffix.lower() != extension:
            continue
        if source.is_symlink():
            raise DetectorError("Detector corpus sources must be regular files without symbolic links.")
        if source.is_file():
            paths.append(source)
            if len(paths) > file_limit:
                raise DetectorError("Detector corpus exceeds max_files.")
    if not paths:
        raise DetectorError("Detector corpus has no source files for the selected language.")
    aliases, unique_ids = {}, []
    digest = hashlib.sha256()
    for index, source in enumerate(sorted(paths)):
        file_id = source.relative_to(corpus).as_posix()
        SourceSpan(file_id, 1, 1)  # Validate the portable ID before staging bytes.
        with source.open("rb") as stream:
            content = stream.read(byte_limit + 1)
        if len(content) > byte_limit:
            raise DetectorError("Detector corpus exceeds max_source_bytes.")
        alias = f"s{index:06d}{extension}"
        staged = destination / alias
        staged.write_bytes(content)
        for name in (alias, staged.as_posix(), f"{destination.name}/{alias}"):
            aliases[name] = file_id
        unique_ids.append(file_id)
        digest.update(json.dumps([file_id, hashlib.sha256(content).hexdigest()]).encode("utf-8"))
        byte_limit -= len(content)
    return aliases, unique_ids, digest.hexdigest()
