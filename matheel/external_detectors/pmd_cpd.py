"""PMD Copy/Paste Detector reports and optional local Java duplication runner."""

import os
import re
import tempfile
from itertools import combinations
from pathlib import Path

from ..clone_discovery import CloneDetectorReport, CloneFinding
from ._native import MAX_REPORT_BYTES, native_id, read_xml_report, stage_corpus, xml_span
from .common import DetectorError, file_sha256, positive_number, required_file, run_local, tree_sha256

VERSION = "7.29.0"
_NAMESPACE = "{https://pmd-code.org/schema/cpd-report}"


def parse_report(path, source_map=None, *, max_report_bytes=MAX_REPORT_BYTES):
    """Read CPD 7 XML, retaining the token count and duplication group."""
    root, report_hash = read_xml_report(path, max_report_bytes)
    if root.tag != f"{_NAMESPACE}pmd-cpd":
        raise DetectorError("Expected a namespaced PMD CPD XML report.")
    if root.find(f"{_NAMESPACE}error") is not None:
        raise DetectorError("PMD CPD reported source processing errors.")
    files = {
        native_id(element.get("path"), source_map) for element in root.findall(f"{_NAMESPACE}file")
    }
    groups = root.findall(f"{_NAMESPACE}duplication")
    findings = []
    for index, group in enumerate(groups, 1):
        occurrences = [
            xml_span(element, source_map, path_field="path", start_field="line", end_field="endline")
            for element in group.findall(f"{_NAMESPACE}file")
        ]
        if len(occurrences) < 2 or len(set(occurrences)) != len(occurrences):
            raise DetectorError("Each PMD duplication requires at least two distinct source spans.")
        if files and any(span.file_id not in files for span in occurrences):
            raise DetectorError("PMD duplication references an undeclared source file.")
        tokens = positive_number(group.get("tokens"), "CPD tokens", integer=True)
        findings.extend(
            CloneFinding(left, right, matched_tokens=tokens, group_id=f"cpd-{index:06d}")
            for left, right in combinations(occurrences, 2)
        )
    metadata = {
        "detector": "pmd_cpd",
        "format": "cpd_xml_v7",
        "version": root.get("pmdVersion"),
        "report_sha256": report_hash,
        "reported_groups": len(groups),
        "reported_pairs": len(findings),
        "parsed_files": len(files),
        "granularity": "block",
    }
    return CloneDetectorReport(tuple(findings), metadata)


def run_detector(
    corpus_dir, executable, *, java_home=None, language="java", minimum_tokens=50,
    ignore_identifiers=False, ignore_literals=False, expected_installation_sha256=None,
    timeout_seconds=300, max_files=500, max_source_bytes=50_000_000,
    max_output_bytes=2_000_000, max_report_bytes=MAX_REPORT_BYTES,
):
    """Run the pinned PMD release once on a staged Java corpus."""
    if language != "java":
        raise DetectorError("The pinned PMD CPD runner currently supports Java.")
    if not isinstance(ignore_identifiers, bool) or not isinstance(ignore_literals, bool):
        raise DetectorError("PMD normalization options must be boolean.")
    minimum_tokens = positive_number(minimum_tokens, "minimum_tokens", integer=True)
    pmd = required_file(executable, "PMD executable")
    pmd_root = pmd.parent.parent
    if pmd.parent.name != "bin" or not (pmd_root / "lib").is_dir():
        raise DetectorError("Use the PMD executable inside a PMD distribution's bin directory.")
    java_home = java_home or os.environ.get("JAVA_HOME")
    if not java_home:
        raise DetectorError("Specify java_home or configure JAVA_HOME for PMD CPD.")
    java = required_file(Path(java_home) / "bin" / "java", "Java executable")
    java_root = java.parent.parent
    release = required_file(java_root / "release", "Java release metadata")
    modules = required_file(java_root / "lib" / "modules", "Java runtime modules")
    environment = {"JAVA_HOME": str(java_root)}
    installation = tree_sha256(pmd_root)
    if expected_installation_sha256 is not None and installation != expected_installation_sha256:
        raise DetectorError("PMD installation SHA256 does not match the expected artifact.")

    with tempfile.TemporaryDirectory(prefix="matheel-cpd-") as temporary:
        root = Path(temporary).resolve()
        version_text = run_local([pmd, "--version"], cwd=root, timeout_seconds=15, environment_overrides=environment)
        if re.search(rf"\b{re.escape(VERSION)}\b", version_text) is None:
            raise DetectorError(f"This PMD CPD runner requires upstream PMD {VERSION}.")
        sources, file_ids, corpus_hash = stage_corpus(
            corpus_dir, root / "inputs", ".java", max_files=max_files, max_source_bytes=max_source_bytes,
        )
        report_path = root / "cpd.xml"
        command = [
            pmd, "cpd", "--dir", root / "inputs", "--language", language, "--encoding", "UTF-8",
            "--minimum-tokens", str(minimum_tokens), "--format", "xml", "--report-file", report_path,
            "--no-fail-on-violation",
        ]
        if ignore_identifiers:
            command.append("--ignore-identifiers")
        if ignore_literals:
            command.append("--ignore-literals")
        run_local(
            command, cwd=root, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes,
            result_paths=(report_path,), max_result_bytes=max_report_bytes, environment_overrides=environment,
        )
        report = parse_report(report_path, sources, max_report_bytes=max_report_bytes)
    if report.metadata["parsed_files"] != len(file_ids):
        raise DetectorError("PMD CPD did not report processing every staged source file.")
    return CloneDetectorReport(report.findings, {
        **report.metadata, "version": VERSION, "corpus_sha256": corpus_hash,
        "execution": "local", "installation_sha256": installation,
        "runtime": {
            "executable_sha256": file_sha256(java), "release_sha256": file_sha256(release),
            "modules_sha256": file_sha256(modules),
        },
        "configuration": {
            "language": language, "minimum_tokens": minimum_tokens,
            "ignore_identifiers": ignore_identifiers, "ignore_literals": ignore_literals,
            "timeout_seconds": timeout_seconds, "max_files": max_files, "max_source_bytes": max_source_bytes,
            "max_output_bytes": max_output_bytes, "max_report_bytes": max_report_bytes,
        },
    })
