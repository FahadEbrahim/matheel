"""NiCad clone-pair XML and optional local NiCad 7 Java runner."""

import os
import re
import shutil
import tempfile
from pathlib import Path

from ..clone_discovery import CloneDetectorReport, CloneFinding
from ._native import MAX_REPORT_BYTES, read_xml_report, stage_corpus, xml_span
from .common import DetectorError, checked_score, file_sha256, required_file, run_local, tree_sha256

SUPPORTED_CONFIGURATIONS = ("default", "type1", "type2", "type2c", "type3-1", "type3-2", "type3-2c")


def _report_count(element, attribute):
    value = element.get(attribute, "")
    if re.fullmatch(r"[0-9]+", value) is None:
        raise DetectorError(f"NiCad report requires a valid {attribute} count.")
    return int(value)


def parse_report(path, source_map=None, *, max_report_bytes=MAX_REPORT_BYTES):
    """Read a native NiCad pair report and its original inclusive line ranges."""
    root, report_hash = read_xml_report(path, max_report_bytes)
    info, counts = root.find("systeminfo"), root.find("cloneinfo")
    if root.tag != "clones" or info is None or counts is None or root.find("class") is not None:
        raise DetectorError("Expected NiCad clone-pair XML with systeminfo and cloneinfo.")
    matches = root.findall("clone")
    if _report_count(counts, "npairs") != len(matches):
        raise DetectorError("NiCad clone count does not match the number of reported pairs.")
    threshold = info.get("threshold", "")
    if not threshold.endswith("%"):
        raise DetectorError("NiCad report has an invalid normalization difference threshold.")
    try:
        threshold = checked_score(float(threshold[:-1]) / 100)
        findings = []
        for index, match in enumerate(matches, 1):
            sources = match.findall("source")
            if len(sources) != 2:
                raise DetectorError("NiCad clone-pair XML must contain two sources per match.")
            left, right = (
                xml_span(source, source_map, path_field="file", start_field="startline", end_field="endline")
                for source in sources
            )
            score = checked_score(float(match.get("similarity", "")) / 100)
            findings.append(CloneFinding(left, right, score=score, group_id=f"nicad-{index:06d}"))
    except ValueError as exc:
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError("NiCad report has invalid clone scores or source spans.") from exc
    return CloneDetectorReport(tuple(findings), {
        "detector": "nicad", "format": "clone_pair_xml", "report_sha256": report_hash,
        "processor": info.get("processor"), "granularity": info.get("granularity"),
        "maximum_difference": threshold,
        "minimum_pretty_lines": _report_count(info, "minlines"),
        "maximum_pretty_lines": _report_count(info, "maxlines"),
        "parsed_fragments": _report_count(counts, "npcs"), "reported_pairs": len(findings),
    })


def run_detector(
    corpus_dir, executable, *, txl_path=None, language="java", granularity="functions",
    configuration="default", expected_installation_sha256=None, timeout_seconds=300,
    max_files=500, max_source_bytes=50_000_000, max_output_bytes=2_000_000,
    max_report_bytes=MAX_REPORT_BYTES,
):
    """Run an optional NiCad 7 distribution once on an opaque staged Java corpus."""
    if language != "java" or granularity not in {"functions", "blocks", "files"}:
        raise DetectorError("The NiCad runner supports Java functions, blocks, or files.")
    if configuration not in SUPPORTED_CONFIGURATIONS:
        raise DetectorError("Unsupported NiCad built-in configuration.")
    nicad = required_file(executable, "NiCad executable")
    library = nicad.parent.parent / "lib" / "nicad"
    if not (library / "scripts" / "nicadpair.sh").is_file():
        raise DetectorError("Use the NiCad executable from a distribution with local lib/nicad.")
    txl = required_file(txl_path or shutil.which("txl"), "OpenTxl executable")
    if txl.name != "txl":
        raise DetectorError("OpenTxl executable must be named txl.")
    tree_hash = tree_sha256(library)
    if expected_installation_sha256 is not None and tree_hash != expected_installation_sha256:
        raise DetectorError("NiCad installation SHA256 does not match the expected artifact.")
    environment = {"PATH": os.pathsep.join((str(txl.parent), os.environ.get("PATH", os.defpath)))}

    with tempfile.TemporaryDirectory(prefix="matheel-nicad-") as temporary:
        root = Path(temporary).resolve()
        # OpenTxl's version-only command emits its version and exits with 1.
        txl_version = run_local([txl, "-V"], cwd=root, timeout_seconds=15, allowed_return_codes=(0, 1)).splitlines()
        version_line = next((line for line in txl_version if line.startswith("OpenTxl v11.")), None)
        if version_line is None:
            raise DetectorError("The NiCad 7 runner requires OpenTxl 11.")
        (root / "lib").mkdir()
        (root / "lib" / "nicad").symlink_to(library, target_is_directory=True)
        sources, file_ids, corpus_hash = stage_corpus(
            corpus_dir, root / "inputs", ".java", max_files=max_files, max_source_bytes=max_source_bytes,
        )
        sources.update({f"nicadclones/inputs/{name}": file_id for name, file_id in sources.items()
                        if name.startswith("inputs/")})
        output = root / "nicadclones"
        log = run_local(
            [nicad, granularity, language, "inputs", configuration], cwd=root, timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes, result_paths=(output,), max_result_bytes=max_report_bytes,
            environment_overrides=environment,
        )
        version_match = re.search(r"^NiCad Clone Detector (v7\.0[^\r\n]*)", log, re.MULTILINE)
        if version_match is None:
            raise DetectorError("The NiCad executable did not identify itself as upstream NiCad 7.0.")
        reports = sorted(output.glob("inputs/*-clones/*-clones-[0-9]*.xml"))
        reports = [path for path in reports if re.search(r"-clones-[0-9]+\.[0-9]+\.xml$", path.name)]
        if len(reports) != 1:
            raise DetectorError("NiCad did not produce exactly one native clone-pair report.")
        report = parse_report(reports[0], sources, max_report_bytes=max_report_bytes)
    return CloneDetectorReport(report.findings, {
        **report.metadata, "version": version_match.group(1), "execution": "local",
        "installation_sha256": tree_hash, "executable_sha256": file_sha256(nicad),
        "runtime": {"version": version_line, "executable_sha256": file_sha256(txl)},
        "corpus_sha256": corpus_hash, "staged_files": len(file_ids),
        "configuration": {
            "language": language, "granularity": granularity, "profile": configuration,
            "timeout_seconds": timeout_seconds, "max_files": max_files, "max_source_bytes": max_source_bytes,
            "max_output_bytes": max_output_bytes, "max_report_bytes": max_report_bytes,
        },
    })
