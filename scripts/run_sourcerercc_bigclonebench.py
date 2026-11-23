"""Reproduce source-verified SourcererCC recall on the pinned BigCloneBench v2 stratified slice."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from matheel.clone_discovery import SourceSpan, evaluate_clone_discovery, load_clone_references, write_clone_findings
from matheel.external_detectors.common import file_sha256, tree_sha256
from matheel.external_detectors.sourcerercc import parse_report
from matheel.grammars import get_parser

from .check_sourcerercc_artifact import IMAGE, SEARCH_JAR_SHA256, THRESHOLD, run_linux_native
from .run_research_detectors import (
    BCE_DATABASE_LAYER,
    BCE_DATABASE_SHA256,
    BCE_FILTERS,
    BCE_IMAGE,
    BCE_SOURCE_LAYER,
    digest_json,
    evidence,
)


REFERENCE_SHA256 = "f2eccde45d6c484aa82b7a6cf6856dcc27bff1b837c2bba3b0e2bf420676da50"
SELECTED_SOURCE_SHA256 = "94ffb153a1f03ed180b3e255e4fcf5d19df2eac2374843a42cc6eb61fdc80b84"
MINIMUM_TOKENS = 65


def _method_end_lines(parser, text):
    """Index complete Java methods and constructors by plausible native start line."""
    end_lines = defaultdict(list)
    stack = [parser.parse(text.encode("utf-8")).root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.named_children)
        if node.type not in {"method_declaration", "constructor_declaration"} or node.has_error:
            continue
        identifier = node.child_by_field_name("name")
        if identifier is not None:
            for line in range(node.start_point.row + 1, identifier.start_point.row + 2):
                end_lines[line].append(node.end_point.row + 1)
    return end_lines


def _native_md5(text):
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _verified_method_map(corpus, native_stats, prefix):
    parser = get_parser("java")
    source_files, blocks, mapping = set(), set(), {}
    project = None
    corrected_end_lines = 0
    with native_stats.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, strict=True):
            if len(row) == 9 and row[0].startswith("f"):
                if not row[2].startswith(prefix):
                    raise ValueError(f"Unexpected corpus prefix in SourcererCC statistics: {row[2]!r}.")
                project = row[0][1:]
                file_id = SourceSpan(row[2].removeprefix(prefix), 1, 1).file_id
                # Match the native tokenizer's UTF-8 decoding and universal newlines.
                text = (corpus / file_id).read_text(encoding="utf-8", errors="ignore")
                if file_id in source_files or _native_md5(text) != row[4]:
                    raise ValueError(f"Unexpected or mismatched SourcererCC source file {file_id!r}.")
                source_files.add(file_id)
                endings = _method_end_lines(parser, text)
                lines = text.split("\n")
                continue
            if not source_files or len(row) != 8 or row[0] != f"b{project}" or (project, row[1]) in blocks:
                raise ValueError("Unexpected or duplicate SourcererCC block statistics row.")
            key = project, row[1]
            blocks.add(key)
            start, recorded_end = (int(value) for value in row[-2:])
            # The native hash skips empty lines. The independently parsed end
            # line remains in the original source coordinate system.
            matches = [end for end in endings.get(start, ())
                       if _native_md5("\n".join(line for line in lines[start - 1:end] if line)) == row[2]]
            if len(matches) == 1:
                end = matches[0]
                corrected_end_lines += end != recorded_end
                mapping[key] = SourceSpan(file_id, start, end)
    return source_files, blocks, mapping, corrected_end_lines


def _retain_verified_findings(native_findings, verified_findings, native_blocks, source_map):
    row_count = retained = 0
    with (native_findings.open(encoding="utf-8", newline="") as source,
          verified_findings.open("w", encoding="utf-8", newline="") as destination):
        writer = csv.writer(destination, lineterminator="\n")
        for row in csv.reader(source, strict=True):
            row_count += 1
            left, right = tuple(row[:2]), tuple(row[2:])
            if len(row) != 4 or left not in native_blocks or right not in native_blocks:
                raise ValueError("SourcererCC finding references a block absent from native statistics.")
            if left in source_map and right in source_map:
                writer.writerow(row)
                retained += 1
    return row_count, retained


def _evaluate(corpus, references, native, work, prefix, source_digests):
    runtime = (native / "runtime.txt").read_text(encoding="utf-8").splitlines()
    if not runtime or runtime[-1] != SEARCH_JAR_SHA256:
        raise ValueError("Unexpected SearchManager JAR digest in the native run.")
    native_stats = native / "file-block.stats"
    native_findings = native / "native-findings.csv"
    sources, blocks, mapping, corrected = _verified_method_map(corpus, native_stats, prefix)
    verified = work / "verified-native-findings.csv"
    raw_pairs, retained = _retain_verified_findings(native_findings, verified, blocks, mapping)
    report = parse_report(verified, mapping, version=IMAGE, threshold=THRESHOLD, granularity="function",
                          minimum_tokens=MINIMUM_TOKENS, corpus_sha256=tree_sha256(corpus))
    normalized = write_clone_findings(work / "findings.csv", report.findings)
    results = evaluate_clone_discovery(load_clone_references(references), report.findings, **BCE_FILTERS)
    grammar_dir = Path(os.environ["TREE_SITTER_LANGUAGE_PACK_LIBS_DIR"])
    grammar = {path.name: file_sha256(path) for path in sorted(grammar_dir.glob("*tree_sitter_java.*"))}
    return evidence(
        "sourcerercc-bigclonebench-v2-stratified", "sourcerercc", "BigCloneBench v2 / IJaDataset 2.0",
        "First 20 non-internal references in each of seven BOTH strata; original endpoints verified by Java syntax and native block hash",
        {**results["overall"], "strata": results["strata"], "staged_files": len(source_digests),
         "files_with_tokenizer_statistics": len(sources), "native_blocks": len(blocks),
         "verified_blocks": len(mapping), "corrected_end_lines": corrected,
         "native_pairs": raw_pairs, "verified_native_pairs": retained, "reported_pairs": len(report.findings),
         "excluded_native_pairs": raw_pairs - retained},
        {**report.metadata, "configuration": {"sharding": False}, "native_runtime": runtime,
         "typed_references_sha256": file_sha256(references), "selected_source_digest": digest_json(source_digests),
         "native_findings_sha256": file_sha256(native_findings), "native_stats_sha256": file_sha256(native_stats),
         "normalized_findings_sha256": file_sha256(normalized), "protocol": results["protocol"],
         "bcb_oci_manifest": BCE_IMAGE, "database_oci_layer": BCE_DATABASE_LAYER,
         "source_oci_layer": BCE_SOURCE_LAYER, "h2_database_sha256": BCE_DATABASE_SHA256,
         "selection": {"max_references_per_type": 20},
         "tree_sitter": {"version": version("tree-sitter"), "language_pack": version("tree-sitter-language-pack"),
                         "java_grammar": grammar}},
    )


def _stage_corpus(ija_root, references, destination):
    copied = {}
    for reference in load_clone_references(references):
        functionality = reference.reference_id.split(":", 1)[0]
        for file_id in (reference.left.file_id, reference.right.file_id):
            original = ija_root / functionality / file_id
            digest = file_sha256(original)
            if file_id in copied:
                if digest != copied[file_id]:
                    raise ValueError("A selected IJaDataset file ID identifies conflicting source bytes.")
                continue
            target = destination / file_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            copied[file_id] = digest
    return destination


def run(corpus, references, output, *, ija_root=None, recorded_native=None, recorded_prefix="tiny/"):
    if file_sha256(references) != REFERENCE_SHA256:
        raise ValueError("The reference CSV differs from the pinned stratified BigCloneBench slice.")
    with TemporaryDirectory(prefix="matheel-sourcerercc-bcb-") as directory:
        work = Path(directory)
        if corpus is None:
            corpus = _stage_corpus(ija_root, references, work / "corpus")
        source_digests = {path.relative_to(corpus).as_posix(): file_sha256(path)
                          for path in sorted(corpus.rglob("*.java"))}
        if digest_json(source_digests) != SELECTED_SOURCE_SHA256 or len(source_digests) != 141:
            raise ValueError("The source bytes differ from the pinned 141-file BigCloneBench slice.")
        if recorded_native is None:
            recorded_native = work / "native"
            run_linux_native(corpus, recorded_native, minimum_tokens=MINIMUM_TOKENS)
        result = _evaluate(corpus, references, recorded_native, work, recorded_prefix, source_digests)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--corpus", type=Path, help="Staged 141-file Java slice.")
    source_group.add_argument("--ija-root", type=Path, help="Original IJaDataset bcb_reduced directory.")
    parser.add_argument("--references", type=Path, required=True, help="Normalized stratified BigCloneBench references.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recorded-native", type=Path, help="Replay a supplied tokenizer stats and pair output directory.")
    parser.add_argument("--recorded-prefix", default="tiny/", help="Path prefix used by the supplied native tokenizer stats.")
    args = parser.parse_args()
    run(args.corpus, args.references, args.output, ija_root=args.ija_root, recorded_native=args.recorded_native,
        recorded_prefix=args.recorded_prefix)
