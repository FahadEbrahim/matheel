#!/usr/bin/env python3
"""Reproduce the source-free Matheel 0.8 detector evidence from supplied local artifacts."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from matheel import __version__
from matheel.benchmark_protocol import dataset_content_fingerprint
from matheel.clone_discovery import (
    CloneFinding,
    SourceSpan,
    evaluate_clone_discovery,
    load_clone_references,
    write_bigcloneeval_findings,
    write_clone_findings,
)
from matheel.datasets import adapt_pair_dataset, load_pair_dataset
from matheel.evaluation import pair_classification_metrics, score_pair_dataset
from matheel.external_detectors import bigcloneeval, copydetect, moss, nicad, pmd_cpd, sourcerercc
from matheel.external_detectors.common import file_sha256, run_local

BCE_REVISION = "8108126f68f67f4148af91daa0b4126ee4e3cda8"
IRPLAG_REVISION = "f07e95120680b25f78659ed63cd6130fd05b2df8"
IRPLAG_NORMALIZED_SHA256 = "0460f56a3a07245b2598ac10db8374b610a1d54d9b7912473a53cbb8f6a6e889"
SIMILCODE_RELEASE = "v2.4.0"
BCE_IMAGE = "sha256:34171964a86f33798888b40ab3cfa38d1cf6329a34f20b6a1eae1dccb051ed1d"
BCE_DATABASE_LAYER = "sha256:2350bf0d5f74ceb96bf0a2b2533602946a18ec989b412d505eb59b62b959e237"
BCE_SOURCE_LAYER = "sha256:03aaf2c598806abc5d889c1c5d756cc721da9406a778dfeb1ca4b6350145e547"
BCE_DATABASE_SHA256 = "c5453e3a7ce1604c52a35f3b931afa9f70515b17b956a9da472a13e2bcbae43b"
BCE_FILTERS = {
    "minimum_lines": 6, "minimum_pretty_lines": 6, "minimum_tokens": 50, "minimum_syntax_similarity": 0.0,
}
SOURCE = """class CLASSNAME {
    static int calculate(int count) {
        int total = 1;
        for (int i = 0; i < count; i++) {
            if (i % 2 == 0) {
                total = total + i;
            } else {
                total = total - i;
            }
        }
        return total;
    }
}
"""
BCE_LABELS = {
    "Type-1": "type1",
    "Type-2": "type2",
    "Type-2 (blind)": "type2_blind",
    "Type-2 (consistent)": "type2_consistent",
    "Very-Strongly Type-3": "type3_very_strong",
    "Strongly Type-3": "type3_strong",
    "Moderatly Type-3": "type3_moderate",
    "Weakly Type-3/Type-4": "weak_type3_type4",
}
BCE_SECTIONS = {
    "-- Recall Per Clone Type (type: numDetected / numClones = recall) --": "all",
    "-- Inter-Project Recall Per Clone Type (type: numDetected / numClones = recall) --": "inter_project",
    "-- Intra-Project Recall Per Clone Type (type: numDetected / numClones = recall) --": "intra_project",
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evidence(case_id, detector, dataset, scope, metrics, provenance):
    return {
        "id": case_id, "detector": detector, "dataset": dataset, "scope": scope,
        "status": "verified", "metrics": metrics, "provenance": provenance,
    }


def span_smoke(args, root):
    if not (args.nicad or args.pmd):
        return []
    corpus = root / "java"
    corpus.mkdir()
    for name in ("Alpha", "Beta"):
        (corpus / f"{name}.java").write_text(SOURCE.replace("CLASSNAME", name), encoding="utf-8")
    scripts = []
    if args.nicad:
        scripts.append(("nicad", nicad.run_detector, args.nicad, {"txl_path": args.txl}, "Java functions"))
    if args.pmd:
        scripts.append(("pmd-cpd", pmd_cpd.run_detector, args.pmd,
                        {"java_home": args.java_home, "minimum_tokens": 20}, "Java token duplication groups"))
    results = []
    for detector, run, executable, options, scope in scripts:
        report = run(corpus, executable, **options)
        if len(report.findings) != 1:
            raise ValueError(f"Unexpected clone count from {detector}.")
        normalized = write_clone_findings(root / f"{detector}.csv", report.findings)
        finding = report.findings[0]
        metrics = {"reported_pairs": len(report.findings), "file_count": 2}
        if finding.matched_tokens is not None:
            metrics["matched_tokens"] = finding.matched_tokens
        if finding.score is not None:
            metrics["native_similarity"] = finding.score
        metadata = {key: value for key, value in report.metadata.items() if key != "report_sha256"}
        metadata["normalized_findings_sha256"] = file_sha256(normalized)
        results.append(evidence(
            f"{detector}-java-smoke", detector, "Synthetic Java duplicate",
            f"{scope} on a fixed two-file corpus with one authored duplicate", metrics, metadata,
        ))
    return results


def sourcerer_replay(root):
    report_path, source_map = root / "sourcerer.pairs", root / "sourcerer-map.csv"
    report_path.write_text("100,200\n200,100\n", encoding="utf-8")
    source_map.write_text(
        "native_id,file_id,start_line,end_line\n100,src/A.java,1,12\n200,src/B.java,1,12\n", encoding="utf-8",
    )
    report = sourcerercc.parse_report(report_path, source_map, threshold=0.8)
    if len(report.findings) != 1 or report.metadata["duplicate_pair_rows"] != 1:
        raise ValueError("SourcererCC mapping fixture did not preserve the expected pair.")
    output = write_clone_findings(root / "sourcerer-normalized.csv", report.findings)
    return evidence(
        "sourcerercc-report-import", "SourcererCC", "Synthetic paired block IDs",
        "Native CSV-format import with local source-span mapping and reversed-pair deduplication",
        {"reported_pairs": len(report.findings), "duplicate_rows": report.metadata["duplicate_pair_rows"]},
        {**report.metadata, "upstream_format_revision": "88314f51fdf3cd89103eb2c19fb56464308846c8",
         "source_map_sha256": file_sha256(source_map),
         "normalized_findings_sha256": file_sha256(output)},
    )


def moss_replay(directory):
    paths = {name: directory / name for name in ("metadata.csv", "moss_raw.csv", "moss_java.html", "moss_cs.html")}
    manifest = pd.read_csv(paths["metadata.csv"], dtype=str).fillna("")
    published = pd.read_csv(paths["moss_raw.csv"])
    aggregate = Counter()
    languages = {}
    for language, filename in (("Java", "moss_java.html"), ("C#", "moss_cs.html")):
        raw = moss.parse_report(paths[filename])
        cases = manifest[manifest["language"].str.casefold() == language.casefold()]
        expected_pairs = {}
        for case in cases.itertuples(index=False):
            right = case.code_b_id if case.code_a_id != case.code_b_id else f"{case.code_b_id}_COPY"
            expected_pairs[tuple(sorted((case.code_a_id, right)))] = case.case_id
        if len(expected_pairs) != len(cases):
            raise ValueError("SimilCode MOSS case-to-pair mapping is ambiguous.")
        recovered = {}
        for row in raw.matches:
            key = tuple(sorted((Path(row.left_id).name, Path(row.right_id).name)))
            if key in expected_pairs:
                recovered[expected_pairs[key]] = (row.left_percent + row.right_percent) / 2
        comparison = published[published["language"].str.casefold() == language.casefold()]
        sent = {row.case_id: float(row.similarity) for row in comparison.itertuples(index=False)
                if row.case_id in recovered}
        if recovered != sent:
            raise ValueError("Imported directional means differ from the archived cohort results.")
        totals = {"expected_pairs": len(expected_pairs), "reported_pairs": len(recovered),
                  "missing_expected_pairs": len(expected_pairs) - len(recovered),
                  "reported_other_pairs": len(raw.matches) - len(recovered)}
        aggregate.update(totals)
        languages[language] = totals
    if (aggregate["expected_pairs"], aggregate["reported_pairs"]) != (120, 65):
        raise ValueError(f"Unexpected SimilCode MOSS coverage: {aggregate}.")
    return evidence(
        "moss-similcode-replay", "MOSS", f"SimilCode {SIMILCODE_RELEASE}",
        "Offline replay of cohort directory-mode Java and C# HTML reports with sparse case coverage",
        {**aggregate, "languages": languages},
        {"release": SIMILCODE_RELEASE, "policy": "mean of two native percentages for explicitly reported pairs",
         "source_sha256": {name: file_sha256(path) for name, path in paths.items()}},
    )


def irplag_run(source, comparison, root):
    converted = adapt_pair_dataset(source, adapter="irplag_pair", destination=root / "irplag")
    dataset = load_pair_dataset(converted)
    source_digest = dataset_content_fingerprint(converted, "pair")["sha256"]
    if source_digest != IRPLAG_NORMALIZED_SHA256:
        raise ValueError("IRPlag does not match the published normalized dataset fingerprint.")
    labelled = ["left_id", "right_id", "label"]
    options = {"language": "java"}
    all_scores = []
    for name, kwargs in (
        ("Copydetect 0.5.0", {"algorithm": copydetect, "algorithm_options": options}),
        ("Matheel Winnowing", {"similarity_options": {"feature_weights": {"winnowing": 1.0}}}),
    ):
        scores = score_pair_dataset(dataset, **kwargs)
        score_path = root / f"{name.split()[0]}.csv"
        scores[labelled + ["similarity_score"]].to_csv(score_path, index=False)
        metrics = score_metrics(scores)
        all_scores.append({"name": name, "metrics": metrics, "scores_sha256": file_sha256(score_path)})
    if comparison:
        for name, algorithm_id in (("JPlag 6.3.0 average", "1cb01c582411128bd212"), ("Dolos 2.10.0", "aa4c49c55f5540844523")):
            (path,) = comparison.glob(f"*_{algorithm_id}_pair.csv")
            previous = pd.read_csv(path)
            current_keys = dataset.pairs[labelled].sort_values(labelled).reset_index(drop=True)
            previous_keys = previous[labelled].sort_values(labelled).reset_index(drop=True)
            if not current_keys.equals(previous_keys):
                raise ValueError(f"Historical {name} scores contain a different labelled cohort.")
            all_scores.append({"name": name, "metrics": score_metrics(previous), "scores_sha256": file_sha256(path),
                               "scope": "historical scores replay"})
    return evidence(
        "copydetect-irplag-java", "Copydetect", "IRPlag Java", "Full corpus with 460 existing labelled pairs",
        all_scores[0]["metrics"],
        {"source_revision": IRPLAG_REVISION, "normalized_dataset_sha256": source_digest,
         "source_archive_sha256": "288efbe08cc60837ae6509e49fd0285e7d22cb9b6e4dd6740d86229969fe9b8d",
         "file_count": len(dataset.files), "configuration": copydetect.portable_configuration(options),
         "comparisons": all_scores, "score_columns": labelled + ["similarity_score"]},
    )


def score_metrics(scores):
    return {
        **pair_classification_metrics(scores, threshold=0.5),
        "average_precision": float(average_precision_score(scores["label"], scores["similarity_score"])),
        "auroc": float(roc_auc_score(scores["label"], scores["similarity_score"])),
    }


def build_bce_fixture(root):
    database = root / "bigclonebenchdb" / "bcb"
    database.parent.mkdir()
    (root / "toolsdb").mkdir()
    rows = (
        (1, 1.0, 1.0, False, False, 66),
        (2, 1.0, 1.0, True, False, 66),
        (3, 1.0, 1.0, False, False, 66),
        (3, 0.9, 0.94, True, False, 66),
        (3, 0.95, 0.7, True, False, 66),
        (3, 0.50, 0.82, False, False, 66),
        (3, 0.45, 0.79, True, False, 66),
        (1, 1.0, 1.0, False, True, 66),
        (2, 1.0, 1.0, False, False, 49),
    )
    sql = [
        "CREATE TABLE version(version VARCHAR); INSERT INTO version VALUES ('controlled-h2-fixture');",
        "CREATE TABLE functions(id BIGINT PRIMARY KEY, name VARCHAR, type VARCHAR, startline INT, endline INT, "
        "normalized_size INT, project VARCHAR);",
        "CREATE TABLE functionalities(id BIGINT, name VARCHAR, description VARCHAR, search_heuristic VARCHAR); "
        "INSERT INTO functionalities VALUES (1, 'fixture', 'generated only', 'none');",
        "CREATE TABLE clones(function_id_one BIGINT, function_id_two BIGINT, syntactic_type INT, internal BOOLEAN, "
        "functionality_id BIGINT, min_size INT, min_pretty_size INT, min_tokens INT, similarity_line DOUBLE, "
        "similarity_token DOUBLE);",
        "CREATE TABLE false_positives(function_id_one BIGINT, function_id_two BIGINT, functionality_id BIGINT);",
    ]
    findings = []
    for i, (clone_type, line, token, inter, internal, tokens) in enumerate(rows, 1):
        left, right = 2 * i - 1, 2 * i
        right_project = "p2" if inter else "p1"
        sql.extend((
            f"INSERT INTO functions VALUES ({left}, 'f{left}.java', 'default', 5, 14, 10, 'p1');",
            f"INSERT INTO functions VALUES ({right}, 'f{right}.java', 'selected', 15, 24, 10, '{right_project}');",
            f"INSERT INTO clones VALUES ({left}, {right}, {clone_type}, {str(internal).upper()}, 1, 10, 10, "
            f"{tokens}, {line}, {token});",
        ))
        if i in {1, 3, 4, 6, 8, 9}:
            left_start, right_end = (5, 24) if i == 1 else (7, 22) if i == 3 else (8, 21)
            findings.append(CloneFinding(
                SourceSpan(f"selected/f{right}.java", 15, right_end), SourceSpan(f"default/f{left}.java", left_start, 14),
            ))
        elif i == 5:
            findings.append(CloneFinding(
                SourceSpan(f"default/f{left}.java", 5, 10), SourceSpan(f"selected/f{right}.java", 15, 24),
            ))
    sql_path = root / "fixture.sql"
    sql_path.write_text("\n".join(sql) + "\n", encoding="utf-8")
    return database, sql_path, findings


def bce_sections(report):
    sections = {}
    current = None
    for line in report.split("All Functionalities", 1)[-1].splitlines():
        line = " ".join(line.split())
        if line.startswith("--"):
            current = BCE_SECTIONS.get(line) if line != "-- Recall Per Clone Type --" else current
            if line.startswith("-- Type-3 Recall per 5% Region"):
                break
        elif current and (match := re.fullmatch(r"([^:]+):\s+(\d+) / (\d+) = .+", line)):
            if match[1] in BCE_LABELS:
                sections[BCE_LABELS[match[1]], current] = (int(match[2]), int(match[3]))
    return sections


def bce_replay(checkout, java, root):
    revision = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    if revision != BCE_REVISION:
        raise ValueError("Build BigCloneEval from its pinned v1.1 commit before this replay.")
    database, fixture, findings = build_bce_fixture(root)
    jar = checkout / "libs" / "h2-1.3.176.jar"
    run_local(
        [java, "-cp", jar, "org.h2.tools.RunScript", "-url", f"jdbc:h2:{database}", "-user", "sa",
         "-password", "", "-script", fixture], cwd=root, timeout_seconds=30,
    )
    db = database.with_suffix(".h2.db")
    before = file_sha256(db)
    normalized = bigcloneeval.extract_references(db, root / "reference.csv", java_path=java, h2_jar=jar)
    if before != file_sha256(db):
        raise ValueError("BigCloneEval extraction modified the source benchmark database.")
    references = tuple(load_clone_references(normalized))
    report = evaluate_clone_discovery(references, findings, **BCE_FILTERS)
    bce_format = write_bigcloneeval_findings(root / "findings.csv", findings)
    official_output = root / "native-report.txt"
    native = [java, "-cp", os.pathsep.join((str(checkout / "bin"), str(checkout / "libs" / "*"))),
              "tasks.BigCloneBench"]
    for options in (
        ["init"], ["registerTool", "-n", "Matheel replay", "-d", "Controlled fixture"],
        ["importClones", "-t", "1", "-c", bce_format],
        ["evaluateTool", "-t", "1", "-o", official_output,
         "--st", "BOTH", "--mil", "6", "--mip", "6", "--mit", "50", "-s", "0", "-m", "CoverageMatcher 0.7"],
    ):
        log = run_local([*native, *options], cwd=root, timeout_seconds=90)
        if options[0] == "importClones" and log.strip() != str(len(findings)):
            raise ValueError("The official BigCloneEval tool did not import the complete finding fixture.")
    # The legacy H2 matcher truncates fractional overlap in the pinned runtime.
    # Compare both the documented fractional rule and native full-span behavior.
    native_emulation = evaluate_clone_discovery(references, findings, coverage_threshold=1.0, **BCE_FILTERS)
    rows = {(row["clone_type"], row["project_scope"]): (row["matched"], row["eligible"])
            for row in native_emulation["strata"] if row["clone_type"] in BCE_LABELS.values()
            and row["project_scope"] in BCE_SECTIONS.values()}
    native_rows = bce_sections(official_output.read_text(encoding="utf-8"))
    if rows != native_rows or report["overall"] != {"matched": 4, "eligible": 7, "recall": 4 / 7}:
        raise ValueError(f"BigCloneEval parity failed: Matheel {rows}, native {native_rows}.")
    return evidence(
        "bigcloneeval-reference-replay", "BigCloneEval v1.1", "Controlled v1.1 H2 fixture",
        "Reference extraction and documented fractional coverage, with legacy matcher comparison",
        {**report["overall"], "reference_pairs": report["reference_pairs"], "reported_pairs": report["reported_pairs"],
         "strata_compared": len(native_rows), "strata": report["strata"], "official": native_emulation["overall"]},
        {"source_revision": revision, "h2_sha256": file_sha256(jar), "fixture_sql_sha256": file_sha256(fixture),
         "normalized_references_sha256": file_sha256(normalized), "native_findings_sha256": file_sha256(bce_format),
         "native_counts_sha256": digest_json({"/".join(key): value for key, value in native_rows.items()}),
         "protocol": report["protocol"], "source_scope": "generated H2 fixture",
         "native_matcher_behavior": "Legacy H2 rejects tested 70% and 80% partial overlaps; exact coverage agrees."},
    )


def bcb_subset(args, java, root, *, stratified, database_sha):
    jar = args.bigcloneeval_root / "libs" / "h2-1.3.176.jar"
    limits = {"max_references_per_type": 20} if stratified else {"max_references": 100}
    suffix = "typed-slice" if stratified else "subset"
    scope = (
        "First 20 non-internal references in each of seven BOTH strata, ordered by functionality and function IDs"
        if stratified else "First 100 non-internal references ordered by functionality and function IDs"
    ) + ", with both source files available"
    root.mkdir()
    normalized = bigcloneeval.extract_references(
        args.bcb_db, root / "bcb-reference.csv", java_path=java, h2_jar=jar, **limits,
    )
    corpus = root / "bcb-source"
    corpus.mkdir()
    references = tuple(load_clone_references(normalized))
    usable, copied = [], {}
    for reference in references:
        group = reference.reference_id.split(":", 1)[0]
        ids = [reference.left.file_id, reference.right.file_id]
        files = [args.ija_root / group / file_id for file_id in ids]
        if any(not file.is_file() for file in files):
            continue
        usable.append(reference)
        for file_id, source in zip(ids, files, strict=True):
            target = corpus / file_id
            digest = file_sha256(source)
            if file_id not in copied:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied[file_id] = digest
            elif copied[file_id] != digest:
                raise ValueError("A BigCloneBench file ID points to conflicting IJaDataset contents.")
    if not usable:
        raise ValueError("No pinned BigCloneBench reference pair has both sources in IJaDataset.")
    reports = []
    if args.nicad:
        reports.append(("nicad", nicad.run_detector(corpus, args.nicad, txl_path=args.txl)))
    if args.pmd:
        reports.append(("pmd-cpd", pmd_cpd.run_detector(corpus, args.pmd, java_home=args.java_home, minimum_tokens=50)))
    results = []
    for detector, native in reports:
        scored = evaluate_clone_discovery(usable, native.findings, **BCE_FILTERS)
        if not scored["overall"]["eligible"]:
            raise ValueError("The pinned BigCloneBench subset has no eligible references under the declared filters.")
        output = write_clone_findings(root / f"bcb-{detector}.csv", native.findings)
        provenance = {key: value for key, value in native.metadata.items() if key != "report_sha256"}
        results.append(evidence(
            f"{detector}-bigclonebench-v2-{suffix}", detector, "BigCloneBench v2 / IJaDataset 2.0", scope,
            {**scored["overall"], "reference_pairs_exported": len(references), "reference_pairs_with_source": len(usable),
             "reported_pairs": len(native.findings), "staged_files": len(copied), "strata": scored["strata"]},
            {**provenance, "cloreco_oci_manifest": BCE_IMAGE, "h2_database_sha256": database_sha,
             "database_oci_layer": BCE_DATABASE_LAYER, "source_oci_layer": BCE_SOURCE_LAYER,
             "typed_references_sha256": file_sha256(normalized), "selected_source_digest": digest_json(copied),
             "normalized_findings_sha256": file_sha256(output), "selection": limits, "protocol": scored["protocol"]},
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nicad", type=Path)
    parser.add_argument("--txl", type=Path)
    parser.add_argument("--pmd", type=Path)
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--bigcloneeval-root", type=Path, help="Prebuilt official BigCloneEval v1.1 checkout.")
    parser.add_argument("--bcb-db", type=Path, help="Local BigCloneBench v2 H2 database.")
    parser.add_argument("--ija-root", type=Path, help="Extracted IJaDataset bcb_reduced directory.")
    parser.add_argument("--bcb-slice", default="both", choices=("both", "prefix", "stratified"))
    parser.add_argument("--similcode-dir", type=Path, help="Local SimilCode v2.4.0 metadata and archived MOSS reports.")
    parser.add_argument("--irplag-dir", type=Path, help="Extracted IRPlag Java dataset directory.")
    parser.add_argument("--irplag-comparison", type=Path, help="Existing Matheel JPlag/Dolos IRPlag scores directory.")
    args = parser.parse_args()
    if args.nicad and not args.txl:
        parser.error("--txl must accompany --nicad.")
    if (args.pmd or args.bigcloneeval_root) and not args.java_home:
        parser.error("--java-home must accompany --pmd or --bigcloneeval-root.")
    if args.irplag_comparison and not args.irplag_dir:
        parser.error("--irplag-dir must accompany --irplag-comparison.")
    if (args.bcb_db or args.ija_root) and not (args.bcb_db and args.ija_root and args.bigcloneeval_root
                                               and (args.nicad or args.pmd)):
        parser.error("A BigCloneBench subset run needs --bcb-db, --ija-root, --bigcloneeval-root, and a local detector.")
    with TemporaryDirectory(prefix="matheel-research-detectors-") as temp:
        root = Path(temp)
        results = [sourcerer_replay(root), *span_smoke(args, root)]
        if args.similcode_dir:
            results.append(moss_replay(args.similcode_dir))
        if args.bigcloneeval_root:
            java = args.java_home.resolve() / "bin" / "java"
            results.append(bce_replay(args.bigcloneeval_root.resolve(), java, root))
            if args.bcb_db:
                database_sha = file_sha256(args.bcb_db)
                if database_sha != BCE_DATABASE_SHA256:
                    raise ValueError("BigCloneBench H2 database does not match the pinned CloReCo archive.")
                if args.bcb_slice in {"both", "prefix"}:
                    results.extend(bcb_subset(args, java, root / "prefix", stratified=False, database_sha=database_sha))
                if args.bcb_slice in {"both", "stratified"}:
                    results.extend(bcb_subset(args, java, root / "typed", stratified=True, database_sha=database_sha))
        if args.irplag_dir:
            results.append(irplag_run(args.irplag_dir, args.irplag_comparison, root))
        report = {
            "schema_version": 1, "matheel_version": __version__, "cases": results,
            "runtime": {
                "python": platform.python_version(), "machine": platform.machine(), "os": platform.system(),
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            },
            "excluded_scopes": ["Full BigCloneBench database evaluation", "Live MOSS submissions",
                                "SourcererCC end-to-end tool execution"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
