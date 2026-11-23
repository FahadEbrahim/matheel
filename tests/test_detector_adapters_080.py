import json
import os
from pathlib import Path

import pytest

from matheel.clone_discovery import SourceSpan
from matheel.external_detectors import bigcloneeval, copydetect, moss, nicad, pmd_cpd, sourcerercc
from matheel.external_detectors.common import DetectorError


def test_bigcloneeval_recovers_archived_database_only_in_private_copy(tmp_path, monkeypatch):
    database = tmp_path / "bcb.h2.db"
    database.write_bytes(b"original")
    staging = tmp_path / "work"
    staging.mkdir()
    calls = []

    def fake_run(command, *, cwd, **options):
        calls.append(command[command.index("-url") + 1])
        if len(calls) == 1:
            (cwd / "process.log").write_text("The database is read only [90097-176]\nPageLog.recover")
            raise DetectorError("Local detector exited with status 1.")
        local_copy = cwd / database.name
        assert local_copy.read_bytes() == b"original"
        local_copy.write_bytes(b"recovered")
        options["result_paths"][0].write_text("complete report")

    monkeypatch.setattr(bigcloneeval, "run_local", fake_run)
    output = staging / "raw.csv"
    bigcloneeval._export_h2(staging, database, staging / "extract.sql", output, "java", "h2.jar", 512, 30, 2000)
    assert calls == [f"jdbc:h2:{tmp_path / 'bcb'};IFEXISTS=TRUE;ACCESS_MODE_DATA=r",
                     f"jdbc:h2:{staging / 'bcb'};IFEXISTS=TRUE"]
    assert database.read_bytes() == b"original"
    assert output.read_text() == "complete report"


def test_nicad_imports_native_percentage_and_normalization(tmp_path):
    report = tmp_path / "nicad.xml"
    report.write_text('''<clones><systeminfo processor="nicad6" granularity="functions-blind"
        threshold="30%" minlines="6" maxlines="2500"/><cloneinfo npcs="2" npairs="1"/>
        <clone similarity="82"><source file="/archive/A.java" startline="3" endline="8"/>
        <source file="/archive/B.java" startline="4" endline="10"/></clone></clones>''')
    result = nicad.parse_report(report, {"/archive/A.java": "one", "/archive/B.java": "two"})
    assert result.findings[0].left == SourceSpan("one", 3, 8)
    assert result.findings[0].score == 0.82 and result.metadata["maximum_difference"] == 0.3
    assert "archive" not in json.dumps(result.metadata)
    with pytest.raises(DetectorError, match="missing from the supplied mapping"):
        nicad.parse_report(report, {"/archive/A.java": "one"})
    report.write_text('<clones><systeminfo threshold="30%" minlines="6" maxlines="2500"/>'
                      '<cloneinfo npcs="0" npairs="0"/></clones>')
    assert nicad.parse_report(report).findings == ()


def test_pmd_preserves_duplication_groups_and_within_file_occurrences(tmp_path):
    report = tmp_path / "cpd.xml"
    report.write_text('''<pmd-cpd xmlns="https://pmd-code.org/schema/cpd-report" pmdVersion="7.29.0">
      <file path="A.java"/><file path="B.java"/>
      <duplication lines="4" tokens="43">
        <file path="A.java" line="1" endline="4"/>
        <file path="A.java" line="9" endline="12"/>
        <file path="B.java" line="1" endline="4"/>
      </duplication>
    </pmd-cpd>''')
    result = pmd_cpd.parse_report(report)
    assert result.metadata["parsed_files"] == 2
    assert len(result.findings) == 3
    assert result.findings[0].left.file_id == result.findings[0].right.file_id
    assert {finding.group_id for finding in result.findings} == {"cpd-000001"}
    assert {finding.matched_tokens for finding in result.findings} == {43}
    with pytest.raises(DetectorError, match="max_report_bytes"):
        pmd_cpd.parse_report(report, max_report_bytes=10)
    report.write_text('<!DOCTYPE pmd-cpd [ <!ENTITY test "bad"> ]><pmd-cpd/>')
    with pytest.raises(DetectorError, match="DTD"):
        pmd_cpd.parse_report(report)


def test_sourcerercc_joins_native_ids_with_spans_and_counts_reversed_duplicates(tmp_path):
    report, mapping = tmp_path / "results.pairs", tmp_path / "mapping.csv"
    report.write_text("0042,01\n01,0042\n")
    mapping.write_text("native_id,file_id,start_line,end_line\n0042,first/Foo.java,4,18\n01,second/Foo.java,9,23\n")
    result = sourcerercc.parse_report(report, mapping, granularity="function", threshold=0.8)
    assert len(result.findings) == 1 and result.metadata["duplicate_pair_rows"] == 1
    assert result.findings[0].left_native_id == "0042" and result.findings[0].right_native_id == "01"
    assert result.findings[0].right == SourceSpan("second/Foo.java", 9, 23)
    with pytest.raises(DetectorError, match="absent from the source map"):
        sourcerercc.parse_report(report, {"0042": SourceSpan("first/Foo.java", 4, 18)})
    with mapping.open("a") as stream:
        stream.write("01,more/Foo.java,9,23\n")
    with pytest.raises(DetectorError, match="Duplicate or ambiguous"):
        sourcerercc.parse_report(report, mapping)


def test_moss_retains_reported_zeros_and_sparse_coverage(tmp_path):
    path = tmp_path / "index.html"
    path.write_text('''<html><head><title>Moss Results</title></head><body><table>
        <tr><th>File 1<th>File 2<th>Lines Matched
        <tr><td><a href="https://example.test/match0.html">a/ (0%)</a>
            <td><a href="https://example.test/match0.html">b/ (10%)</a><td>5
        </table></body></html>''')
    (tmp_path / "match0-top.html").write_text('''<table><tr><th>a (0%)<th><th>b (10%)<th>
        <tr><td><a href="match0-0.html#0">3-5</a><td><img src="x">
        <td><a href="match0-1.html#0">6</a><td><img src="x"></table>''')
    result = moss.parse_report(path, {"a/": "left", "b/": "right"}, match_directory=tmp_path)
    item = result.matches[0]
    assert (item.left_percent, item.right_percent, item.matched_lines) == (0, 10, 5)
    assert item.passages == ((SourceSpan("left", 3, 5), SourceSpan("right", 6, 6)),)
    assert moss.compare_coverage(result, [("right", "left"), ("left", "omitted")]) == {
        "expected_pairs": 2, "reported_expected_pairs": 1, "missing_expected_pairs": 1, "reported_other_pairs": 0,
    }
    exported = moss.write_moss_matches(tmp_path / "moss.csv", result).read_text()
    assert "[[3,5,6,6]]" in exported
    assert "example.test" not in exported and "omitted" not in exported


def test_copydetect_rejects_missing_scores_and_keeps_directional_match_offsets(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(["a", "b", 0.8, 0.3, 6, [[2, 8]], [[1, 7]]]) + "\n")
    identity = {"configuration": {"include_spans": True, "score_mode": "query"}}
    context = copydetect._score_records(path, {"a": "first", "b": "second"}, identity)
    assert context.score({"left_id": "first", "right_id": "second"}) == 0.8
    assert context.score({"left_id": "second", "right_id": "first"}) == 0.3
    assert context.matched_slices("second", "first")["left_slices"] == ((1, 7),)
    with pytest.raises(DetectorError, match="Incomplete detector pair coverage"):
        copydetect._score_records(path, {"a": "first", "b": "second", "c": "third"}, identity)


@pytest.mark.integration
def test_copydetect_real_wheel_accepts_empty_and_short_sources():
    pytest.importorskip("copydetect")
    text = "class A { int square(int n) { return n * n; } }"
    options = {"language": "java", "noise_threshold": 5, "guarantee_threshold": 8, "include_spans": True}
    sources = {"a": text, "b": text, "short": "x", "empty": ""}
    first = copydetect.prepare_dataset(None, prepared_texts=sources, algorithm_options=options)
    second = copydetect.prepare_dataset(None, prepared_texts=sources, algorithm_options=options)
    assert first.scores == second.scores and first.scores["a", "b"] == 1
    assert first.scores["a", "short"] == first.scores["a", "empty"] == 0
    assert first.matched_slices("a", "b")["matched_tokens"] > 0
    assert copydetect.portable_configuration(first.identity) == first.identity


@pytest.mark.integration
@pytest.mark.parametrize("module", [nicad, pmd_cpd])
def test_real_java_span_detector(module, tmp_path):
    if module is nicad:
        binary, txl = os.getenv("MATHEEL_NICAD_BIN"), os.getenv("MATHEEL_TXL_PATH")
        if not binary or not txl:
            pytest.skip("Set MATHEEL_NICAD_BIN and MATHEEL_TXL_PATH to run the pinned tools.")
        options = {"txl_path": txl}
    else:
        binary, java_home = os.getenv("MATHEEL_PMD_BIN"), os.getenv("MATHEEL_JAVA_HOME")
        if not binary or not java_home:
            pytest.skip("Set MATHEEL_PMD_BIN and MATHEEL_JAVA_HOME to run the pinned tools.")
        options = {"java_home": java_home, "minimum_tokens": 20}
    lines = "\n".join(f"        total = total + {i};" for i in range(12))
    source = "class CLASSNAME {\n    int sample(int total) {\n" + lines + "\n        return total;\n    }\n}\n"
    for name in ("Alpha", "Beta"):
        (tmp_path / f"{name}.java").write_text(source.replace("CLASSNAME", name))
    result = module.run_detector(tmp_path, Path(binary), **options)
    assert len(result.findings) == 1
    assert {result.findings[0].left.file_id, result.findings[0].right.file_id} == {"Alpha.java", "Beta.java"}
    assert result.metadata["execution"] == "local"
