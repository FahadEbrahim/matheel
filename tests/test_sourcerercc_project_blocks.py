import pytest

from matheel.clone_discovery import SourceSpan
from matheel.external_detectors.common import DetectorError
from matheel.external_detectors.sourcerercc import parse_report


def test_project_and_block_ids_use_both_mapping_columns(tmp_path):
    report = tmp_path / "findings.csv"
    report.write_text("12,1000,11,1000\n11,1000,12,1000\n", encoding="utf-8")
    mapping = tmp_path / "source-map.csv"
    mapping.write_text(
        "native_project_id,native_id,file_id,start_line,end_line\n"
        "11,1000,selected/Left.java,2,15\n"
        "12,1000,default/Right.java,3,16\n",
        encoding="utf-8",
    )
    expected_right = SourceSpan("default/Right.java", 3, 16)
    expected_left = SourceSpan("selected/Left.java", 2, 15)
    provenance = {"granularity": "function", "threshold": 0.8, "minimum_tokens": 20, "corpus_sha256": "c" * 64}
    result = parse_report(report, mapping, **provenance)
    equivalent = parse_report(report, {(11, 1000): expected_left, (12, 1000): expected_right},
                              **provenance)
    assert result == equivalent
    assert result.metadata["format"] == "native_project_block_ids"
    assert result.metadata["duplicate_pair_rows"] == 1
    assert result.metadata["minimum_tokens"] == 20
    assert result.metadata["corpus_sha256"] == "c" * 64
    finding = result.findings[0]
    assert (finding.left, finding.right) == (expected_right, expected_left)
    assert (finding.left_native_id, finding.right_native_id) == ("12:1000", "11:1000")


def test_project_and_block_id_mapping_rejects_invalid_and_unknown_keys(tmp_path):
    report = tmp_path / "findings.csv"
    report.write_text("12,1000,13,1000\n", encoding="utf-8")
    source = SourceSpan("selected/Left.java", 2, 15)
    with pytest.raises(DetectorError, match="absent"):
        parse_report(report, {(12, 1000): source})
    with pytest.raises(DetectorError, match="one SourcererCC identifier layout"):
        parse_report(report, {(12, 1000): source, "1001": source})
    with pytest.raises(DetectorError, match="decimal IDs"):
        parse_report(report, {("12:1", 1000): source})
    with pytest.raises(DetectorError, match="minimum_tokens"):
        parse_report(report, {(12, 1000): source}, minimum_tokens=0)
    with pytest.raises(DetectorError, match="SHA-256 hex digest"):
        parse_report(report, {(12, 1000): source}, corpus_sha256="wrong")
    mapping = tmp_path / "duplicate.csv"
    mapping.write_text(
        "native_project_id,native_id,file_id,start_line,end_line\n"
        "12,1000,selected/Left.java,2,15\n"
        "12,1000,default/Right.java,3,16\n",
        encoding="utf-8",
    )
    with pytest.raises(DetectorError, match="Duplicate or ambiguous"):
        parse_report(report, mapping)
