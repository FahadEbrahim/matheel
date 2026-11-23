import csv

import pytest

from matheel.clone_discovery import (
    CloneFinding, CloneReference, SourceSpan, evaluate_clone_discovery,
    load_clone_findings, load_clone_references, write_bigcloneeval_findings,
    write_clone_findings, write_clone_references,
)


def test_typed_recall_uses_both_overlaps_reversed_pairs_and_unique_findings():
    left = SourceSpan("selected/A.java", 1, 10)
    right = SourceSpan("default/B.java", 20, 29)
    other = SourceSpan("selected/C.java", 40, 49)
    references = [
        CloneReference(left, right, "type1", "inter_project"),
        CloneReference(right, other, "weak_type3_type4", "intra_project"),
        CloneReference(left, other, "type2_consistent", "inter_project"),
    ]
    forward = CloneFinding(SourceSpan(left.file_id, 4, 10), SourceSpan(right.file_id, 17, 26))
    reverse = CloneFinding(forward.right, forward.left)
    found_wrong_on_right = CloneFinding(right, SourceSpan(other.file_id, 44, 49))
    found_other = CloneFinding(left, other)
    result = evaluate_clone_discovery(references, [forward, reverse, found_wrong_on_right, found_other])
    assert result["reported_pairs"] == 3
    assert result["overall"] == {"matched": 2, "eligible": 3, "recall": 2 / 3}
    slices = {(row["clone_type"], row["project_scope"]): row for row in result["strata"]}
    assert slices[("type2", "all")]["recall"] == 1
    assert slices[("weak_type3_type4", "intra_project")]["recall"] == 0
    assert slices[("type4", "all")]["recall"] is None


def test_source_measurements_and_reference_identity_survive_csv(tmp_path):
    left, right = SourceSpan("a.java", 1, 10), SourceSpan("b.java", 5, 14)
    references = [
        CloneReference(left, right, "type3_strong", "inter_project", 8, 60, 0.79, 0.72, "scope-1", 7),
        CloneReference(left, right, "weak_type3_type4", "inter_project", 8, 60, 0.8, 0.4, "scope-2", 7),
    ]
    path = write_clone_references(tmp_path / "reference.csv", references)
    assert list(load_clone_references(path)) == references
    result = evaluate_clone_discovery(load_clone_references(path), [CloneFinding(left, right)],
        minimum_lines=6, minimum_pretty_lines=6, minimum_tokens=50, minimum_syntax_similarity=0.5)
    assert result["reference_pairs"] == 2 and result["overall"]["eligible"] == 1
    with pytest.raises(ValueError, match="repeated reference ID"):
        evaluate_clone_discovery([references[0], references[0]], [])
    with pytest.raises(ValueError, match="missing tokens"):
        evaluate_clone_discovery([CloneReference(left, right, "type1")], [], minimum_tokens=50)


def test_findings_allow_distinct_spans_in_one_file_and_keep_provenance(tmp_path):
    finding = CloneFinding(SourceSpan("a.java", 1, 3), SourceSpan("a.java", 5, 8), 0.0, 20, "g", "x", "y")
    path = write_clone_findings(tmp_path / "findings.csv", [finding])
    assert list(load_clone_findings(path)) == [finding]
    result = evaluate_clone_discovery([CloneReference(finding.right, finding.left, "type1")], [finding])
    assert result["overall"]["matched"] == 1


def test_bigcloneeval_export_is_eight_columns_headerless_and_deduplicated(tmp_path):
    left, right = SourceSpan("one", 2, 11), SourceSpan("two", 1, 10)
    path = write_bigcloneeval_findings(tmp_path / "bce.csv", [CloneFinding(left, right), CloneFinding(right, left)],
        {"one": "selected/A.java", "two": "default/B.java"})
    with path.open() as stream:
        assert list(csv.reader(stream)) == [["default", "B.java", "1", "10", "selected", "A.java", "2", "11"]]
    with pytest.raises(ValueError, match="IJaDataset"):
        write_bigcloneeval_findings(path, [CloneFinding(left, right)])


@pytest.mark.parametrize("file_id", ["../escape.java", "/tmp/private.java", "C:/private.java", "a//b.java"])
def test_source_id_must_be_portable(file_id):
    with pytest.raises(ValueError, match="corpus-relative"):
        SourceSpan(file_id, 1, 1)
