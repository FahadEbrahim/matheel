import hashlib
import json
import zipfile

import pandas as pd
import pytest

from matheel.benchmark_protocol import dataset_content_fingerprint
from matheel.datasets import load_code_texts
from scripts.prepare_plagiarism_benchmark import (
    prepare_benchmark,
    prepare_conplag,
    prepare_soco14,
    verify_archive,
)


def _source():
    return {
        "source_url": "https://example.invalid/pinned-source", "source_revision": "synthetic-version-1",
        "license": "synthetic", "archives": [{"sha256": "0" * 64}],
        "ground_truth": "synthetic annotations", "granularity": "file",
    }


def test_verify_archive_rejects_corruption_and_missing_files(tmp_path):
    path = tmp_path / "tiny.zip"
    path.write_bytes(b"archive")
    lock = {"size": 7, "sha256": hashlib.sha256(b"archive").hexdigest()}
    assert verify_archive(path, lock) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_archive(path, lock)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="--download"):
        verify_archive(path, lock)


def test_conplag_preparation_uses_pair_specific_raw_copies_not_tool_rewrites(tmp_path):
    versions = tmp_path / "raw" / "versions"
    for directory, text in (("version_1/a_b", "raw first"), ("version_1/a_c", "raw second"), ("bplag_version_1/a_b/a", "rewritten")):
        target = versions / directory
        target.mkdir(parents=True)
        (target / "a.java").write_text(text)
    (versions / "version_1/a_b/b.java").write_text("b")
    (versions / "version_1/a_c/c.java").write_text("c")
    pd.DataFrame([{"sub1": "a", "sub2": "b", "problem": 1, "verdict": 1},
                  {"sub1": "a", "sub2": "c", "problem": 1, "verdict": 0}]).to_csv(versions / "labels.csv", index=False)
    dataset = prepare_conplag(tmp_path / "raw", tmp_path / "normalized", _source())
    assert len(dataset.pairs) == 2
    assert len(dataset.files) == 4
    assert set(load_code_texts(dataset).values()) == {"raw first", "raw second", "b", "c"}
    assert dataset.pairs.iloc[0]["left_id"] != dataset.pairs.iloc[1]["left_id"]


def _soco_fixture(root):
    root.mkdir(parents=True)
    for language in ("c", "java"):
        directory = root / language / "A1"
        directory.mkdir(parents=True)
        for token in ("A10001", "A10002", "A10003"):
            (directory / token).write_text(f"{language} {token}")
        (root / f"soco14-test-{language}-update.qrel").write_text("A10001 A10002\nA10001 A19999\n")
    return root


def test_soco_preparation_separates_language_identifiers_and_records_missing_qrels(tmp_path):
    source = _soco_fixture(tmp_path / "first" / "fire14-source-code-test-dataset")
    datasets, omissions = prepare_soco14(source, tmp_path / "output", _source())
    assert len(datasets) == 2
    assert len(omissions["missing_qrel_rows"]) == 2
    assert len(omissions["excluded_unjudged_views"]) == 10
    for dataset in datasets:
        assert len(dataset.queries) == len(dataset.corpus) == 2
        assert len(dataset.qrels) == 2
        assert set(zip(dataset.qrels["query_id"], dataset.qrels["document_id"])) == {("A10001", "A10002"), ("A10002", "A10001")}
        assert {text.split()[0] for text in load_code_texts(dataset).values()} == ({"c"} if dataset.metadata["languages"] == ["cpp"] else {"java"})
    relocated = _soco_fixture(tmp_path / "second" / "fire14-source-code-test-dataset")
    second, _ = prepare_soco14(relocated, tmp_path / "second-output", _source())
    assert [dataset_content_fingerprint(ds.root, "retrieval") for ds in datasets] == [dataset_content_fingerprint(ds.root, "retrieval") for ds in second]


def test_prepare_benchmark_uses_verified_local_archives_without_network(tmp_path, monkeypatch):
    import scripts.prepare_plagiarism_benchmark as prepare_module

    source = _soco_fixture(tmp_path / "raw" / "fire14-source-code-test-dataset")
    cache = tmp_path / "cache" / "soco14"
    cache.mkdir(parents=True)
    archive = cache / "test.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source.parent).as_posix())
    lock = {"schema_version": 1, "benchmark_id": "Matheel-Plagiarism-v1", "sources": {"soco14": {
        **_source(), "selection": "synthetic judged-endpoint scenario fixtures",
        "archives": [{"filename": "test.zip", "size": archive.stat().st_size,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "url": "https://example.invalid/test.zip"}],
    }}}
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(lock))
    monkeypatch.setattr(prepare_module, "_download_url_to_file", lambda *args, **kwargs: pytest.fail("Unexpected network access"))
    manifest_path, summary = prepare_benchmark(tmp_path / "cache", tmp_path / "normalized", sources_path=sources_path)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["datasets"]) == len(summary["datasets"]) == 2
    assert all(entry["selection"]["candidate_policy"] == "exclude_same_file" for entry in manifest["datasets"])
    assert manifest["threshold_policy"] == {"mode": "fixed", "value": 0.5}
