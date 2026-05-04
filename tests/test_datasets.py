import json
import sys
import zipfile
from types import ModuleType

import pandas as pd
import pytest

import matheel
from matheel.dataset_validation import (
    dataset_validation_report_payload,
    validate_dataset_report,
    write_dataset_validation_report,
)
from matheel.datasets import (
    adapt_pair_dataset,
    adapt_retrieval_dataset,
    available_dataset_adapters,
    available_dataset_kinds,
    available_dataset_presets,
    available_dataset_presets_by_task,
    available_dataset_sources,
    available_dataset_task_types,
    get_dataset_entry,
    get_dataset_preset,
    load_code_texts,
    load_dataset_manifest,
    load_pair_dataset,
    load_pair_datasets,
    load_pair_datasets_from_manifest,
    load_retrieval_dataset,
    load_retrieval_datasets,
    load_retrieval_datasets_from_manifest,
    registered_datasets,
    register_dataset_adapter,
    register_dataset_entry,
    register_dataset_preset,
    register_dataset_source,
    resolve_dataset_source,
    write_retrieval_dataset,
    write_pair_dataset,
    _resolve_kaggle_dataset_source,
    _safe_extract_archive,
)


def test_dataset_registry_tracks_plagiarism_entries():
    entry = register_dataset_entry(
        "unit_plagiarism_pairs",
        task_type="plagiarism",
        dataset_kind="pair_classification",
        languages=("python", "java"),
        license="unknown",
        source_url="https://example.invalid/unit",
        access="manual",
        overwrite=True,
    )

    assert available_dataset_task_types() == ("plagiarism",)
    assert "pair_classification" in available_dataset_kinds()
    assert get_dataset_entry("unit_plagiarism_pairs") == entry
    assert entry in registered_datasets(task_type="plagiarism", dataset_kind="pair_classification")


def test_dataset_source_registry_resolves_local_sources(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)"},
                {"file_id": "b", "text": "print(2)"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
    )

    assert "local" in available_dataset_sources()
    assert resolve_dataset_source("local", dataset_root) == dataset_root.resolve()


def test_dataset_validation_report_exports_clean_pair_dataset(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
        metadata={"name": "validation_pairs"},
    )

    report, artifacts = write_dataset_validation_report(
        dataset_root,
        tmp_path / "validation",
        kind="pair",
        basename="tiny",
    )
    payload = dataset_validation_report_payload(report)

    assert report["status"] == "warning"
    assert report["warning_count"] == 1
    assert report["counts"]["pairs"] == 1
    assert any(issue["code"] == "single_class_labels" for issue in report["issues"])
    assert payload["schema_version"] == 1
    assert artifacts["issues_csv"].exists()
    assert artifacts["report_html"].exists()
    assert matheel.validate_dataset_report is validate_dataset_report
    assert matheel.write_dataset_validation_report is write_dataset_validation_report


def test_dataset_validation_report_finds_pair_reference_errors(tmp_path):
    dataset_root = tmp_path / "broken_pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "", "suffix": ".py"},
                {"file_id": "b", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
        metadata={"name": "broken_pairs"},
    )
    pairs_path = dataset_root / "pairs.csv"
    pairs = pd.read_csv(pairs_path)
    pairs.loc[0, "right_id"] = "missing"
    pairs.to_csv(pairs_path, index=False)

    report = validate_dataset_report(dataset_root, kind="pair")
    codes = {issue["code"] for issue in report["issues"]}

    assert report["status"] == "error"
    assert "unknown_pair_file_ids" in codes
    assert "empty_files" in codes


def test_dataset_validation_report_finds_retrieval_reference_errors(tmp_path):
    dataset_root = tmp_path / "broken_retrieval"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "doc_a"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
        metadata={"name": "broken_retrieval"},
    )
    qrels_path = dataset_root / "qrels.csv"
    qrels = pd.read_csv(qrels_path)
    qrels.loc[0, "document_id"] = "missing"
    qrels.loc[0, "relevance"] = -1
    qrels.to_csv(qrels_path, index=False)

    report = validate_dataset_report(dataset_root, kind="retrieval")
    codes = {issue["code"] for issue in report["issues"]}

    assert report["status"] == "error"
    assert "unknown_qrel_document_ids" in codes
    assert "invalid_qrel_relevance" in codes


def test_default_dataset_sources_include_generic_resolvers():
    sources = set(available_dataset_sources())

    assert {"local", "github", "zenodo", "huggingface", "kaggle"}.issubset(sources)


def test_default_dataset_presets_include_only_approved_plagiarism_presets():
    presets = set(available_dataset_presets())
    pair_presets = set(available_dataset_presets_by_task("pair"))
    retrieval_presets = set(available_dataset_presets_by_task("retrieval"))

    assert presets == {
        "soco14",
        "irplag",
        "conplag",
        "ipca",
        "student_code_similarity",
        "criminal_minds",
    }
    assert "soco14" not in pair_presets
    assert {
        "irplag",
        "conplag",
        "ipca",
        "student_code_similarity",
        "criminal_minds",
    }.issubset(pair_presets)
    assert {"soco14", "irplag"}.issubset(retrieval_presets)
    assert "conplag" not in retrieval_presets
    assert "ipca" not in retrieval_presets
    assert "student_code_similarity" not in retrieval_presets
    assert "criminal_minds" not in retrieval_presets
    assert get_dataset_preset("soco14")["identifier"] == "7433031"
    assert get_dataset_preset("irplag")["source"] == "github"
    assert get_dataset_preset("conplag")["url"] == "https://zenodo.org/records/7332790"
    assert get_dataset_preset("ipca")["adapter"] == "ipca_pair"
    assert get_dataset_preset("student_code_similarity")["source"] == "kaggle"
    assert get_dataset_preset("criminal_minds")["identifier"] == "19115559"


def test_safe_archive_extraction_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "nope")

    with pytest.raises(ValueError, match="escape output directory"):
        _safe_extract_archive(archive_path, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_kaggle_api_resolver_downloads_then_safe_extracts(tmp_path, monkeypatch):
    captured = {}

    class FakeKaggleApi:
        def authenticate(self):
            captured["authenticated"] = True

        def dataset_download_files(self, identifier, path, unzip=False):
            captured["identifier"] = identifier
            captured["path"] = path
            captured["unzip"] = unzip
            with zipfile.ZipFile(tmp_path / "kaggle" / "download.zip", "w") as archive:
                archive.writestr("rows.csv", "value\n1\n")

    kaggle_module = ModuleType("kaggle")
    kaggle_api_module = ModuleType("kaggle.api")
    kaggle_extended_module = ModuleType("kaggle.api.kaggle_api_extended")
    kaggle_extended_module.KaggleApi = FakeKaggleApi
    monkeypatch.setitem(sys.modules, "kaggle", kaggle_module)
    monkeypatch.setitem(sys.modules, "kaggle.api", kaggle_api_module)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", kaggle_extended_module)

    destination = tmp_path / "kaggle"
    resolved = _resolve_kaggle_dataset_source("owner/dataset", destination)

    assert resolved == destination.resolve()
    assert captured == {
        "authenticated": True,
        "identifier": "owner/dataset",
        "path": str(destination.resolve()),
        "unzip": False,
    }
    assert (destination / "rows.csv").read_text(encoding="utf-8") == "value\n1\n"
    assert not (destination / "download.zip").exists()


def test_kaggle_cli_resolver_downloads_then_safe_extracts(tmp_path, monkeypatch):
    captured = {}

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "kaggle.api.kaggle_api_extended":
            raise ImportError("blocked kaggle api")
        return real_import(name, *args, **kwargs)

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        output_path = command[command.index("-p") + 1]
        with zipfile.ZipFile(tmp_path / "kaggle_cli" / "download.zip", "w") as archive:
            archive.writestr("rows.csv", "value\n1\n")
        assert output_path == str((tmp_path / "kaggle_cli").resolve())

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr("matheel.datasets.shutil.which", lambda name: "kaggle" if name == "kaggle" else None)
    monkeypatch.setattr("matheel.datasets.subprocess.run", fake_run)

    destination = tmp_path / "kaggle_cli"
    resolved = _resolve_kaggle_dataset_source("owner/dataset", destination)

    assert resolved == destination.resolve()
    assert "--unzip" not in captured["command"]
    assert captured["check"] is True
    assert (destination / "rows.csv").read_text(encoding="utf-8") == "value\n1\n"
    assert not (destination / "download.zip").exists()


def test_custom_source_and_preset_load_pair_dataset(tmp_path):
    dataset_root = tmp_path / "custom_pair"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "x"},
                {"file_id": "b", "text": "y"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": "negative"}]),
        metadata={"name": "custom_pair", "task_type": "plagiarism"},
    )
    captured = {}

    def resolver(identifier, destination=None, revision="main", token=None, split=None):
        captured["identifier"] = identifier
        captured["destination"] = destination
        captured["revision"] = revision
        captured["token"] = token
        captured["split"] = split
        return tmp_path / str(identifier)

    register_dataset_source("unit_source", resolver, overwrite=True)
    preset = register_dataset_preset(
        "unit_pair_preset",
        {"source": "unit_source", "identifier": "custom_pair", "task_families": ("pair",)},
        overwrite=True,
    )

    loaded = load_pair_datasets(["unit_pair_preset"])

    assert "unit_pair_preset" in available_dataset_presets()
    assert get_dataset_preset("unit_pair_preset") == preset
    assert available_dataset_presets_by_task("pair").count("unit_pair_preset") == 1
    assert len(loaded.pairs) == 1
    assert captured["identifier"] == "custom_pair"
    assert captured["revision"] == "main"


def test_load_pair_datasets_from_manifest_resolves_relative_paths(tmp_path):
    project_root = tmp_path / "project"
    raw_root = project_root / "raw_pairs"
    raw_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"left_code": "print('x')", "right_code": "print('x')", "label": 1},
            {"left_code": "print('x')", "right_code": "print('y')", "label": 0},
        ]
    ).to_csv(raw_root / "pairs.csv", index=False)
    manifest_path = project_root / "datasets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "pair",
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_pairs",
                        "name": "manifest_pairs",
                        "adapter": "auto_pair_tabular",
                        "adapted_destination": "normalized_pairs",
                        "adapter_options": {
                            "left_text_column": "left_code",
                            "right_text_column": "right_code",
                            "label_column": "label",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(manifest_path)
    loaded = load_pair_datasets_from_manifest(manifest_path)

    spec = manifest["datasets"][0]
    assert spec["identifier"] == raw_root.resolve()
    assert spec["adapted_destination"] == (project_root / "normalized_pairs").resolve()
    assert loaded.metadata["name"] == "manifest_pairs"
    assert loaded.pairs["label"].tolist() == [1, 0]


def test_dataset_manifest_resolves_relative_normalized_dataset_paths(tmp_path):
    project_root = tmp_path / "project"
    dataset_root = project_root / "normalized_pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "x"},
                {"file_id": "b", "text": "x"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
    )
    manifest_path = project_root / "datasets.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "task": "pair", "datasets": ["normalized_pairs"]}),
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(manifest_path)
    loaded = load_pair_datasets_from_manifest(manifest_path)

    assert manifest["datasets"] == [dataset_root.resolve()]
    assert len(loaded.pairs) == 1


def test_load_retrieval_datasets_from_manifest_supports_yaml_when_available(tmp_path):
    yaml = pytest.importorskip("yaml")
    project_root = tmp_path / "retrieval_project"
    raw_root = project_root / "raw_retrieval"
    raw_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "document_id": "d1",
                "query_code": "print(1)",
                "candidate_code": "print(1)",
                "relevance": 1,
            },
            {
                "query_id": "q1",
                "document_id": "d2",
                "query_code": "print(1)",
                "candidate_code": "print(2)",
                "relevance": 0,
            },
        ]
    ).to_csv(raw_root / "retrieval.csv", index=False)
    manifest_path = project_root / "datasets.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "task": "retrieval",
                "dataset": {
                    "source": "local",
                    "identifier": "raw_retrieval",
                    "adapter": "auto_retrieval_tabular",
                    "adapted_destination": "normalized_retrieval",
                    "adapter_options": {
                        "retrieval_table": "retrieval.csv",
                        "query_text_column": "query_code",
                        "document_text_column": "candidate_code",
                        "relevance_column": "relevance",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    loaded = load_retrieval_datasets_from_manifest(manifest_path)

    assert loaded.queries["query_id"].tolist() == ["q1"]
    assert loaded.corpus["document_id"].tolist() == ["d1", "d2"]
    assert sorted(loaded.qrels["relevance"].tolist()) == [0.0, 1.0]


def test_dataset_manifest_rejects_credentials(tmp_path):
    manifest_path = tmp_path / "datasets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "api_key": "do-not-store",
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_pairs",
                        "adapter": "auto_pair_tabular",
                        "adapted_destination": "normalized_pairs",
                        "token": "do-not-store",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="api_key"):
        load_dataset_manifest(manifest_path)


def test_dataset_manifest_rejects_nested_credentials(tmp_path):
    manifest_path = tmp_path / "datasets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_pairs",
                        "adapter": "auto_pair_tabular",
                        "adapted_destination": "normalized_pairs",
                        "adapter_options": {"token": "do-not-store"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adapter_options.token"):
        load_dataset_manifest(manifest_path)


def test_dataset_manifest_requires_reproducible_adapter_destination(tmp_path):
    manifest_path = tmp_path / "datasets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_pairs",
                        "adapter": "auto_pair_tabular",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adapted_destination"):
        load_dataset_manifest(manifest_path)


def test_dataset_manifest_requires_remote_destination(tmp_path):
    manifest_path = tmp_path / "datasets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "source": "github",
                        "identifier": "owner/repo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remote sources must include destination"):
        load_dataset_manifest(manifest_path)


def test_custom_adapter_registry_can_adapt_pair_dataset(tmp_path):
    source_root = tmp_path / "raw"
    source_root.mkdir()

    def adapter(source_root, destination, dataset_name=None):
        assert source_root == source_root.resolve()
        return write_pair_dataset(
            destination,
            files=pd.DataFrame(
                [
                    {"file_id": "a", "text": "x"},
                    {"file_id": "b", "text": "x"},
                ]
            ),
            pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
            metadata={"name": dataset_name or "adapted"},
        ).root

    register_dataset_adapter("unit_pair_adapter", adapter, overwrite=True)
    adapted_root = adapt_pair_dataset(source_root, adapter="unit_pair_adapter", dataset_name="unit")
    loaded = load_pair_dataset(adapted_root)

    assert "unit_pair_adapter" in available_dataset_adapters()
    assert loaded.metadata["dataset_kind"] == "pair_classification"
    assert loaded.metadata["task_type"] == "plagiarism"
    assert loaded.pairs["label"].tolist() == [1]


def test_auto_pair_tabular_adapter_supports_configured_text_columns(tmp_path):
    source_root = tmp_path / "raw_pairs"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"lhs": "print('x')", "rhs": "print('x')", "target": "plagiarism"},
            {"lhs": "print('x')", "rhs": "print('y')", "target": "negative"},
        ]
    ).to_csv(source_root / "custom.csv", index=False)

    adapted_root = adapt_pair_dataset(
        source_root,
        adapter="auto_pair_tabular",
        dataset_name="custom_pairs",
        adapter_options={
            "pair_table": "custom.csv",
            "left_text_column": "lhs",
            "right_text_column": "rhs",
            "label_column": "target",
            "suffix": ".py",
        },
    )
    loaded = load_pair_dataset(adapted_root)
    texts = load_code_texts(loaded)

    assert len(loaded.pairs) == 2
    assert sorted(loaded.pairs["label"].tolist()) == [0, 1]
    assert loaded.metadata["adapter"] == "auto_pair_tabular"
    assert "print('x')" in set(texts.values())


def test_auto_pair_tabular_adapter_supports_path_columns(tmp_path):
    source_root = tmp_path / "raw_path_pairs"
    source_root.mkdir()
    (source_root / "a.py").write_text("print('a')\n", encoding="utf-8")
    (source_root / "b.py").write_text("print('b')\n", encoding="utf-8")
    pd.DataFrame(
        [{"left_file": "a.py", "right_file": "b.py", "is_match": False}]
    ).to_csv(source_root / "pairs.csv", index=False)

    adapted_root = adapt_pair_dataset(
        source_root,
        adapter="auto_pair_tabular",
        adapter_options={
            "left_path_column": "left_file",
            "right_path_column": "right_file",
            "label_column": "is_match",
        },
    )
    loaded = load_pair_dataset(adapted_root)
    texts = load_code_texts(loaded)

    assert loaded.pairs["label"].tolist() == [0]
    assert sorted(texts.values()) == ["print('a')\n", "print('b')\n"]


def test_auto_pair_tabular_adapter_preserves_explicit_column_errors(tmp_path):
    source_root = tmp_path / "raw_pairs"
    source_root.mkdir()
    pd.DataFrame(
        [{"left_code": "print('x')", "right_code": "print('x')", "label": 1}]
    ).to_csv(source_root / "pairs.csv", index=False)

    with pytest.raises(ValueError, match="left text column does not exist: missing_left"):
        adapt_pair_dataset(
            source_root,
            adapter="auto_pair_tabular",
            adapter_options={
                "pair_table": "pairs.csv",
                "left_text_column": "missing_left",
                "right_text_column": "right_code",
                "label_column": "label",
            },
        )


def test_auto_retrieval_tabular_adapter_supports_configured_columns(tmp_path):
    source_root = tmp_path / "raw_retrieval"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"qid": "q1", "did": "d1", "query": "print('x')", "candidate": "print('x')", "rel": 1},
            {"qid": "q1", "did": "d2", "query": "print('x')", "candidate": "print('y')", "rel": 0},
        ]
    ).to_csv(source_root / "retrieval.csv", index=False)

    adapted_root = adapt_pair_dataset(source_root, adapter=None)
    assert adapted_root == source_root.resolve()
    adapted_root = load_retrieval_datasets(
        [
            {
                "source": "local",
                "identifier": source_root,
                "name": "custom_retrieval",
                "adapter": "auto_retrieval_tabular",
                "adapter_options": {
                    "retrieval_table": "retrieval.csv",
                    "query_text_column": "query",
                    "document_text_column": "candidate",
                    "relevance_column": "rel",
                },
            }
        ]
    ).root
    loaded = load_retrieval_dataset(adapted_root)

    assert loaded.metadata["adapter"] == "auto_retrieval_tabular"
    assert loaded.queries["query_id"].tolist() == ["q1"]
    assert loaded.corpus["document_id"].tolist() == ["d1", "d2"]
    assert sorted(loaded.qrels["relevance"].tolist()) == [0.0, 1.0]


def test_auto_retrieval_tabular_adapter_preserves_explicit_column_errors(tmp_path):
    source_root = tmp_path / "raw_retrieval"
    source_root.mkdir()
    pd.DataFrame(
        [
            {
                "query_code": "print('x')",
                "candidate_code": "print('x')",
                "relevance": 1,
            }
        ]
    ).to_csv(source_root / "retrieval.csv", index=False)

    with pytest.raises(ValueError, match="document text column does not exist: missing_document"):
        adapt_retrieval_dataset(
            source_root,
            adapter="auto_retrieval_tabular",
            adapter_options={
                "retrieval_table": "retrieval.csv",
                "query_text_column": "query_code",
                "document_text_column": "missing_document",
                "relevance_column": "relevance",
            },
        )


def test_soco14_retrieval_adapter_uses_document_id_schema(tmp_path):
    source_root = tmp_path / "raw_soco14"
    split_root = source_root / "fire14-source-code-test-dataset_unzipped" / "fire14-source-code-test-dataset"
    (split_root / "c" / "A1").mkdir(parents=True)
    (split_root / "c" / "A1" / "C10172").write_text("int main(){return 0;}\n", encoding="utf-8")
    (split_root / "soco14-test-c-update.qrel").write_text(
        "C10172 C10172\n"
        "C10172 C99999\n",
        encoding="utf-8",
    )

    adapted_root = adapt_retrieval_dataset(
        source_root,
        adapter="soco14_retrieval",
        dataset_name="soco14_test",
        adapter_options={"split": "test"},
    )
    loaded = load_retrieval_dataset(adapted_root)

    assert loaded.metadata["task_type"] == "plagiarism"
    assert loaded.metadata["dataset_kind"] == "retrieval"
    assert loaded.metadata["qrel_rows_skipped_missing_files"] == 1
    assert "document_id" in loaded.corpus.columns
    assert "doc_id" not in loaded.corpus.columns
    assert len(loaded.qrels) == 1


def test_irplag_pair_and_retrieval_adapters_support_tabular_layout(tmp_path):
    source_root = tmp_path / "raw_irplag"
    source_root.mkdir()
    (source_root / "left.py").write_text("print('left')\n", encoding="utf-8")
    (source_root / "right.py").write_text("print('right')\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "submission_1": " FolderA\\Task1\\Left.py ",
                "submission_2": "folderB/task2/Right.py",
                "left_path": "left.py",
                "right_path": "right.py",
                "label": 1,
            }
        ]
    ).to_csv(source_root / "pairs.csv", index=False)

    pair_root = adapt_pair_dataset(source_root, adapter="irplag_pair", dataset_name="irplag")
    retrieval_root = adapt_retrieval_dataset(
        source_root,
        adapter="irplag_retrieval",
        dataset_name="irplag",
        adapter_options={
            "left_id_column": "submission_1",
            "right_id_column": "submission_2",
            "left_path_column": "left_path",
            "right_path_column": "right_path",
        },
    )
    pair_dataset = load_pair_dataset(pair_root)
    retrieval_dataset = load_retrieval_dataset(retrieval_root)

    assert pair_dataset.metadata["adapter"] == "irplag_pair"
    assert pair_dataset.pairs["label"].tolist() == [1]
    assert retrieval_dataset.metadata["adapter"] == "irplag_retrieval"
    assert retrieval_dataset.qrels["relevance"].tolist() == [1.0]
    assert retrieval_dataset.queries["source_id"].tolist() == ["foldera/task1/left.py"]
    assert retrieval_dataset.corpus["source_id"].tolist() == ["folderb/task2/right.py"]
    assert "document_id" in retrieval_dataset.qrels.columns


def test_conplag_pair_adapter_supports_nested_versions_layout(tmp_path):
    source_root = tmp_path / "conplag_source"
    versions_root = source_root / "conplag_unzipped" / "versions"
    version_1 = versions_root / "version_1" / "a_b"
    version_2 = versions_root / "version_2" / "c_d"
    version_1.mkdir(parents=True)
    version_2.mkdir(parents=True)
    (version_1 / "a.java").write_text("class A {}\n", encoding="utf-8")
    (version_1 / "b.java").write_text("class B {}\n", encoding="utf-8")
    (version_2 / "c.java").write_text("class C {}\n", encoding="utf-8")
    (version_2 / "d.java").write_text("class D {}\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"sub1": "a", "sub2": "b", "problem": 1, "verdict": 1},
            {"sub1": "c", "sub2": "d", "problem": 2, "verdict": 0},
        ]
    ).to_csv(versions_root / "labels.csv", index=False)

    adapted_root = adapt_pair_dataset(source_root, adapter="conplag_pair", dataset_name="conplag")
    loaded = load_pair_dataset(adapted_root)

    assert loaded.metadata["adapter"] == "conplag_pair"
    assert len(loaded.files) == 4
    assert sorted(loaded.pairs["label"].tolist()) == [0, 1]


def test_ipca_pair_adapter_derives_labels_from_txt_filenames(tmp_path):
    source_root = tmp_path / "ipca_source"
    ipc_root = source_root / "IPC" / "ITC"
    ipc_root.mkdir(parents=True)
    (ipc_root / "A1-NP-Sol-1.txt").write_text("original one\n", encoding="utf-8")
    (ipc_root / "A1-NP-Sol-1.ast").write_text("(ignored)\n", encoding="utf-8")
    (ipc_root / "A1-P-Sol-3.txt").write_text("plag base\n", encoding="utf-8")
    (ipc_root / "A1-P-Sol-3-comments.txt").write_text("plag comments\n", encoding="utf-8")
    (ipc_root / "A1-P-Sol-4.txt").write_text("other plag\n", encoding="utf-8")

    adapted_root = adapt_pair_dataset(source_root, adapter="ipca_pair", dataset_name="ipca")
    loaded = load_pair_dataset(adapted_root)

    assert loaded.metadata["adapter"] == "ipca_pair"
    assert loaded.metadata["ast_files_ignored"] is True
    assert len(loaded.files) == 4
    assert sorted(loaded.pairs["label"].tolist()) == [0, 0, 0, 0, 0, 1]
    assert set(loaded.pairs["course"]) == {"ITC"}


def test_student_code_similarity_pair_adapter_supports_kaggle_layout(tmp_path):
    source_root = tmp_path / "student_code_similarity"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"student1_code": "print(1)", "student2_code": "print(1)", "cheating": "cheating"},
            {"student1_code": "print(1)", "student2_code": "print(2)", "cheating": "not cheating"},
        ]
    ).to_csv(source_root / "pairs.csv", index=False)

    adapted_root = adapt_pair_dataset(
        source_root,
        adapter="student_code_similarity_pair",
        dataset_name="student_code_similarity",
    )
    loaded = load_pair_dataset(adapted_root)

    assert loaded.metadata["adapter"] == "student_code_similarity_pair"
    assert loaded.pairs["label"].tolist() == [1, 0]
    assert len(loaded.files) == 2


def test_criminal_minds_pair_adapter_matches_plagiarized_to_originals(tmp_path):
    source_root = tmp_path / "criminal_minds_source"
    code_root = source_root / "ReplicationPackage" / "code"
    orig_root = code_root / "orig"
    plag_root = code_root / "plag"
    (orig_root / "uni1-A-o1").mkdir(parents=True)
    (orig_root / "uni1-A-o2").mkdir(parents=True)
    (plag_root / "uni1-A-o1-p1").mkdir(parents=True)
    (orig_root / "uni1-A-o1" / "Main.java").write_text("class OriginalOne {}\n", encoding="utf-8")
    (orig_root / "uni1-A-o2" / "Main.java").write_text("class OriginalTwo {}\n", encoding="utf-8")
    (plag_root / "uni1-A-o1-p1" / "Main.java").write_text("class Plagiarized {}\n", encoding="utf-8")

    adapted_root = adapt_pair_dataset(source_root, adapter="criminal_minds_pair", dataset_name="criminal_minds")
    loaded = load_pair_dataset(adapted_root)

    assert loaded.metadata["adapter"] == "criminal_minds_pair"
    assert len(loaded.files) == 3
    assert sorted(loaded.pairs["label"].tolist()) == [0, 1]
    assert loaded.pairs.loc[loaded.pairs["label"] == 1, "original_id"].tolist() == ["o1"]


def test_load_retrieval_datasets_rejects_pair_only_preset(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "x"},
                {"file_id": "b", "text": "y"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
    )
    register_dataset_preset(
        "unit_pair_only_preset",
        {"source": "local", "identifier": dataset_root, "task_families": ("pair",)},
        overwrite=True,
    )

    with pytest.raises(ValueError, match="does not support retrieval"):
        load_retrieval_datasets(["unit_pair_only_preset"])


def test_pair_dataset_roundtrip_writes_manifest_files(tmp_path):
    dataset_root = tmp_path / "pairs"
    files = pd.DataFrame(
        [
            {"file_id": "a", "text": "print(1)", "suffix": ".py", "language": "python"},
            {"file_id": "b", "text": "print(1)", "suffix": ".py", "language": "python"},
            {"file_id": "c", "text": "print(2)", "suffix": ".py", "language": "python"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {"left_id": "a", "right_id": "b", "label": "plagiarism"},
            {"left_id": "a", "right_id": "c", "label": "negative"},
        ]
    )

    written = write_pair_dataset(
        dataset_root,
        files=files,
        pairs=pairs,
        metadata={"name": "tiny", "task_type": "plagiarism"},
    )
    loaded = load_pair_dataset(dataset_root)
    texts = load_code_texts(loaded)

    assert written.metadata["task_type"] == "plagiarism"
    assert loaded.metadata["dataset_kind"] == "pair_classification"
    assert loaded.pairs["label"].tolist() == [1, 0]
    assert texts == {"a": "print(1)", "b": "print(1)", "c": "print(2)"}
    assert (dataset_root / "files" / "a.py").exists()


def test_pair_dataset_rejects_unknown_file_reference(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame([{"file_id": "a", "text": "print(1)"}]),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "a", "label": 1}]),
    )
    pairs_path = dataset_root / "pairs.csv"
    pairs_path.write_text("left_id,right_id,label\na,missing,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown file ids: missing"):
        load_pair_dataset(dataset_root)


def test_pair_dataset_rejects_path_separator_file_ids(tmp_path):
    with pytest.raises(ValueError, match="path separators"):
        write_pair_dataset(
            tmp_path / "pairs",
            files=pd.DataFrame([{"file_id": "../a", "text": "print(1)"}]),
            pairs=pd.DataFrame([{"left_id": "../a", "right_id": "../a", "label": 1}]),
        )


def test_retrieval_dataset_roundtrip_writes_manifests(tmp_path):
    dataset_root = tmp_path / "retrieval"

    written = write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "doc_b", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame(
            [
                {"document_id": "d1", "file_id": "doc_a"},
                {"document_id": "d2", "file_id": "doc_b"},
            ]
        ),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
        metadata={"name": "tiny_retrieval_fixture", "task_type": "plagiarism"},
    )
    loaded = load_retrieval_dataset(dataset_root)
    texts = load_code_texts(loaded)

    assert written.metadata["dataset_kind"] == "retrieval"
    assert loaded.metadata["task_type"] == "plagiarism"
    assert loaded.queries["query_id"].tolist() == ["q1"]
    assert loaded.corpus["document_id"].tolist() == ["d1", "d2"]
    assert loaded.qrels["relevance"].tolist() == [1.0]
    assert texts["query_a"] == "print(1)"
    assert (dataset_root / "queries.csv").exists()
    assert (dataset_root / "corpus.csv").exists()
    assert (dataset_root / "qrels.csv").exists()


def test_retrieval_dataset_rejects_unknown_qrels_reference(tmp_path):
    dataset_root = tmp_path / "retrieval"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)"},
                {"file_id": "doc_a", "text": "print(1)"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "doc_a"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
    )
    qrels_path = dataset_root / "qrels.csv"
    qrels_path.write_text("query_id,document_id,relevance\nq1,missing,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown document ids: missing"):
        load_retrieval_dataset(dataset_root)
