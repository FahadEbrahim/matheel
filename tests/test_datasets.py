import pandas as pd
import pytest

from matheel.datasets import (
    adapt_pair_dataset,
    available_dataset_adapters,
    available_dataset_kinds,
    available_dataset_presets,
    available_dataset_presets_by_task,
    available_dataset_sources,
    available_dataset_task_types,
    get_dataset_entry,
    get_dataset_preset,
    load_code_texts,
    load_pair_dataset,
    load_pair_datasets,
    load_retrieval_dataset,
    load_retrieval_datasets,
    registered_datasets,
    register_dataset_adapter,
    register_dataset_entry,
    register_dataset_preset,
    register_dataset_source,
    resolve_dataset_source,
    write_retrieval_dataset,
    write_pair_dataset,
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
