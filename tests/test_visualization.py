import json

import pandas as pd
import pytest

import matheel
from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.visualization import (
    available_projection_methods,
    build_dataset_embedding_map,
    build_embedding_projection,
    dataset_map_html,
    dataset_map_payload,
    project_embeddings,
    write_dataset_embedding_map,
    write_dataset_map_artifacts,
)


def test_visualization_helpers_are_exported_from_package_root():
    assert matheel.available_projection_methods is available_projection_methods
    assert matheel.project_embeddings is project_embeddings
    assert matheel.build_dataset_embedding_map is build_dataset_embedding_map
    assert matheel.write_dataset_embedding_map is write_dataset_embedding_map


def test_project_embeddings_pca_is_deterministic():
    embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    first = project_embeddings(embeddings, method="pca", seed=7)
    second = project_embeddings(embeddings, method="pca", seed=99)

    pd.testing.assert_frame_equal(first, second)
    assert first.attrs["projection_method"] == "pca"
    assert first.attrs["requested_projection_method"] == "pca"
    assert first.attrs["embedding_count"] == 3
    assert first.shape == (3, 2)


def test_project_embeddings_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite"):
        project_embeddings([[1.0], [float("nan")]], method="pca")

    with pytest.raises(ValueError, match="same length"):
        build_embedding_projection([[1.0], [2.0]], ids=["a"], method="pca")


def test_build_embedding_projection_merges_metadata():
    projection = build_embedding_projection(
        [[1.0, 0.0], [0.0, 1.0]],
        ids=["a", "b"],
        metadata=[{"document_id": "a", "role": "query"}, {"document_id": "b", "role": "document"}],
        method="pca",
    )

    assert projection["document_id"].tolist() == ["a", "b"]
    assert projection["role"].tolist() == ["query", "document"]


def test_dataset_embedding_map_writes_pair_artifacts(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "def add(a, b): return a + b", "suffix": ".py"},
                {"file_id": "b", "text": "def add(x, y): return x + y", "suffix": ".py"},
                {"file_id": "c", "text": "def sub(a, b): return a - b", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
            ]
        ),
        metadata={"name": "tiny_pairs"},
    )

    projection, artifacts = write_dataset_embedding_map(
        dataset_root,
        tmp_path / "viz",
        kind="pair",
        method="pca",
        seed=7,
        static_vector_dim=32,
    )

    assert projection.attrs["dataset_name"] == "tiny_pairs"
    assert projection.attrs["dataset_kind"] == "pair_classification"
    assert projection.attrs["embedding_source"] == "static_hash"
    assert projection["document_id"].tolist() == ["a", "b", "c"]
    assert set(artifacts) == {"csv", "json", "html"}
    assert artifacts["csv"].exists()
    assert artifacts["json"].exists()
    assert artifacts["html"].exists()
    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert payload["metadata"]["projection_method"] == "pca"
    assert [point["document_id"] for point in payload["points"]] == ["a", "b", "c"]
    assert "tiny_pairs" in artifacts["html"].read_text(encoding="utf-8")


def test_dataset_embedding_map_marks_retrieval_roles(tmp_path):
    dataset_root = tmp_path / "retrieval"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "q_file", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d_file", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "q_file"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "d_file"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
        metadata={"name": "tiny_retrieval"},
    )

    projection = build_dataset_embedding_map(dataset_root, kind="retrieval", method="pca")

    assert projection.attrs["dataset_kind"] == "retrieval"
    assert projection.set_index("document_id").loc["q_file", "role"] == "query"
    assert projection.set_index("document_id").loc["d_file", "role"] == "document"


def test_dataset_map_html_escapes_document_ids():
    projection = build_embedding_projection(
        [[1.0, 0.0]],
        ids=["<script>"],
        metadata=[{"document_id": "<script>", "role": "submission"}],
        method="pca",
    )

    output = dataset_map_html(projection, title="<unsafe>")

    assert "<unsafe>" not in output
    assert "&lt;unsafe&gt;" in output
    assert "<script>" not in output
    assert "&lt;script&gt;" in output


def test_dataset_map_artifact_payload_requires_coordinates():
    with pytest.raises(ValueError, match="required columns"):
        dataset_map_payload(pd.DataFrame([{"document_id": "a"}]))


def test_write_dataset_map_artifacts_accepts_projection(tmp_path):
    projection = build_embedding_projection([[1.0, 0.0]], ids=["a"], method="pca")
    artifacts = write_dataset_map_artifacts(projection, tmp_path / "artifacts")

    assert artifacts["csv"].name == "dataset_map.csv"
    assert json.loads(artifacts["json"].read_text(encoding="utf-8"))["points"][0]["document_id"] == "a"
