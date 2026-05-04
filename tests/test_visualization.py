import json

import pandas as pd
import pytest

import matheel
from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.visualization import (
    available_pair_explanation_segment_modes,
    available_projection_methods,
    build_pair_dataset_explanation,
    build_dataset_embedding_map,
    build_embedding_projection,
    build_pair_explanation,
    dataset_map_html,
    dataset_map_payload,
    pair_explanation_html,
    project_embeddings,
    write_pair_dataset_explanation,
    write_dataset_embedding_map,
    write_dataset_map_artifacts,
    write_pair_explanation_artifacts,
)


def test_visualization_helpers_are_exported_from_package_root():
    assert matheel.available_projection_methods is available_projection_methods
    assert matheel.available_pair_explanation_segment_modes is available_pair_explanation_segment_modes
    assert matheel.project_embeddings is project_embeddings
    assert matheel.build_dataset_embedding_map is build_dataset_embedding_map
    assert matheel.write_dataset_embedding_map is write_dataset_embedding_map
    assert matheel.build_pair_explanation is build_pair_explanation


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


def test_build_pair_explanation_marks_high_medium_low_and_no_match_regions():
    explanation = build_pair_explanation(
        "same line\nvalue = total + 1\nx=1\nabcdef",
        "same line\nvalue = amount + 1\ny=2\nUVWXYZ",
        left_id="left.py",
        right_id="right.py",
        high_threshold=0.95,
        medium_threshold=0.7,
        low_threshold=0.3,
    )

    assert explanation["metadata"]["segment_mode"] == "line"
    assert explanation["metadata"]["thresholds"] == {"high": 0.95, "medium": 0.7, "low": 0.3}
    assert [segment["level"] for segment in explanation["left"]["segments"]] == [
        "high",
        "medium",
        "low",
        "none",
    ]
    assert [match["level"] for match in explanation["matches"]] == ["high", "medium", "low"]
    assert explanation["left"]["segments"][0]["match_id"] == "m1"
    assert explanation["right"]["segments"][3]["match_id"] is None


def test_pair_explanation_supports_token_and_chunk_metadata():
    token_explanation = build_pair_explanation(
        "return total + 1",
        "return amount + 1",
        segment_mode="token",
        low_threshold=0.1,
    )
    chunk_explanation = build_pair_explanation(
        "a\nb\nc\nd",
        "a\nx\nc\ny",
        segment_mode="chunk",
        chunk_size=2,
        low_threshold=0.1,
    )

    assert token_explanation["metadata"]["tokenizer"] == "regex_code_tokens"
    assert token_explanation["left"]["segments"][0]["text"] == "return"
    assert chunk_explanation["metadata"]["chunk_size"] == 2
    assert [segment["start_line"] for segment in chunk_explanation["left"]["segments"]] == [1, 3]


def test_pair_explanation_html_escapes_text_and_uses_level_classes():
    explanation = build_pair_explanation(
        "<script>alert(1)</script>\nsame",
        "<script>alert(2)</script>\nsame",
        left_id="<left.py>",
        right_id="right.py",
        low_threshold=0.1,
    )

    output = pair_explanation_html(explanation, title="<unsafe>")

    assert "<unsafe>" not in output
    assert "&lt;unsafe&gt;" in output
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "level-high" in output
    assert "level-medium" in output or "level-low" in output


def test_write_pair_explanation_artifacts_writes_json_and_html(tmp_path):
    explanation = build_pair_explanation("print(1)", "print(1)", left_id="a.py", right_id="b.py")
    artifacts = write_pair_explanation_artifacts(
        explanation,
        tmp_path / "pair",
        basename="a/b pair",
    )

    assert artifacts["json"].name == "a_b_pair.json"
    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert payload["matches"][0]["level"] == "high"
    assert artifacts["html"].read_text(encoding="utf-8").startswith("<!doctype html>")


def test_build_pair_dataset_explanation_selects_pair_by_index_and_ids(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "same\nleft only", "suffix": ".py"},
                {"file_id": "b", "text": "same\nright only", "suffix": ".py"},
                {"file_id": "c", "text": "different", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "c", "label": 0},
                {"left_id": "a", "right_id": "b", "label": 1},
            ]
        ),
        metadata={"name": "tiny_pairs"},
    )

    by_index = build_pair_dataset_explanation(dataset_root, pair_index=1)
    by_id, artifacts = write_pair_dataset_explanation(
        dataset_root,
        tmp_path / "pair_viz",
        left_id="a",
        right_id="b",
    )

    assert by_index["metadata"]["dataset_name"] == "tiny_pairs"
    assert by_index["metadata"]["label"] == 1
    assert by_id["metadata"]["left_id"] == "a"
    assert artifacts["json"].exists()
