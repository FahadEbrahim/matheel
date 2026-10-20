import json
import shutil

import pandas as pd
import pytest

pytest.importorskip("gradio")

from gradio_app.official_benchmark import (
    build_official_benchmark_view,
    load_official_benchmark_payload,
    update_official_benchmark_gradio,
    update_published_benchmark_gradio,
)
from matheel.benchmark_protocol import freeze_benchmark_protocol, run_benchmark_protocol
from matheel.datasets import write_pair_dataset, write_retrieval_dataset


@pytest.fixture
def published_benchmark(tmp_path):
    files = pd.DataFrame(
        [
            {"file_id": "a", "text": "print(1)", "suffix": ".py"},
            {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            {"file_id": "c", "text": "unrelated long program", "suffix": ".py"},
        ]
    )
    pair = tmp_path / "original-pairs"
    retrieval = tmp_path / "original-retrieval"
    write_pair_dataset(
        pair,
        files=files,
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
            ]
        ),
        metadata={"task_type": "plagiarism"},
    )
    write_retrieval_dataset(
        retrieval,
        files=files,
        queries=pd.DataFrame([{"query_id": "q", "file_id": "a"}]),
        corpus=pd.DataFrame(
            [
                {"document_id": "positive", "file_id": "b"},
                {"document_id": "negative", "file_id": "c"},
            ]
        ),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "positive", "relevance": 1}]),
        metadata={"task_type": "plagiarism"},
    )
    broken = tmp_path / "broken.py"
    broken.write_text(
        'def score_pair(code_a, code_b):\n    raise RuntimeError("unsupported fixture")\n'
    )
    protocol = freeze_benchmark_protocol(
        {
            "domain": "plagiarism",
            "seed": 7,
            "pair_metrics": ["f1", "auroc"],
            "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
            "datasets": [
                {
                    "name": name,
                    "task": task,
                    "path": str(path),
                    "source_url": "https://example.org/versioned-data",
                    "source_revision": "a" * 40,
                    "split": "test",
                    "selection": {"mode": "all_labeled_rows"},
                }
                for name, task, path in (
                    ("fixture_pair", "pair", pair),
                    ("fixture_retrieval", "retrieval", retrieval),
                )
            ],
        }
    )
    _, artifacts = run_benchmark_protocol(
        protocol,
        [
            {"name": "Levenshtein", "feature_weights": {"levenshtein": 1}},
            {"name": "Unsupported", "algorithm_path": str(broken)},
        ],
        output_dir=tmp_path / "published",
        basename="matheel_plagiarism_v1",
    )
    # A public result must be usable without the author's local data or code.
    shutil.rmtree(pair)
    shutil.rmtree(retrieval)
    broken.unlink()
    return artifacts["json"], protocol


def test_missing_official_results_show_pending_without_placeholder_scores(tmp_path):
    view = build_official_benchmark_view(artifact_path=tmp_path / "not-published.json")

    assert not view["available"]
    assert "Results pending publication" in view["summary"]
    assert view["aggregate"].empty
    assert view["per_dataset"].empty
    assert view["artifact"] is None


def test_official_results_validate_without_original_data_and_show_frozen_scope(published_benchmark):
    path, protocol = published_benchmark
    view = build_official_benchmark_view(artifact_path=path)

    assert view["available"]
    assert protocol["benchmark_id"] in view["summary"]
    assert protocol["protocol_id"] in view["summary"]
    assert "Equal weight per declared dataset" in view["summary"]
    assert view["task"] == "pair"
    assert view["metric"] == "f1"
    assert set(view["aggregate"]["task_family"]) == {"pair"}
    assert view["aggregate"].columns[:5].tolist() == [
        "rank", "algorithm_name", "mean_score", "coverage", "eligible",
    ]
    assert view["per_dataset"].columns[:5].tolist() == [
        "dataset_name", "rank", "algorithm_name", "score", "eligible",
    ]
    assert set(view["scope"]["Category"]) == {"Plagiarism"}
    assert set(view["scope"]["Evaluation Task"]) == {"Pair classification", "Retrieval"}
    assert set(view["scope"]["Source Revision"]) == {"a" * 40}
    assert set(view["configurations"]) == set(view["aggregate"]["algorithm_id"])
    assert (
        load_official_benchmark_payload(path)["metadata"]["protocol_id"] == protocol["protocol_id"]
    )


def test_official_task_change_uses_retrieval_metrics_and_preserves_failed_rows(published_benchmark):
    path, _ = published_benchmark
    view = build_official_benchmark_view("retrieval", "f1", artifact_path=path)

    assert view["metric"] == "mean_average_precision"
    assert "f1" not in view["metrics"]
    assert "Retrieval k" in view["summary"]
    assert "Threshold Policy" not in view["summary"]
    assert set(view["aggregate"]["task_family"]) == {"retrieval"}
    failed = view["aggregate"].query('algorithm_name == "Unsupported"')
    assert len(failed) == 1
    assert not failed["eligible"].any()
    assert failed["rank"].isna().all()
    assert failed["mean_score"].isna().all()
    assert failed["coverage"].eq(0).all()
    failures = view["per_dataset"].query('algorithm_name == "Unsupported"')
    assert failures["error"].str.contains("unsupported fixture").all()


def test_official_method_filter_preserves_full_identity_and_authoritative_rank(published_benchmark):
    path, _ = published_benchmark
    all_methods = build_official_benchmark_view(artifact_path=path)
    selected = all_methods["aggregate"].query('algorithm_name == "Levenshtein"').iloc[0]
    view = build_official_benchmark_view(method_id=selected["algorithm_id"], artifact_path=path)

    assert set(view["aggregate"]["algorithm_id"]) == {selected["algorithm_id"]}
    assert set(view["per_dataset"]["algorithm_id"]) == {selected["algorithm_id"]}
    assert view["aggregate"].iloc[0]["rank"] == selected["rank"]
    assert set(view["aggregate"]["protocol_id"]) == set(all_methods["aggregate"]["protocol_id"])


@pytest.mark.parametrize("field", ["rank", "mean_score", "protocol_id"])
def test_tampered_official_results_display_validation_error_and_no_rankings(
    published_benchmark, field
):
    path, _ = published_benchmark
    payload = json.loads(path.read_text())
    if field == "protocol_id":
        payload["metadata"][field] = "b" * 64
    else:
        payload["aggregate"][0][field] = 987
    path.write_text(json.dumps(payload))

    view = build_official_benchmark_view(artifact_path=path)

    assert not view["available"]
    assert "Artifact validation failed" in view["summary"]
    assert view["aggregate"].empty
    assert view["artifact"] is None


def test_official_refresh_updates_controls_from_the_validated_artifact(published_benchmark):
    path, _ = published_benchmark
    updates = update_official_benchmark_gradio("retrieval", "f1", "all", path)

    assert updates[1]["value"] == "retrieval"
    assert updates[1]["interactive"] is True
    assert updates[2]["value"] == "mean_average_precision"
    assert not updates[4].empty
    assert updates[-1] == str(path)


def test_edition_selector_uses_only_a_known_bundle_and_its_own_scope(published_benchmark, monkeypatch):
    from gradio_app import official_benchmark

    path, protocol = published_benchmark
    monkeypatch.setitem(official_benchmark.PUBLISHED_BENCHMARK_EDITIONS, "fixture", ("Fixture", path))
    updates = update_published_benchmark_gradio("fixture", "retrieval", "f1", "all")
    assert updates[2]["value"] == "mean_average_precision"
    assert updates[-1] == str(path)
    assert updates[7]["protocol_id"] == protocol["protocol_id"]
    assert "1800 seconds" in updates[0]
    with pytest.raises(ValueError, match="published benchmark edition"):
        update_published_benchmark_gradio(str(path), "pair", "f1", "all")
