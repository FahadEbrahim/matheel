import pandas as pd
import pytest

from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.evaluation import (
    evaluate_pair_dataset,
    evaluate_pair_resamples,
    evaluate_retrieval_dataset,
    evaluate_retrieval_resamples,
    pair_classification_metrics,
    retrieval_ranking_metrics,
    score_pair_dataset,
    score_retrieval_dataset,
)
from matheel.resampling import kfold_splits


def _write_tiny_pair_dataset(path):
    return write_pair_dataset(
        path,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
                {"file_id": "c", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "label": 1},
                {"left_id": "a", "right_id": "c", "label": 0},
            ]
        ),
    )


def _write_tiny_retrieval_dataset(path):
    return write_retrieval_dataset(
        path,
        files=pd.DataFrame(
            [
                {"file_id": "query_one", "text": "print(1)", "suffix": ".py"},
                {"file_id": "query_two", "text": "print(2)", "suffix": ".py"},
                {"file_id": "doc_one", "text": "print(1)", "suffix": ".py"},
                {"file_id": "doc_two", "text": "print(2)", "suffix": ".py"},
                {"file_id": "doc_three", "text": "print(3)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame(
            [
                {"query_id": "q1", "file_id": "query_one"},
                {"query_id": "q2", "file_id": "query_two"},
            ]
        ),
        corpus=pd.DataFrame(
            [
                {"document_id": "d1", "file_id": "doc_one"},
                {"document_id": "d2", "file_id": "doc_two"},
                {"document_id": "d3", "file_id": "doc_three"},
            ]
        ),
        qrels=pd.DataFrame(
            [
                {"query_id": "q1", "document_id": "d1", "relevance": 1},
                {"query_id": "q2", "document_id": "d2", "relevance": 1},
            ]
        ),
    )


def test_score_pair_dataset_uses_custom_scorer(tmp_path):
    dataset = _write_tiny_pair_dataset(tmp_path / "pairs")

    scored = score_pair_dataset(
        dataset,
        scorer=lambda left, right, pair: 1.0 if left == right else 0.0,
    )

    assert scored["similarity_score"].tolist() == [1.0, 0.0]
    assert scored["label"].tolist() == [1, 0]


def test_evaluate_pair_dataset_returns_scored_pairs_and_metrics(tmp_path):
    dataset = _write_tiny_pair_dataset(tmp_path / "pairs")

    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=0.5,
        scorer=lambda left, right, pair: 1.0 if left == right else 0.0,
    )

    assert len(scored) == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_pair_dataset_supports_custom_algorithm_module(tmp_path):
    dataset = _write_tiny_pair_dataset(tmp_path / "pairs")
    module_path = tmp_path / "algo.py"
    module_path.write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, bias=0.0):",
                "    return {'bias': float(bias), 'file_count': len(dataset.files)}",
                "",
                "def score_pair(code_a, code_b, dataset_context=None, row=None):",
                "    return (1.0 if code_a == code_b else 0.0) + dataset_context['bias']",
            ]
        ),
        encoding="utf-8",
    )

    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=0.5,
        algorithm=module_path,
        algorithm_options={"bias": 0.1},
    )

    assert scored["similarity_score"].tolist() == [1.1, 0.1]
    assert metrics["accuracy"] == 1.0
    assert scored.attrs["algorithm"]["algorithm_options"] == {"bias": 0.1}


def test_evaluate_pair_dataset_preprocesses_custom_algorithm_inputs(tmp_path):
    dataset = write_pair_dataset(
        tmp_path / "commented_pairs",
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)  # comment", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
    )

    def score_pair(code_a, code_b):
        return 1.0 if code_a == code_b else 0.0

    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=0.5,
        algorithm=score_pair,
        similarity_options={"preprocess_mode": "basic", "code_language": "python"},
    )

    assert scored["similarity_score"].tolist() == [1.0]
    assert metrics["accuracy"] == 1.0


def test_evaluate_pair_dataset_prepare_hook_receives_prepared_texts(tmp_path):
    dataset = write_pair_dataset(
        tmp_path / "prepared_pairs",
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)  # comment", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
    )
    module_path = tmp_path / "prepared_algo.py"
    module_path.write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, prepared_texts):",
                "    return {",
                "        'same': prepared_texts['a'] == prepared_texts['b'],",
                "        'files': len(dataset.files),",
                "    }",
                "",
                "def score_pair(code_a, code_b, dataset_context=None):",
                "    _ = (code_a, code_b)",
                "    return 1.0 if (",
                "        dataset_context['same'] and dataset_context['files'] == 2",
                "    ) else 0.0",
            ]
        ),
        encoding="utf-8",
    )

    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=0.5,
        algorithm=module_path,
        similarity_options={"preprocess_mode": "basic", "code_language": "python"},
    )

    assert scored["similarity_score"].tolist() == [1.0]
    assert metrics["accuracy"] == 1.0


def test_pair_classification_metrics_requires_score_column():
    with pytest.raises(ValueError, match="similarity_score"):
        pair_classification_metrics([{"label": 1}], threshold=0.5)


def test_score_retrieval_dataset_uses_custom_scorer(tmp_path):
    dataset = _write_tiny_retrieval_dataset(tmp_path / "retrieval")

    scored = score_retrieval_dataset(
        dataset,
        scorer=lambda query, document, row: 1.0 if query == document else 0.0,
    )

    assert len(scored) == 6
    assert scored.loc[scored["query_id"] == "q1", "similarity_score"].tolist() == [1.0, 0.0, 0.0]
    assert scored["relevance"].sum() == 2.0


def test_evaluate_retrieval_dataset_returns_ranking_metrics(tmp_path):
    dataset = _write_tiny_retrieval_dataset(tmp_path / "retrieval")

    scored, metrics = evaluate_retrieval_dataset(
        dataset,
        k=1,
        scorer=lambda query, document, row: 1.0 if query == document else 0.0,
    )

    assert len(scored) == 6
    assert metrics["query_count"] == 2
    assert metrics["result_count"] == 6
    assert metrics["relevant_count"] == 2
    assert metrics["mean_average_precision"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0


def test_evaluate_retrieval_dataset_supports_custom_algorithm_module(tmp_path):
    dataset = _write_tiny_retrieval_dataset(tmp_path / "retrieval")
    module_path = tmp_path / "retrieval_algo.py"
    module_path.write_text(
        "\n".join(
            [
                "def score_pair(code_a, code_b, boost=0.0, row=None):",
                "    return (1.0 if code_a == code_b else 0.0) + float(boost)",
            ]
        ),
        encoding="utf-8",
    )

    scored, metrics = evaluate_retrieval_dataset(
        dataset,
        k=1,
        algorithm=module_path,
        algorithm_options={"boost": 0.1},
    )

    assert len(scored) == 6
    assert scored.attrs["algorithm"]["algorithm_options"] == {"boost": 0.1}
    assert metrics["mean_average_precision"] == 1.0


def test_evaluate_retrieval_dataset_preprocesses_custom_algorithm_inputs(tmp_path):
    dataset = write_retrieval_dataset(
        tmp_path / "commented_retrieval",
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)  # comment", "suffix": ".py"},
                {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "doc_a"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
    )

    def score_pair(code_a, code_b):
        return 1.0 if code_a == code_b else 0.0

    scored, metrics = evaluate_retrieval_dataset(
        dataset,
        k=1,
        algorithm=score_pair,
        similarity_options={"preprocess_mode": "basic", "code_language": "python"},
    )

    assert scored["similarity_score"].tolist() == [1.0]
    assert metrics["mean_average_precision"] == 1.0


def test_evaluate_retrieval_dataset_prepare_hook_receives_prepared_texts(tmp_path):
    dataset = write_retrieval_dataset(
        tmp_path / "prepared_retrieval",
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)  # comment", "suffix": ".py"},
                {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "doc_a"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
    )
    module_path = tmp_path / "prepared_retrieval_algo.py"
    module_path.write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, prepared_texts):",
                "    return {",
                "        'same': prepared_texts['query_a'] == prepared_texts['doc_a'],",
                "        'queries': len(dataset.queries),",
                "    }",
                "",
                "def score_pair(code_a, code_b, dataset_context=None):",
                "    _ = (code_a, code_b)",
                "    return 1.0 if (",
                "        dataset_context['same'] and dataset_context['queries'] == 1",
                "    ) else 0.0",
            ]
        ),
        encoding="utf-8",
    )

    scored, metrics = evaluate_retrieval_dataset(
        dataset,
        k=1,
        algorithm=module_path,
        similarity_options={"preprocess_mode": "basic", "code_language": "python"},
    )

    assert scored["similarity_score"].tolist() == [1.0]
    assert metrics["mean_average_precision"] == 1.0


def test_retrieval_ranking_metrics_handles_queries_without_relevant_documents():
    metrics = retrieval_ranking_metrics(
        pd.DataFrame(
            [
                {"query_id": "q1", "document_id": "d1", "similarity_score": 1.0, "relevance": 0},
                {"query_id": "q1", "document_id": "d2", "similarity_score": 0.5, "relevance": 0},
            ]
        ),
        k=2,
    )

    assert metrics["query_count"] == 1
    assert metrics["relevant_count"] == 0
    assert metrics["mean_average_precision"] == 0.0
    assert metrics["mean_reciprocal_rank"] == 0.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["ndcg_at_k"] == 0.0


def test_retrieval_ranking_metrics_requires_score_column():
    with pytest.raises(ValueError, match="similarity_score"):
        retrieval_ranking_metrics([{"query_id": "q1", "document_id": "d1", "relevance": 1}])


def test_evaluate_pair_resamples_returns_fold_metrics_and_summary():
    scored_pairs = pd.DataFrame(
        [
            {"similarity_score": 0.9, "label": 1},
            {"similarity_score": 0.8, "label": 1},
            {"similarity_score": 0.4, "label": 0},
            {"similarity_score": 0.3, "label": 0},
        ]
    )
    splits = kfold_splits(len(scored_pairs), n_splits=2, shuffle=False)

    metrics, summary = evaluate_pair_resamples(scored_pairs, splits, threshold=0.5)

    assert metrics["split"].tolist() == ["fold_1", "fold_2"]
    assert metrics["accuracy"].tolist() == [1.0, 1.0]
    assert summary.loc[summary["metric"] == "accuracy", "mean"].item() == pytest.approx(1.0)


def test_evaluate_retrieval_resamples_uses_query_level_splits():
    scored_results = pd.DataFrame(
        [
            {"query_id": "q1", "document_id": "d1", "similarity_score": 1.0, "relevance": 1},
            {"query_id": "q1", "document_id": "d2", "similarity_score": 0.1, "relevance": 0},
            {"query_id": "q2", "document_id": "d1", "similarity_score": 0.2, "relevance": 0},
            {"query_id": "q2", "document_id": "d2", "similarity_score": 0.9, "relevance": 1},
        ]
    )
    splits = kfold_splits(["q1", "q2"], n_splits=2, shuffle=False)

    metrics, summary = evaluate_retrieval_resamples(scored_results, splits, k=1)

    assert metrics["query_count"].tolist() == [1, 1]
    assert metrics["mean_average_precision"].tolist() == [1.0, 1.0]
    assert summary.loc[summary["metric"] == "ndcg_at_k", "mean"].item() == pytest.approx(1.0)
