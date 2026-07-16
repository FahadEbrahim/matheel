import pandas as pd

from matheel.datasets import (
    load_pair_dataset,
    load_retrieval_dataset,
    write_pair_dataset,
    write_retrieval_dataset,
)
from scripts.build_ready_leaderboard import _sample_pair_dataset, _sample_retrieval_dataset


def test_ready_leaderboard_pair_sample_is_deterministic_and_keeps_both_labels(tmp_path):
    source = tmp_path / "source_pairs"
    files = [
        {"file_id": f"f{index}", "text": f"print({index})", "suffix": ".py"}
        for index in range(10)
    ]
    pairs = [
        {"left_id": f"f{index}", "right_id": f"f{index + 1}", "label": index % 2}
        for index in range(8)
    ]
    dataset = write_pair_dataset(source, files=pd.DataFrame(files), pairs=pd.DataFrame(pairs))
    metadata = {"name": "sample", "benchmark_sample": {"seed": 7}}

    first = _sample_pair_dataset(
        dataset,
        tmp_path / "first",
        limit=4,
        seed=7,
        metadata=metadata,
    )
    second = _sample_pair_dataset(
        dataset,
        tmp_path / "second",
        limit=4,
        seed=7,
        metadata=metadata,
    )
    first_dataset = load_pair_dataset(first)
    second_dataset = load_pair_dataset(second)

    assert first_dataset.pairs.equals(second_dataset.pairs)
    assert sorted(first_dataset.pairs["label"].unique()) == [0, 1]
    assert len(first_dataset.pairs) == 4
    assert first_dataset.metadata["benchmark_sample"]["source_pairs"] == 8


def test_ready_leaderboard_retrieval_sample_keeps_judged_documents(tmp_path):
    source = tmp_path / "source_retrieval"
    files = [
        {"file_id": f"q{index}", "text": f"query {index}", "suffix": ".txt"}
        for index in range(4)
    ] + [
        {"file_id": f"d{index}", "text": f"document {index}", "suffix": ".txt"}
        for index in range(8)
    ]
    queries = pd.DataFrame(
        [{"query_id": f"query{index}", "file_id": f"q{index}"} for index in range(4)]
    )
    corpus = pd.DataFrame(
        [{"document_id": f"doc{index}", "file_id": f"d{index}"} for index in range(8)]
    )
    qrels = pd.DataFrame(
        [
            {"query_id": f"query{index}", "document_id": f"doc{index}", "relevance": 1}
            for index in range(4)
        ]
    )
    dataset = write_retrieval_dataset(
        source,
        files=pd.DataFrame(files),
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )

    sampled_path = _sample_retrieval_dataset(
        dataset,
        tmp_path / "sampled",
        query_limit=2,
        corpus_limit=3,
        seed=7,
        metadata={"name": "sample", "benchmark_sample": {"seed": 7}},
    )
    sampled = load_retrieval_dataset(sampled_path)

    assert len(sampled.queries) == 2
    assert len(sampled.corpus) == 3
    assert set(sampled.qrels["document_id"]).issubset(set(sampled.corpus["document_id"]))
    assert set(sampled.qrels["query_id"]) == set(sampled.queries["query_id"])
