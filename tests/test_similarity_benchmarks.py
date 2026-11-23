import csv
import json

import pandas as pd
import pytest

from matheel.datasets import (
    adapt_pair_dataset,
    adapt_retrieval_dataset,
    available_dataset_adapters,
    load_code_texts,
    load_pair_dataset,
    load_retrieval_dataset,
    registered_datasets,
)
from matheel.evaluation import retrieval_ranking_metrics, score_pair_dataset
from matheel.leaderboard import run_leaderboard
from matheel.similarity_benchmark import run_similarity_benchmark
from matheel.similarity_datasets import similcode_score_slices


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _similcode_fixture(source):
    java = source / "java"
    java.mkdir(parents=True)
    (java / "Base.java").write_text("class Base { int value() { return 7; } }\n", encoding="utf-8")
    (java / "Other.java").write_text("class Other { void no() {} }\n", encoding="utf-8")
    rows, types = [], []
    for case, category, clone_type, right, low, high in ("A", "identical", "type1", "Base", 100, 100), (
        "B", "different", "unrelated", "Other", 0, 20,
    ):
        common = {"case_id": case, "language": "Java", "reference_group": "1"}
        rows.append({**common, "similarity_category": category, "code_a_id": "Base", "code_a_path": "java/Base.java",
                     "code_b_id": right, "code_b_path": f"java/{right}.java", "expected_similarity_min": low,
                     "expected_similarity_max": high})
        types.append({**common, "category": category, "clone_type": clone_type})
    _csv(source / "metadata.csv", rows)
    _csv(source / "tipos_clon.csv", types)


def test_similarity_adapters_are_discoverable_and_similcode_keeps_categories(tmp_path):
    tasks = {entry.name: entry.task_type for entry in registered_datasets()}
    assert tasks["similcode"] == tasks["poj104"] == tasks["xlcost"] == "code_similarity"
    assert tasks["codexglue_bcb"] == "code_clone"
    assert tasks["bigclonebench_v2"] == "code_clone" and tasks["aigcodeset"] == "ai_authorship"
    assert [entry.name for entry in registered_datasets(dataset_kind="clone_discovery")] == ["bigclonebench_v2"]
    assert "similcode_pair" in available_dataset_adapters()
    source = tmp_path / "download" / "SimilCode-release"
    _similcode_fixture(source)
    target = adapt_pair_dataset(source.parent, adapter="similcode_pair", destination=tmp_path / "prepared")
    dataset = load_pair_dataset(target)
    assert dataset.metadata["license"] == "CC BY 4.0"
    assert dataset.pairs["case_id"].tolist() == ["A", "B"]
    assert dataset.pairs["clone_type"].tolist() == ["type1", "unrelated"]
    assert dataset.pairs["similarity_category"].tolist() == ["identical", "different"]
    assert dataset.pairs["reference_group"].tolist() == ["java_1", "java_1"]
    assert dataset.pairs["right_id"].tolist() == ["Base_COPY", "Other"]
    texts = load_code_texts(dataset)
    assert texts["Base"] == texts["Base_COPY"]
    slices = similcode_score_slices(score_pair_dataset(dataset, scorer=lambda a, b, row: float(a == b)))
    type2 = slices.loc[slices["slice"].eq("type2")].iloc[0]
    assert type2["pair_count"] == 0 and pd.isna(type2["mean"])


def test_poj104_uses_nonself_same_problem_groups_and_map_at_r_is_opt_in(tmp_path):
    source = tmp_path / "source"
    examples = [
        {"index": str(i), "label": str(65 + i // 3), "code": f"int solved{i}() {{ return {i}; }}"}
        for i in range(6)
    ]
    _jsonl(source / "valid.jsonl", examples)
    root = adapt_retrieval_dataset(source, adapter="poj104_retrieval", destination=tmp_path / "out",
                                    adapter_options={"split": "dev"})
    dataset = load_retrieval_dataset(root)
    assert dataset.metadata["split"] == "validation"
    assert dataset.metadata["ranking_metric"] == "mean_average_precision_at_r"
    assert len(dataset.qrels) == 12
    assert not dataset.qrels["query_id"].eq(dataset.qrels["document_id"]).any()
    # Recreate the order of the official README example, append all other candidates.
    prefixes = {"0": "32", "1": "04", "2": "01", "4": "15", "3": "42", "5": "43"}
    rows = []
    for query, prefix in prefixes.items():
        ordering = list(prefix) + sorted(set(prefixes) - set(prefix) - {query})
        rows.extend({"query_id": query, "document_id": doc, "similarity_score": float(6 - pos)}
                    for pos, doc in enumerate(ordering))
    default = retrieval_ranking_metrics(rows, dataset.qrels)
    metrics = retrieval_ranking_metrics(rows, dataset.qrels, include_map_at_r=True)
    assert "mean_average_precision_at_r" not in default
    assert round(metrics["mean_average_precision_at_r"], 4) == 0.5833
    assert metrics["mean_average_precision"] > metrics["mean_average_precision_at_r"]
    result, _ = run_leaderboard({"datasets": [{"path": str(root), "name": "test", "task": "retrieval",
                                                "exclude_self_matches": True}],
                                 "algorithms": [{"name": "edit", "feature_weights": {"levenshtein": 1.0}}],
                                 "retrieval_metrics": ["mean_average_precision_at_r"]})
    assert result["per_dataset"]["metric"].tolist() == ["mean_average_precision_at_r"]


def test_xlcost_both_directions_keep_partition_and_filter_other_languages(tmp_path):
    source = tmp_path / "source"
    program = source / "XLCoST_data" / "retrieval" / "code2code_search" / "program_level"
    (source / "__MACOSX" / "code2code_search" / "program_level").mkdir(parents=True)

    def example(query, doc, left, right):
        return {"idx": f"{query}/{doc}", "docstring_tokens": [left, "NEW_LINE"], "code_tokens": [right, "NEW_LINE"]}

    a = example("1-Java", "1-Python", "class", "def")
    _jsonl(program / "Java" / "test.jsonl", [a, a, example("1-Java", "1-C++", "class", "struct"),
                                              example("2-Java", "2-Python", "private", "return")])
    _jsonl(program / "Java" / "train.jsonl", [example("2-Java", "2-Python", "private", "return")])
    _jsonl(program / "Python" / "test.jsonl", [example("1-Python", "1-Java", "def", "class")])
    java = load_retrieval_dataset(adapt_retrieval_dataset(source, adapter="xlcost_retrieval",
                                                           destination=tmp_path / "j2p"))
    python = load_retrieval_dataset(adapt_retrieval_dataset(
        source, adapter="xlcost_retrieval", destination=tmp_path / "p2j",
        adapter_options={"source_language": "Python", "target_language": "Java"},
    ))
    assert java.metadata["problem_overlap_by_partition"] == {"train": 1}
    assert java.metadata["direction"] == "java_to_python"
    assert java.qrels["document_id"].tolist() == ["1-Python", "2-Python"]
    assert load_code_texts(java)["1-Python"] == "def NEW_LINE"
    assert python.qrels["document_id"].tolist() == ["1-Java"]


def test_codexglue_bcb_retains_order_and_does_not_infer_clone_type(tmp_path):
    source = tmp_path / "raw" / "dataset"
    _jsonl(source / "data.jsonl", [{"idx": number, "func": f"void method{number}() {{}}"} for number in (1, 2, 3)])
    (source / "train.txt").write_text("3\t1\t1\n", encoding="utf-8")
    (source / "test.txt").write_text("2\t1\t1\n1\t2\t0\n", encoding="utf-8")
    root = adapt_pair_dataset(source.parent, adapter="codexglue_bcb_pair", destination=tmp_path / "bcb")
    dataset = load_pair_dataset(root)
    assert dataset.pairs[["left_id", "right_id", "label"]].values.tolist() == [["2", "1", 1], ["1", "2", 0]]
    assert "clone_type" not in dataset.pairs
    assert dataset.metadata["primary_metric"] == "f1"
    assert dataset.metadata["shared_methods_by_partition"] == {"train": 1}


def test_exact_poj_runner_replays_ties_and_outputs_digests(tmp_path):
    pytest.importorskip("sklearn")
    source = tmp_path / "source"
    _jsonl(source / "test.jsonl", [
        {"index": index, "label": label, "code": "int same() { return 0; }"}
        for index, label in (("d", "B"), ("c", "B"), ("b", "A"), ("a", "A"))
    ])
    options = {"batch_size": 1, "k": 1, "source_revision": "frozen-fixture"}
    first = run_similarity_benchmark("poj104_retrieval", source, tmp_path / "first", **options)
    replay = run_similarity_benchmark("poj104_retrieval", source, tmp_path / "replay", **options)
    assert first == replay
    assert first["results"][0]["metrics"]["mean_average_precision_at_r"] == 0.5
    assert first["results"][0]["metrics"]["scored_comparisons"] == 12
    scores = pd.read_csv(tmp_path / "first" / "scores.csv.gz")
    assert scores["document_id"].tolist() == ["b", "a", "a", "a"]
