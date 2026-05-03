import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir


# Configure Matplotlib before importing Matheel's evaluation stack.
_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

import pandas as pd  # noqa: E402

from matheel.datasets import (  # noqa: E402
    available_dataset_adapters,
    available_dataset_presets,
    available_dataset_presets_by_task,
    available_dataset_sources,
    load_pair_dataset,
    load_pair_datasets,
    load_retrieval_datasets,
    register_dataset_preset,
    register_dataset_source,
    write_pair_dataset,
)
from matheel.evaluation import evaluate_pair_dataset, evaluate_retrieval_dataset  # noqa: E402


def exact_match_scorer(left_text, right_text, row):
    _ = row
    return 1.0 if left_text.strip() == right_text.strip() else 0.0


def normalized_pair_dataset_example(workspace):
    dataset_root = workspace / "normalized_pairs"
    write_pair_dataset(
        dataset_root,
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
        metadata={"name": "tiny_pair_example"},
    )

    dataset = load_pair_dataset(dataset_root)
    scored, metrics = evaluate_pair_dataset(dataset, threshold=0.5, scorer=exact_match_scorer)

    print("Normalized pair dataset")
    print(scored[["left_id", "right_id", "label", "similarity_score"]].to_string(index=False))
    print({key: round(value, 4) for key, value in metrics.items()})
    print()


def tabular_pair_adapter_example(workspace):
    raw_root = workspace / "raw_pair_table"
    raw_root.mkdir()
    pd.DataFrame(
        [
            {"left_code": "return x + 1", "right_code": "return x + 1", "is_match": True},
            {"left_code": "return x + 1", "right_code": "return y - 1", "is_match": False},
        ]
    ).to_csv(raw_root / "pairs.csv", index=False)

    dataset = load_pair_datasets(
        [
            {
                "source": "local",
                "identifier": raw_root,
                "name": "tiny_tabular_pairs",
                "adapter": "auto_pair_tabular",
                "adapter_options": {
                    "left_text_column": "left_code",
                    "right_text_column": "right_code",
                    "label_column": "is_match",
                    "suffix": ".py",
                },
            }
        ]
    )
    scored, metrics = evaluate_pair_dataset(dataset, threshold=0.5, scorer=exact_match_scorer)

    print("Generic tabular pair adapter")
    print(scored[["left_id", "right_id", "label", "similarity_score"]].to_string(index=False))
    print({key: round(value, 4) for key, value in metrics.items()})
    print()


def tabular_retrieval_adapter_example(workspace):
    raw_root = workspace / "raw_retrieval_table"
    raw_root.mkdir()
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
            {
                "query_id": "q2",
                "document_id": "d2",
                "query_code": "print(2)",
                "candidate_code": "print(2)",
                "relevance": 1,
            },
        ]
    ).to_csv(raw_root / "retrieval.csv", index=False)

    dataset = load_retrieval_datasets(
        [
            {
                "source": "local",
                "identifier": raw_root,
                "name": "tiny_tabular_retrieval",
                "adapter": "auto_retrieval_tabular",
                "adapter_options": {
                    "retrieval_table": "retrieval.csv",
                    "query_text_column": "query_code",
                    "document_text_column": "candidate_code",
                    "relevance_column": "relevance",
                },
            }
        ]
    )
    scored, metrics = evaluate_retrieval_dataset(dataset, k=1, scorer=exact_match_scorer)

    print("Generic tabular retrieval adapter")
    print(
        scored[["query_id", "document_id", "relevance", "similarity_score"]]
        .sort_values(["query_id", "document_id"])
        .to_string(index=False)
    )
    print({key: round(value, 4) for key, value in metrics.items()})
    print()


def custom_source_and_preset_example(workspace):
    raw_root = workspace / "custom_source" / "course_pairs"
    raw_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"submission_a": "total = price + tax", "submission_b": "total = price + tax", "match": 1},
            {"submission_a": "total = price + tax", "submission_b": "total = price - discount", "match": 0},
        ]
    ).to_csv(raw_root / "pairs.csv", index=False)

    def resolve_course_dataset(identifier, destination=None, revision="main", token=None, split=None):
        _ = (destination, revision, token, split)
        return workspace / "custom_source" / identifier

    register_dataset_source("coursework_examples", resolve_course_dataset, overwrite=True)
    register_dataset_preset(
        "coursework_pairs",
        {
            "source": "coursework_examples",
            "identifier": "course_pairs",
            "adapter": "auto_pair_tabular",
            "adapter_options": {
                "left_text_column": "submission_a",
                "right_text_column": "submission_b",
                "label_column": "match",
            },
            "task_families": ("pair",),
        },
        overwrite=True,
    )

    dataset = load_pair_datasets(["coursework_pairs"])
    scored, metrics = evaluate_pair_dataset(dataset, threshold=0.5, scorer=exact_match_scorer)

    print("Custom source and preset")
    print(scored[["left_id", "right_id", "label", "similarity_score"]].to_string(index=False))
    print({key: round(value, 4) for key, value in metrics.items()})
    print()


def registry_snapshot():
    print("Dataset registries")
    print("Sources:", available_dataset_sources())
    print("Adapters:", available_dataset_adapters())
    print("Presets:", available_dataset_presets())
    print("Pair presets:", available_dataset_presets_by_task("pair"))
    print("Retrieval presets:", available_dataset_presets_by_task("retrieval"))
    print()


def main():
    registry_snapshot()
    with TemporaryDirectory(prefix="matheel_dataset_examples_") as temp_dir:
        workspace = Path(temp_dir)
        normalized_pair_dataset_example(workspace)
        tabular_pair_adapter_example(workspace)
        tabular_retrieval_adapter_example(workspace)
        custom_source_and_preset_example(workspace)


if __name__ == "__main__":
    main()
