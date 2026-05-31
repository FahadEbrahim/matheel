import json

import pandas as pd

import matheel
from matheel.benchmark_cards import (
    algorithm_card,
    card_markdown,
    dataset_card,
    leaderboard_cards,
)
from matheel.datasets import write_pair_dataset


def test_benchmark_card_helpers_are_exported_from_package_root():
    assert matheel.algorithm_card is algorithm_card
    assert matheel.card_markdown is card_markdown
    assert matheel.dataset_card is dataset_card
    assert matheel.leaderboard_cards is leaderboard_cards


def test_dataset_card_uses_metadata_counts_and_fingerprint(tmp_path):
    dataset_root = _write_pair_fixture(tmp_path)

    card = dataset_card(dataset_root, source_spec={"identifier": str(dataset_root), "token": "secret"})

    assert card["card_type"] == "dataset"
    assert card["name"] == "tiny_pairs"
    assert card["task_family"] == "pair"
    assert card["dataset_kind"] == "pair_classification"
    assert card["counts"]["pairs"] == 1
    assert card["fingerprint"]["sha256"]
    assert card["source"]["identifier"] == "pairs"
    assert card["source"]["token"] == "<redacted>"
    assert str(tmp_path) not in json.dumps(card)


def test_algorithm_card_sanitizes_paths_and_fingerprints_custom_algorithm(tmp_path):
    algorithm_path = tmp_path / "custom_algo.py"
    algorithm_path.write_text("def score_pair(code_a, code_b):\n    return 1.0\n", encoding="utf-8")

    card = algorithm_card(
        {
            "name": "custom",
            "algorithm_path": str(algorithm_path),
            "algorithm_options": {"bias": 0.1},
            "similarity_options": {"preprocess_mode": "basic"},
        }
    )

    assert card["card_type"] == "algorithm"
    assert card["name"] == "custom"
    assert card["algorithm_kind"] == "custom"
    assert card["algorithm_path_name"] == "custom_algo.py"
    assert card["fingerprint"]["sha256"]
    assert card["algorithm_options"] == {"bias": 0.1}
    assert str(tmp_path) not in json.dumps(card)


def test_algorithm_card_sanitizes_windows_style_paths():
    card = algorithm_card({"algorithm_path": r"C:\Users\alice\secret\algo.py"})
    text = json.dumps(card)

    assert card["name"] == "algo"
    assert card["algorithm_path_name"] == "algo.py"
    assert card["fingerprint"]["file_name"] == "algo.py"
    assert "alice" not in text
    assert "secret" not in text


def test_algorithm_card_defaults_for_builtin_config():
    card = algorithm_card({"name": "levenshtein", "feature_weights": {"levenshtein": 1.0}})

    assert card["algorithm_kind"] == "builtin"
    assert card["algorithm_path_name"] == ""
    assert card["similarity_options"]["feature_weights"] == {"levenshtein": 1.0}
    assert card["fingerprint"] == {}


def test_leaderboard_cards_and_markdown(tmp_path):
    dataset_root = _write_pair_fixture(tmp_path)
    algorithm_path = tmp_path / "custom_algo.py"
    algorithm_path.write_text("def score_pair(code_a, code_b):\n    return 1.0\n", encoding="utf-8")
    cards = leaderboard_cards(
        {
            "datasets": [
                {
                    "name": "pairs",
                    "task_family": "pair",
                    "spec": {"identifier": str(dataset_root), "task_families": ("pair",)},
                }
            ],
            "algorithms": [
                {"name": "custom", "algorithm_path": str(algorithm_path), "similarity_options": {}}
            ],
        }
    )

    assert cards["datasets"][0]["name"] == "pairs"
    assert cards["algorithms"][0]["name"] == "custom"
    assert "# Dataset: pairs" in card_markdown(cards["datasets"][0])


def test_leaderboard_cards_support_adapter_backed_dataset_specs(tmp_path):
    source_root = tmp_path / "raw"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"left_text": "print(1)", "right_text": "print(1)", "label": 1},
            {"left_text": "print(1)", "right_text": "print(2)", "label": 0},
        ]
    ).to_csv(source_root / "pairs.csv", index=False)

    cards = leaderboard_cards(
        {
            "datasets": [
                {
                    "name": "raw_pairs",
                    "task_family": "pair",
                    "spec": {
                        "identifier": str(source_root),
                        "adapter": "auto_pair_tabular",
                        "adapted_destination": str(tmp_path / "adapted"),
                        "task_families": ("pair",),
                    },
                }
            ],
            "algorithms": [{"name": "levenshtein", "similarity_options": {}}],
        }
    )

    assert cards["datasets"][0]["counts"]["pairs"] == 2


def _write_pair_fixture(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
        metadata={"name": "tiny_pairs", "license": "synthetic"},
    )
    return dataset_root
