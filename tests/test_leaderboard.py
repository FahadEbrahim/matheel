import json

import pandas as pd

import matheel
from matheel.benchmark_registry import (
    compare_benchmark_runs,
    list_benchmark_runs,
    load_benchmark_run,
    register_benchmark_run,
)
from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.leaderboard import (
    available_leaderboard_metrics,
    leaderboard_html,
    leaderboard_payload,
    load_leaderboard_manifest,
    normalize_leaderboard_manifest,
    run_leaderboard,
    write_leaderboard_artifacts,
)
from matheel.leaderboard_presets import (
    available_leaderboard_algorithm_presets,
    get_leaderboard_algorithm_preset,
    leaderboard_algorithm_preset_configs,
    register_leaderboard_algorithm_preset,
)
from matheel.reports import benchmark_detail_report_html


def test_leaderboard_helpers_are_exported_from_package_root():
    assert matheel.available_leaderboard_metrics is available_leaderboard_metrics
    assert matheel.leaderboard_html is leaderboard_html
    assert matheel.leaderboard_payload is leaderboard_payload
    assert matheel.load_leaderboard_manifest is load_leaderboard_manifest
    assert matheel.normalize_leaderboard_manifest is normalize_leaderboard_manifest
    assert matheel.run_leaderboard is run_leaderboard
    assert matheel.write_leaderboard_artifacts is write_leaderboard_artifacts
    assert matheel.available_leaderboard_algorithm_presets is available_leaderboard_algorithm_presets
    assert matheel.get_leaderboard_algorithm_preset is get_leaderboard_algorithm_preset
    assert matheel.leaderboard_algorithm_preset_configs is leaderboard_algorithm_preset_configs
    assert matheel.register_leaderboard_algorithm_preset is register_leaderboard_algorithm_preset
    assert matheel.benchmark_detail_report_html is benchmark_detail_report_html
    assert matheel.register_benchmark_run is register_benchmark_run


def test_leaderboard_algorithm_presets_include_offline_baselines():
    names = available_leaderboard_algorithm_presets()
    configs = leaderboard_algorithm_preset_configs(["Lexical Only", "Jaro-Winkler"])
    preset = get_leaderboard_algorithm_preset("jaro-winkler")

    assert "Jaro-Winkler" in names
    assert "Winnowing" in names
    assert "GST" in names
    assert "CodeBLEU" in names
    assert configs[0]["name"] == "Lexical Only"
    assert configs[1]["feature_weights"] == {"jaro_winkler": 1.0}
    assert preset["name"] == "Jaro-Winkler"


def test_register_leaderboard_algorithm_preset_for_custom_method():
    registered = register_leaderboard_algorithm_preset(
        "Jaro-Winkler",
        {
            "description": "test overwrite for same built-in preset",
            "similarity_options": {
                "feature_weights": {"jaro_winkler": 1.0},
                "code_metric": "codebleu",
                "code_metric_weight": 0.0,
            },
        },
        overwrite=True,
    )

    assert registered["name"] == "Jaro-Winkler"
    assert get_leaderboard_algorithm_preset("Jaro-Winkler")["similarity_options"]["feature_weights"] == {
        "jaro_winkler": 1.0
    }


def test_leaderboard_runs_pair_and_retrieval_datasets(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    retrieval_root = _write_retrieval_fixture(tmp_path)
    exact_path, inverse_path = _write_algorithm_fixtures(tmp_path)
    manifest = {
        "name": "tiny_leaderboard",
        "pair_metrics": ["f1", "auroc"],
        "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
        "datasets": [
            {"name": "pairs", "task": "pair", "path": str(pair_root), "threshold": 0.5},
            {"name": "retrieval", "task": "retrieval", "path": str(retrieval_root), "k": 2},
        ],
        "algorithms": [
            {"name": "exact", "algorithm_path": str(exact_path)},
            {"name": "inverse", "algorithm_path": str(inverse_path)},
        ],
    }

    report, artifacts = run_leaderboard(
        manifest,
        output_dir=tmp_path / "leaderboard",
        basename="tiny",
    )

    assert artifacts["per_dataset_csv"].exists()
    assert artifacts["aggregate_csv"].exists()
    assert artifacts["json"].exists()
    assert artifacts["html"].exists()
    assert artifacts["details_html"].exists()
    assert artifacts["reproducibility_json"].exists()
    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    reproducibility = json.loads(artifacts["reproducibility_json"].read_text(encoding="utf-8"))
    assert str(tmp_path) not in json.dumps(payload)
    assert str(tmp_path) not in json.dumps(reproducibility)
    assert payload["manifest"]["datasets"][0]["spec"]["identifier"] == "pairs"
    assert payload["manifest"]["algorithms"][0]["algorithm_path"] == "exact.py"
    assert payload["cards"]["datasets"][0]["name"] == "pairs"
    assert payload["cards"]["algorithms"][0]["name"] == "exact"
    per_dataset = report["per_dataset"]
    aggregate = report["aggregate"]
    pair_f1 = per_dataset[
        (per_dataset["task_family"] == "pair")
        & (per_dataset["dataset_name"] == "pairs")
        & (per_dataset["metric"] == "f1")
    ].set_index("algorithm_name")
    retrieval_map = per_dataset[
        (per_dataset["task_family"] == "retrieval")
        & (per_dataset["dataset_name"] == "retrieval")
        & (per_dataset["metric"] == "mean_average_precision")
    ].set_index("algorithm_name")

    assert pair_f1.loc["exact", "rank"] == 1
    assert pair_f1.loc["exact", "score"] == 1.0
    assert pair_f1.loc["inverse", "score"] == 0.0
    assert retrieval_map.loc["exact", "rank"] == 1
    assert retrieval_map.loc["exact", "score"] == 1.0
    assert retrieval_map.loc["inverse", "score"] < 1.0
    assert set(aggregate["algorithm_name"]) == {"exact", "inverse"}
    details_html = artifacts["details_html"].read_text(encoding="utf-8")
    assert str(tmp_path) not in details_html
    assert "Dataset Details" in details_html
    assert "Algorithm Details" in details_html


def test_leaderboard_runs_adapter_backed_pair_dataset(tmp_path):
    source_root = tmp_path / "raw_pairs"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"left_text": "print(1)", "right_text": "print(1)", "label": 1},
            {"left_text": "print(1)", "right_text": "print(2)", "label": 0},
        ]
    ).to_csv(source_root / "pairs.csv", index=False)

    report, artifacts = run_leaderboard(
        {
            "name": "adapter_backed",
            "datasets": [
                {
                    "name": "raw_pairs",
                    "task": "pair",
                    "path": str(source_root),
                    "adapter": "auto_pair_tabular",
                    "adapted_destination": str(tmp_path / "adapted_pairs"),
                }
            ],
            "algorithms": [{"name": "levenshtein", "feature_weights": {"levenshtein": 1.0}}],
        },
        output_dir=tmp_path / "leaderboard",
    )

    assert artifacts["json"].exists()
    assert report["cards"]["datasets"][0]["counts"]["pairs"] == 2
    assert not report["per_dataset"].empty


def test_leaderboard_manifest_accepts_algorithm_presets(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    manifest = {
        "name": "preset_leaderboard",
        "datasets": [{"name": "pairs", "task": "pair", "path": str(pair_root)}],
        "algorithms": [
            "Lexical Only",
            {"preset": "Jaro-Winkler", "name": "jw"},
        ],
    }

    report, _ = run_leaderboard(manifest)
    payload = leaderboard_payload(report)

    assert set(report["aggregate"]["algorithm_name"]) == {"Lexical Only", "jw"}
    assert payload["manifest"]["algorithms"][0]["name"] == "Lexical Only"
    assert payload["manifest"]["algorithms"][1]["similarity_options"]["feature_weights"] == {
        "jaro_winkler": 1.0
    }


def test_leaderboard_manifest_resolves_relative_paths(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    exact_path, _ = _write_algorithm_fixtures(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    manifest_path = config_dir / "leaderboard.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "pairs", "task": "pair", "path": "../pairs"},
                ],
                "algorithms": [
                    {"name": "exact", "algorithm_path": "../exact.py"},
                ],
            }
        ),
        encoding="utf-8",
    )

    _ = pair_root
    manifest = load_leaderboard_manifest(manifest_path)

    assert manifest["datasets"][0]["spec"]["identifier"] == str(pair_root.resolve())
    assert manifest["algorithms"][0]["algorithm_path"] == str(exact_path.resolve())


def test_leaderboard_payload_and_html_escape_values(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    exact_path, _ = _write_algorithm_fixtures(tmp_path)
    report, _ = run_leaderboard(
        {
            "name": "<unsafe>",
            "datasets": [{"name": "<pairs>", "task": "pair", "path": str(pair_root)}],
            "algorithms": [{"name": "<exact>", "algorithm_path": str(exact_path)}],
        }
    )

    payload = leaderboard_payload(report)
    output = leaderboard_html(report, title="<unsafe>")

    assert payload["metadata"]["name"] == "<unsafe>"
    assert payload["manifest"]["datasets"][0]["spec"]["identifier"] == "pairs"
    assert payload["manifest"]["algorithms"][0]["algorithm_path"] == "exact.py"
    assert "<unsafe>" not in output
    assert "&lt;unsafe&gt;" in output
    assert "<exact>" not in output


def test_benchmark_detail_report_escapes_values(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    exact_path, _ = _write_algorithm_fixtures(tmp_path)
    report, _ = run_leaderboard(
        {
            "name": "<unsafe>",
            "datasets": [{"name": "<pairs>", "task": "pair", "path": str(pair_root)}],
            "algorithms": [{"name": "<exact>", "algorithm_path": str(exact_path)}],
        }
    )

    output = benchmark_detail_report_html(report, title="<unsafe>")

    assert str(tmp_path) not in output
    assert "<unsafe>" not in output
    assert "&lt;unsafe&gt;" in output
    assert "<exact>" not in output
    assert "&lt;exact&gt;" in output


def test_benchmark_registry_tracks_and_compares_runs(tmp_path):
    pair_root = _write_pair_fixture(tmp_path)
    exact_path, inverse_path = _write_algorithm_fixtures(tmp_path)
    first_report, first_artifacts = run_leaderboard(
        {
            "name": "first",
            "datasets": [{"name": "pairs", "task": "pair", "path": str(pair_root)}],
            "algorithms": [{"name": "exact", "algorithm_path": str(exact_path)}],
        },
        output_dir=tmp_path / "first",
        basename="first",
    )
    second_report, _ = run_leaderboard(
        {
            "name": "second",
            "datasets": [{"name": "pairs", "task": "pair", "path": str(pair_root)}],
            "algorithms": [{"name": "inverse", "algorithm_path": str(inverse_path)}],
        },
        output_dir=tmp_path / "second",
        basename="second",
    )
    registry_path = tmp_path / "registry.json"

    first_entry = register_benchmark_run(registry_path, first_report, artifact_paths=first_artifacts)
    second_entry = register_benchmark_run(registry_path, second_report)
    runs = list_benchmark_runs(registry_path)
    loaded = load_benchmark_run(registry_path, first_entry["run_id"])
    comparison = compare_benchmark_runs(registry_path, run_ids=[first_entry["run_id"], second_entry["run_id"]])
    registry_text = registry_path.read_text(encoding="utf-8")

    assert list(runs["run_id"]) == [first_entry["run_id"], second_entry["run_id"]]
    assert loaded["artifacts"]["html"] == "first.html"
    assert str(tmp_path) not in registry_text
    assert set(comparison["run_id"]) == {first_entry["run_id"], second_entry["run_id"]}
    assert "delta_mean_score" in comparison


def test_leaderboard_payload_keeps_remote_identifiers():
    payload = leaderboard_payload(
        {
            "metadata": {"name": "remote"},
            "manifest": {
                "datasets": [
                    {
                        "name": "remote_pairs",
                        "task_family": "pair",
                        "spec": {"source": "github", "identifier": "owner/repository"},
                    }
                ],
                "algorithms": [{"name": "builtin", "similarity_options": {}}],
            },
            "cards": {"datasets": [], "algorithms": []},
            "per_dataset": pd.DataFrame(),
            "aggregate": pd.DataFrame(),
        }
    )

    assert payload["manifest"]["datasets"][0]["spec"]["identifier"] == "owner/repository"


def _write_pair_fixture(tmp_path):
    pair_root = tmp_path / "pairs"
    write_pair_dataset(
        pair_root,
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
        metadata={"name": "pairs"},
    )
    return pair_root


def _write_retrieval_fixture(tmp_path):
    retrieval_root = tmp_path / "retrieval"
    write_retrieval_dataset(
        retrieval_root,
        files=pd.DataFrame(
            [
                {"file_id": "q", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d1", "text": "print(1)", "suffix": ".py"},
                {"file_id": "d2", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "q"}]),
        corpus=pd.DataFrame(
            [
                {"document_id": "doc1", "file_id": "d1"},
                {"document_id": "doc2", "file_id": "d2"},
            ]
        ),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "doc1", "relevance": 1}]),
        metadata={"name": "retrieval"},
    )
    return retrieval_root


def _write_algorithm_fixtures(tmp_path):
    exact_path = tmp_path / "exact.py"
    exact_path.write_text(
        "def score_pair(code_a, code_b):\n    return 1.0 if code_a == code_b else 0.0\n",
        encoding="utf-8",
    )
    inverse_path = tmp_path / "inverse.py"
    inverse_path.write_text(
        "def score_pair(code_a, code_b):\n    return 0.0 if code_a == code_b else 1.0\n",
        encoding="utf-8",
    )
    return exact_path, inverse_path
