import json
import os
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from click.testing import CliRunner
import pandas as pd

import matheel
import matheel.datasets as datasets_module
from matheel.cli import main
from matheel.datasets import register_dataset_preset, write_pair_dataset, write_retrieval_dataset


def test_cli_and_package_expose_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert version("matheel") in result.output
    assert matheel.__version__ == version("matheel")


def test_visualize_dataset_command_writes_artifacts(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(2)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
        metadata={"name": "tiny_pairs"},
    )
    output_dir = tmp_path / "viz"

    result = CliRunner().invoke(
        main,
        [
            "visualize-dataset",
            str(dataset_root),
            "--kind",
            "pair",
            "--method",
            "pca",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "projection_method=pca" in result.output
    assert (output_dir / "dataset_map.csv").exists()
    assert (output_dir / "dataset_map.json").exists()
    assert (output_dir / "dataset_map.html").exists()


def test_explain_pair_command_writes_file_pair_artifacts(tmp_path):
    left_path = tmp_path / "left.py"
    right_path = tmp_path / "right.py"
    left_path.write_text("same line\nx=1\nabcdef", encoding="utf-8")
    right_path.write_text("same line\ny=2\nUVWXYZ", encoding="utf-8")
    output_dir = tmp_path / "pair_viz"

    result = CliRunner().invoke(
        main,
        [
            "explain-pair",
            str(left_path),
            str(right_path),
            "--output-dir",
            str(output_dir),
            "--high-threshold",
            "0.95",
            "--medium-threshold",
            "0.7",
            "--low-threshold",
            "0.3",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["left_id"] == "left.py"
    assert summary["levels"]["high"] == 2
    assert (output_dir / "left.py_vs_right.py.json").exists()
    assert (output_dir / "left.py_vs_right.py.html").exists()


def test_explain_pair_command_accepts_dataset_pair_ids(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "same\nleft", "suffix": ".py"},
                {"file_id": "b", "text": "same\nright", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
        metadata={"name": "tiny_pairs"},
    )
    output_dir = tmp_path / "pair_viz"

    result = CliRunner().invoke(
        main,
        [
            "explain-pair",
            "--dataset",
            str(dataset_root),
            "--left-id",
            "a",
            "--right-id",
            "b",
            "--output-dir",
            str(output_dir),
            "--basename",
            "selected_pair",
        ],
    )

    assert result.exit_code == 0
    assert "left_id=a" in result.output
    assert (output_dir / "selected_pair.json").exists()
    payload = json.loads((output_dir / "selected_pair.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset_name"] == "tiny_pairs"
    assert payload["metadata"]["label"] == 1


def test_explain_pair_command_accepts_source_archive_names(tmp_path):
    archive_path = tmp_path / "codes.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "same\nx=1")
        archive.writestr("b.py", "same\ny=2")
    output_dir = tmp_path / "pair_viz"

    result = CliRunner().invoke(
        main,
        [
            "explain-pair",
            "--source",
            str(archive_path),
            "--left-name",
            "a.py",
            "--right-name",
            "b.py",
            "--output-dir",
            str(output_dir),
            "--segment-mode",
            "line",
        ],
    )

    assert result.exit_code == 0
    assert "matches=" in result.output
    assert (output_dir / "a.py_vs_b.py.json").exists()


def test_calibration_report_command_writes_artifacts(tmp_path):
    scores_path = tmp_path / "scored_pairs.csv"
    pd.DataFrame(
        [
            {"left_id": "a", "right_id": "b", "similarity_score": 0.9, "label": 1},
            {"left_id": "a", "right_id": "c", "similarity_score": 0.8, "label": 1},
            {"left_id": "d", "right_id": "e", "similarity_score": 0.3, "label": 0},
            {"left_id": "d", "right_id": "f", "similarity_score": 0.1, "label": 0},
        ]
    ).to_csv(scores_path, index=False)
    output_dir = tmp_path / "calibration"

    result = CliRunner().invoke(
        main,
        [
            "calibration-report",
            str(scores_path),
            "--output-dir",
            str(output_dir),
            "--basename",
            "tiny",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["auroc"] == 1.0
    assert summary["average_precision"] == 1.0
    assert (output_dir / "tiny_threshold_sweep.csv").exists()
    assert (output_dir / "tiny_roc.csv").exists()
    assert (output_dir / "tiny_precision_recall.csv").exists()
    assert (output_dir / "tiny_report.json").exists()


def test_compare_command_accepts_new_options(tmp_path, monkeypatch):
    archive_path = tmp_path / "codes.zip"
    archive_path.write_bytes(b"placeholder")

    captured = {}

    def fake_get_sim_list(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        frame = pd.DataFrame(
            [
                {
                    "file_name_1": "a.py",
                    "file_name_2": "b.py",
                    "similarity_score": 1.0,
                }
            ]
        )
        frame.attrs.update(
            {
                "elapsed_seconds": 1.2346,
                "feature_set": "code_metric,semantic",
                "vector_backend": "sentence_transformers",
                "code_metric": "codebleu",
                "chunking_method": "code",
            }
        )
        return frame

    monkeypatch.setattr("matheel.cli.get_sim_list", fake_get_sim_list)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(archive_path),
            "--feature-weight",
            "semantic=0.5",
            "--feature-weight",
            "code_metric=0.5",
            "--preprocess-mode",
            "basic",
            "--chunking-method",
            "code",
            "--chunk-size",
            "50",
            "--chunk-overlap",
            "10",
            "--max-chunks",
            "4",
            "--chunk-language",
            "python",
            "--chunker-option",
            "include_line_numbers=true",
            "--chunk-aggregation",
            "max",
            "--code-metric",
            "codebleu",
            "--code-metric-weight",
            "0.4",
            "--code-language",
            "python",
            "--codebleu-component-weights",
            "0.4,0.3,0.2,0.1",
            "--crystalbleu-max-order",
            "3",
            "--crystalbleu-trivial-ngram-count",
            "40",
            "--ruby-max-order",
            "5",
            "--ruby-mode",
            "string",
            "--ruby-tokenizer",
            "tranx",
            "--ruby-denominator",
            "mean",
            "--ruby-graph-timeout-seconds",
            "0.5",
            "--no-ruby-graph-use-edge-cost",
            "--no-ruby-graph-include-leaf-edges",
            "--ruby-tree-max-nodes",
            "90",
            "--ruby-tree-max-depth",
            "6",
            "--ruby-tree-max-children",
            "4",
            "--tsed-delete-cost",
            "2",
            "--tsed-insert-cost",
            "3",
            "--tsed-rename-cost",
            "4",
            "--codebertscore-model",
            "microsoft/codebert-base",
            "--codebertscore-num-layers",
            "7",
            "--codebertscore-batch-size",
            "8",
            "--codebertscore-max-length",
            "256",
            "--codebertscore-device",
            "cpu",
            "--codebertscore-idf",
            "--codebertscore-nthreads",
            "2",
            "--vector-backend",
            "auto",
            "--similarity-function",
            "dot",
            "--normalize-semantic-scores",
            "--winnowing-kgram",
            "6",
            "--winnowing-window",
            "5",
            "--gst-min-match-length",
            "4",
            "--lexical-tokenizer",
            "parser",
            "--static-vector-dim",
            "512",
            "--max-token-length",
            "128",
            "--pooling-method",
            "max",
            "--device",
            "cpu",
            "--progress",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["feature_weights"] == ("semantic=0.5", "code_metric=0.5")
    assert captured["kwargs"]["preprocess_mode"] == "basic"
    assert captured["kwargs"]["chunking_method"] == "code"
    assert captured["kwargs"]["chunk_size"] == 50
    assert captured["kwargs"]["chunk_overlap"] == 10
    assert captured["kwargs"]["max_chunks"] == 4
    assert captured["kwargs"]["chunk_language"] == "python"
    assert captured["kwargs"]["chunker_options"] == ("include_line_numbers=true",)
    assert captured["kwargs"]["chunk_aggregation"] == "max"
    assert captured["kwargs"]["code_metric"] == "codebleu"
    assert captured["kwargs"]["code_metric_weight"] == 0.4
    assert captured["kwargs"]["code_language"] == "python"
    assert captured["kwargs"]["codebleu_component_weights"] == "0.4,0.3,0.2,0.1"
    assert captured["kwargs"]["crystalbleu_max_order"] == 3
    assert captured["kwargs"]["crystalbleu_trivial_ngram_count"] == 40
    assert captured["kwargs"]["ruby_max_order"] == 5
    assert captured["kwargs"]["ruby_mode"] == "string"
    assert captured["kwargs"]["ruby_tokenizer"] == "tranx"
    assert captured["kwargs"]["ruby_denominator"] == "mean"
    assert captured["kwargs"]["ruby_graph_timeout_seconds"] == 0.5
    assert captured["kwargs"]["ruby_graph_use_edge_cost"] is False
    assert captured["kwargs"]["ruby_graph_include_leaf_edges"] is False
    assert captured["kwargs"]["ruby_tree_max_nodes"] == 90
    assert captured["kwargs"]["ruby_tree_max_depth"] == 6
    assert captured["kwargs"]["ruby_tree_max_children"] == 4
    assert captured["kwargs"]["tsed_delete_cost"] == 2.0
    assert captured["kwargs"]["tsed_insert_cost"] == 3.0
    assert captured["kwargs"]["tsed_rename_cost"] == 4.0
    assert captured["kwargs"]["codebertscore_model"] == "microsoft/codebert-base"
    assert captured["kwargs"]["codebertscore_num_layers"] == 7
    assert captured["kwargs"]["codebertscore_batch_size"] == 8
    assert captured["kwargs"]["codebertscore_max_length"] == 256
    assert captured["kwargs"]["codebertscore_device"] == "cpu"
    assert captured["kwargs"]["codebertscore_idf"] is True
    assert captured["kwargs"]["codebertscore_nthreads"] == 2
    assert captured["kwargs"]["vector_backend"] == "auto"
    assert captured["kwargs"]["similarity_function"] == "dot"
    assert captured["kwargs"]["normalize_semantic_scores"] is True
    assert captured["kwargs"]["winnowing_kgram"] == 6
    assert captured["kwargs"]["winnowing_window"] == 5
    assert captured["kwargs"]["gst_min_match_length"] == 4
    assert captured["kwargs"]["lexical_tokenizer"] == "parser"
    assert captured["kwargs"]["static_vector_dim"] == 512
    assert captured["kwargs"]["max_token_length"] == 128
    assert captured["kwargs"]["pooling_method"] == "max"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["progress"] is True
    assert "Elapsed: 1.2346s" in result.stderr
    assert "features=code_metric,semantic" in result.stderr


def test_compare_command_marks_backend_inactive_without_semantic_feature(tmp_path, monkeypatch):
    archive_path = tmp_path / "codes.zip"
    archive_path.write_bytes(b"placeholder")

    def fake_get_sim_list(*args, **kwargs):
        _ = (args, kwargs)
        frame = pd.DataFrame(
            [
                {
                    "file_name_1": "a.py",
                    "file_name_2": "b.py",
                    "similarity_score": 1.0,
                }
            ]
        )
        frame.attrs.update(
            {
                "elapsed_seconds": 0.5,
                "feature_set": "levenshtein",
                "vector_backend": "inactive",
                "code_metric": "none",
                "chunking_method": "none",
            }
        )
        return frame

    monkeypatch.setattr("matheel.cli.get_sim_list", fake_get_sim_list)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(archive_path),
            "--feature-weight",
            "levenshtein=1.0",
        ],
    )

    assert result.exit_code == 0
    assert "features=levenshtein" in result.stderr
    assert "backend=inactive" in result.stderr
    assert "backend=auto" not in result.stderr


def test_compare_command_accepts_directory_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")

    captured = {}

    def fake_get_sim_list(*args, **kwargs):
        captured["args"] = args
        return pd.DataFrame(
            [
                {
                    "file_name_1": "a.py",
                    "file_name_2": "b.py",
                    "similarity_score": 1.0,
                }
            ]
        )

    monkeypatch.setattr("matheel.cli.get_sim_list", fake_get_sim_list)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(source_dir)])

    assert result.exit_code == 0
    assert captured["args"][0] == str(source_dir)


def test_compare_command_loads_custom_algorithm_and_writes_reproducibility(tmp_path):
    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)\n", encoding="utf-8")
    (source_dir / "b.py").write_text("print(1)\n", encoding="utf-8")
    algorithm_path = tmp_path / "algo.py"
    algorithm_path.write_text(
        "\n".join(
            [
                "def score_pair(code_a, code_b, bias=0.0, **kwargs):",
                "    _ = kwargs",
                "    return (1.0 if code_a == code_b else 0.0) + float(bias)",
            ]
        ),
        encoding="utf-8",
    )
    reproducibility_path = tmp_path / "repro.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(source_dir),
            "--algorithm-path",
            str(algorithm_path),
            "--algorithm-option",
            "bias=0.2",
            "--reproducibility-out",
            str(reproducibility_path),
        ],
    )

    assert result.exit_code == 0
    assert "1.2" in result.output
    assert "features=custom" in result.stderr
    payload = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    assert payload["source"]["source_type"] == "directory"
    assert payload["run_metadata"]["algorithm"]["algorithm_options"] == {"bias": 0.2}
    assert len(payload["run_metadata"]["algorithm"]["algorithm_source_fingerprint"]["sha256"]) == 64


def test_compare_command_rejects_regular_file_source(tmp_path):
    source_file = tmp_path / "single.py"
    source_file.write_text("print(1)", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(source_file),
            "--feature-weight",
            "levenshtein=1",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "directory or a ZIP archive" in str(result.exception)


def test_compare_command_rejects_inactive_code_metric_weight(tmp_path):
    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")
    (source_dir / "b.py").write_text("print(1)", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            str(source_dir),
            "--feature-weight",
            "levenshtein=1",
            "--code-metric-weight",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "code_metric_weight requires an active code_metric" in str(result.exception)


def test_datasets_list_command_outputs_reproducible_json():
    runner = CliRunner()
    result = runner.invoke(main, ["datasets", "list", "--task", "retrieval", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["sources"] == sorted(payload["sources"])
    assert "local" in payload["sources"]
    assert "auto_retrieval_tabular" in payload["adapters"]
    assert payload["presets"]
    assert all("retrieval" in preset["task_families"] for preset in payload["presets"])


def test_datasets_validate_command_reports_pair_summary(tmp_path):
    dataset_root = tmp_path / "pairs"
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
        metadata={"name": "unit_pairs"},
    )

    runner = CliRunner()
    result = runner.invoke(main, ["datasets", "validate", str(dataset_root), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "unit_pairs"
    assert payload["dataset_kind"] == "pair_classification"
    assert payload["counts"] == {
        "files": 3,
        "negative_pairs": 1,
        "pairs": 2,
        "positive_pairs": 1,
    }


def test_datasets_adapt_command_writes_pair_dataset(tmp_path):
    source_root = tmp_path / "raw_pairs"
    source_root.mkdir()
    pd.DataFrame(
        [
            {"left_code": "return a", "right_code": "return a", "label": 1},
            {"left_code": "return a", "right_code": "return b", "label": 0},
        ]
    ).to_csv(source_root / "pairs.csv", index=False)
    output_root = tmp_path / "normalized_pairs"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "datasets",
            "adapt",
            str(source_root),
            "--kind",
            "pair",
            "--output",
            str(output_root),
            "--dataset-name",
            "custom_pairs",
            "--adapter-option",
            "left_text_column=left_code",
            "--adapter-option",
            "right_text_column=right_code",
            "--adapter-option",
            "label_column=label",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["adapter"] == "auto_pair_tabular"
    assert payload["counts"]["pairs"] == 2
    assert (output_root / "files.csv").exists()
    assert (output_root / "pairs.csv").exists()


def test_datasets_adapt_command_copies_normalized_pair_dataset_to_output(tmp_path):
    dataset_root = tmp_path / "normalized_source"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "b", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
        metadata={"name": "source_pairs"},
    )
    output_root = tmp_path / "normalized_copy"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "datasets",
            "adapt",
            str(dataset_root),
            "--kind",
            "pair",
            "--output",
            str(output_root),
            "--dataset-name",
            "copied_pairs",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "copied_pairs"
    assert payload["counts"]["pairs"] == 1
    assert (output_root / "files.csv").exists()
    assert (output_root / "pairs.csv").exists()


def test_datasets_adapt_command_writes_retrieval_dataset(tmp_path):
    source_root = tmp_path / "raw_retrieval"
    source_root.mkdir()
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
        ]
    ).to_csv(source_root / "retrieval.csv", index=False)
    output_root = tmp_path / "normalized_retrieval"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "datasets",
            "adapt",
            str(source_root),
            "--kind",
            "retrieval",
            "--output",
            str(output_root),
            "--adapter-option",
            "retrieval_table=retrieval.csv",
            "--adapter-option",
            "query_text_column=query_code",
            "--adapter-option",
            "document_text_column=candidate_code",
            "--adapter-option",
            "relevance_column=relevance",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["adapter"] == "auto_retrieval_tabular"
    assert payload["counts"]["queries"] == 1
    assert payload["counts"]["documents"] == 2
    assert (output_root / "qrels.csv").exists()


def test_datasets_adapt_command_copies_normalized_retrieval_dataset_to_output(tmp_path):
    dataset_root = tmp_path / "normalized_retrieval_source"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)", "suffix": ".py"},
                {"file_id": "doc_a", "text": "print(1)", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q1", "file_id": "query_a"}]),
        corpus=pd.DataFrame([{"document_id": "d1", "file_id": "doc_a"}]),
        qrels=pd.DataFrame([{"query_id": "q1", "document_id": "d1", "relevance": 1}]),
        metadata={"name": "source_retrieval"},
    )
    output_root = tmp_path / "normalized_retrieval_copy"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "datasets",
            "adapt",
            str(dataset_root),
            "--kind",
            "retrieval",
            "--output",
            str(output_root),
            "--dataset-name",
            "copied_retrieval",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "copied_retrieval"
    assert payload["counts"]["queries"] == 1
    assert (output_root / "qrels.csv").exists()


def test_evaluate_pairs_command_writes_scores_and_metrics(tmp_path):
    dataset_root = tmp_path / "pairs"
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
    )
    scores_path = tmp_path / "scores.csv"
    metrics_path = tmp_path / "metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            str(dataset_root),
            "--threshold",
            "0.8",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    assert scores_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pair_count"] == 2
    assert "accuracy=" in result.output


def test_evaluate_pairs_command_uses_custom_algorithm(tmp_path):
    dataset_root = tmp_path / "pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "print(1)  # comment", "suffix": ".py"},
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
    algorithm_path = tmp_path / "pair_algo.py"
    algorithm_path.write_text(
        "\n".join(
            [
                "def score_pair(code_a, code_b, bias=0.0):",
                "    return (1.0 if code_a == code_b else 0.0) + float(bias)",
            ]
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "custom_scores.csv"
    metrics_path = tmp_path / "custom_metrics.json"
    reproducibility_path = tmp_path / "custom_repro.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            str(dataset_root),
            "--algorithm-path",
            str(algorithm_path),
            "--algorithm-option",
            "bias=0.1",
            "--preprocess-mode",
            "basic",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
            "--reproducibility-out",
            str(reproducibility_path),
        ],
    )

    assert result.exit_code == 0
    scored = pd.read_csv(scores_path)
    assert scored["similarity_score"].tolist() == [1.1, 0.1]
    payload = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    assert payload["run_metadata"]["algorithm"]["algorithm_options"] == {"bias": 0.1}


def test_evaluate_pairs_command_loads_tabular_adapter_spec(tmp_path):
    raw_root = tmp_path / "raw_pairs"
    raw_root.mkdir()
    pd.DataFrame(
        [
            {"left_code": "print(1)", "right_code": "print(1)", "target": 1},
            {"left_code": "print(1)", "right_code": "print(2)", "target": 0},
        ]
    ).to_csv(raw_root / "pairs.csv", index=False)
    scores_path = tmp_path / "adapter_scores.csv"
    metrics_path = tmp_path / "adapter_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            str(raw_root),
            "--adapter",
            "auto_pair_tabular",
            "--adapter-option",
            "left_text_column=left_code",
            "--adapter-option",
            "right_text_column=right_code",
            "--adapter-option",
            "label_column=target",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pair_count"] == 2
    assert pd.read_csv(scores_path)["label"].tolist() == [1, 0]


def test_evaluate_pairs_command_loads_registered_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_DATASET_PRESETS", dict(datasets_module._DATASET_PRESETS))
    dataset_root = tmp_path / "preset_pairs"
    write_pair_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "a", "text": "return 1", "suffix": ".py"},
                {"file_id": "b", "text": "return 1", "suffix": ".py"},
            ]
        ),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1}]),
    )
    register_dataset_preset(
        "unit_cli_pair_preset",
        {
            "source": "local",
            "identifier": dataset_root,
            "task_families": ("pair",),
        },
        overwrite=True,
    )
    scores_path = tmp_path / "preset_scores.csv"
    metrics_path = tmp_path / "preset_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            "--preset",
            "unit_cli_pair_preset",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    assert pd.read_csv(scores_path)["label"].tolist() == [1]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pair_count"] == 1


def test_evaluate_pairs_command_loads_dataset_manifest(tmp_path):
    raw_root = tmp_path / "raw_pairs"
    raw_root.mkdir()
    pd.DataFrame(
        [
            {"left_code": "x = 1", "right_code": "x = 1", "label": 1},
            {"left_code": "x = 1", "right_code": "x = 2", "label": 0},
        ]
    ).to_csv(raw_root / "pairs.csv", index=False)
    manifest_path = tmp_path / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "pair",
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_pairs",
                        "name": "cli_manifest_pairs",
                        "adapter": "auto_pair_tabular",
                        "adapted_destination": "normalized_pairs",
                        "adapter_options": {
                            "left_text_column": "left_code",
                            "right_text_column": "right_code",
                            "label_column": "label",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "manifest_scores.csv"
    metrics_path = tmp_path / "manifest_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            "--manifest",
            str(manifest_path),
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pair_count"] == 2
    assert (tmp_path / "normalized_pairs" / "pairs.csv").exists()


def test_evaluate_pairs_command_rejects_manifest_with_dataset_options(tmp_path):
    manifest_path = tmp_path / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "task": "pair", "datasets": ["normalized_pairs"]}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-pairs",
            "--manifest",
            str(manifest_path),
            "--adapter",
            "auto_pair_tabular",
        ],
    )

    assert result.exit_code != 0
    assert "--manifest cannot be combined with dataset source options" in result.output


def test_evaluate_retrieval_command_writes_scores_and_metrics(tmp_path):
    dataset_root = tmp_path / "retrieval"
    write_retrieval_dataset(
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
    )
    scores_path = tmp_path / "retrieval_scores.csv"
    metrics_path = tmp_path / "retrieval_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-retrieval",
            str(dataset_root),
            "--k",
            "1",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    assert scores_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["result_count"] == 2
    assert "map=" in result.output


def test_evaluate_retrieval_command_uses_custom_algorithm(tmp_path):
    dataset_root = tmp_path / "retrieval"
    write_retrieval_dataset(
        dataset_root,
        files=pd.DataFrame(
            [
                {"file_id": "query_a", "text": "print(1)  # comment", "suffix": ".py"},
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
    )
    algorithm_path = tmp_path / "retrieval_algo.py"
    algorithm_path.write_text(
        "\n".join(
            [
                "def score_pair(code_a, code_b, bias=0.0, row=None):",
                "    return (1.0 if code_a == code_b else 0.0) + float(bias)",
            ]
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "custom_retrieval_scores.csv"
    metrics_path = tmp_path / "custom_retrieval_metrics.json"
    reproducibility_path = tmp_path / "custom_retrieval_repro.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-retrieval",
            str(dataset_root),
            "--algorithm-path",
            str(algorithm_path),
            "--algorithm-option",
            "bias=0.1",
            "--preprocess-mode",
            "basic",
            "--k",
            "1",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
            "--reproducibility-out",
            str(reproducibility_path),
        ],
    )

    assert result.exit_code == 0
    assert pd.read_csv(scores_path)["similarity_score"].tolist() == [1.1, 0.1]
    payload = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    assert payload["run_metadata"]["algorithm"]["algorithm_options"] == {"bias": 0.1}


def test_evaluate_retrieval_command_loads_tabular_adapter_spec(tmp_path):
    raw_root = tmp_path / "raw_retrieval"
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
        ]
    ).to_csv(raw_root / "retrieval.csv", index=False)
    scores_path = tmp_path / "adapter_retrieval_scores.csv"
    metrics_path = tmp_path / "adapter_retrieval_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-retrieval",
            str(raw_root),
            "--adapter",
            "auto_retrieval_tabular",
            "--adapter-option",
            "retrieval_table=retrieval.csv",
            "--adapter-option",
            "query_text_column=query_code",
            "--adapter-option",
            "document_text_column=candidate_code",
            "--adapter-option",
            "relevance_column=relevance",
            "--k",
            "1",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["result_count"] == 2
    assert set(pd.read_csv(scores_path)["document_id"]) == {"d1", "d2"}


def test_evaluate_retrieval_command_loads_dataset_manifest(tmp_path):
    raw_root = tmp_path / "raw_retrieval"
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
        ]
    ).to_csv(raw_root / "retrieval.csv", index=False)
    manifest_path = tmp_path / "retrieval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task": "retrieval",
                "datasets": [
                    {
                        "source": "local",
                        "identifier": "raw_retrieval",
                        "adapter": "auto_retrieval_tabular",
                        "adapted_destination": "normalized_retrieval",
                        "adapter_options": {
                            "retrieval_table": "retrieval.csv",
                            "query_text_column": "query_code",
                            "document_text_column": "candidate_code",
                            "relevance_column": "relevance",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "manifest_retrieval_scores.csv"
    metrics_path = tmp_path / "manifest_retrieval_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-retrieval",
            "--manifest",
            str(manifest_path),
            "--k",
            "1",
            "--scores-out",
            str(scores_path),
            "--metrics-out",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["result_count"] == 2
    assert (tmp_path / "normalized_retrieval" / "qrels.csv").exists()


def test_dataset_example_script_runs():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [sys.executable, "examples/evaluation/datasets_demo.py"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Generic tabular retrieval adapter" in result.stdout
    assert "Custom source and preset" in result.stdout


def test_sample_data_generator_writes_archive(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "sample_pairs.zip"

    result = subprocess.run(
        [
            sys.executable,
            "examples/sample_data.py",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote sample zip" in result.stdout
    with ZipFile(output_path) as archive:
        assert sorted(archive.namelist()) == [
            "code_1.java",
            "code_2_plag.java",
            "code_3_plag.java",
            "code_4_nonplag.java",
            "hello_world.java",
        ]


def test_reproducible_benchmark_example_script_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
    )
    output_dir = tmp_path / "benchmark"

    result = subprocess.run(
        [
            sys.executable,
            "examples/evaluation/reproducible_benchmark_demo.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reproducible synthetic pair benchmark" in result.stdout
    assert (output_dir / "dataset_manifest.json").exists()
    assert (output_dir / "benchmark_config.json").exists()
    assert (output_dir / "results" / "scored_pairs.csv").exists()
    assert (output_dir / "results" / "resample_summary.csv").exists()

    metrics = json.loads((output_dir / "results" / "pair_metrics.json").read_text(encoding="utf-8"))
    assert metrics["pair_count"] == 8
    reproducibility = json.loads(
        (output_dir / "results" / "reproducibility.json").read_text(encoding="utf-8")
    )
    assert reproducibility["schema_version"] == 1
    assert reproducibility["source"]["source_type"] == "directory"


def test_compare_suite_command_runs_config_file(tmp_path, monkeypatch):
    archive_path = tmp_path / "codes.zip"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "runs.json"
    config_path.write_text("[]", encoding="utf-8")

    captured = {}

    def fake_load_run_configs(config_file):
        captured["config_file"] = config_file
        return [{"run_name": "baseline", "options": {}}]

    def fake_run_comparison_suite(
        zipfile,
        run_configs,
        summary_out,
        details_dir,
        output_format,
        reproducibility_out,
        cache_dir,
        use_cache,
        cache_seed,
        progress,
    ):
        captured["zipfile"] = zipfile
        captured["run_configs"] = run_configs
        captured["summary_out"] = summary_out
        captured["details_dir"] = details_dir
        captured["output_format"] = output_format
        captured["reproducibility_out"] = reproducibility_out
        captured["cache_dir"] = cache_dir
        captured["use_cache"] = use_cache
        captured["cache_seed"] = cache_seed
        captured["progress"] = progress
        return (
            pd.DataFrame(
                [
                    {
                        "run_name": "baseline",
                        "pair_count": 1,
                        "mean_score": 0.9,
                        "median_score": 0.9,
                        "max_score": 0.9,
                        "min_score": 0.9,
                        "std_score": 0.0,
                        "top_file_1": "a.py",
                        "top_file_2": "b.py",
                        "top_score": 0.9,
                        "vector_backend": "transformer",
                        "code_metric": "none",
                        "chunking_method": "none",
                    }
                ]
            ),
            {"baseline": pd.DataFrame()},
        )

    monkeypatch.setattr("matheel.cli.load_run_configs", fake_load_run_configs)
    monkeypatch.setattr("matheel.cli.run_comparison_suite", fake_run_comparison_suite)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare-suite",
            str(archive_path),
            str(config_path),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--details-dir",
            str(tmp_path / "details"),
            "--format",
            "json",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--cache-seed",
            "demo",
            "--progress",
        ],
    )

    assert result.exit_code == 0
    assert captured["config_file"] == str(config_path)
    assert captured["zipfile"] == str(archive_path)
    assert captured["summary_out"].endswith("summary.json")
    assert captured["details_dir"].endswith("details")
    assert captured["output_format"] == "json"
    assert captured["reproducibility_out"] is None
    assert captured["cache_dir"].endswith("cache")
    assert captured["use_cache"] is True
    assert captured["cache_seed"] == "demo"
    assert captured["progress"] is True
