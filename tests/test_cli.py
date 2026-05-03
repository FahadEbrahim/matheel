import json

from click.testing import CliRunner
import pandas as pd

from matheel.cli import main
from matheel.datasets import write_pair_dataset, write_retrieval_dataset


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
    assert captured["kwargs"]["static_vector_dim"] == 512
    assert captured["kwargs"]["max_token_length"] == 128
    assert captured["kwargs"]["pooling_method"] == "max"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["progress"] is True
    assert "Elapsed: 1.2346s" in result.stderr
    assert "features=code_metric,semantic" in result.stderr


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
        progress,
    ):
        captured["zipfile"] = zipfile
        captured["run_configs"] = run_configs
        captured["summary_out"] = summary_out
        captured["details_dir"] = details_dir
        captured["output_format"] = output_format
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
            "--progress",
        ],
    )

    assert result.exit_code == 0
    assert captured["config_file"] == str(config_path)
    assert captured["zipfile"] == str(archive_path)
    assert captured["summary_out"].endswith("summary.json")
    assert captured["details_dir"].endswith("details")
    assert captured["output_format"] == "json"
    assert captured["progress"] is True
