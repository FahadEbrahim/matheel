"""Run a tiny local leaderboard over pair and retrieval datasets."""

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import gettempdir

import pandas as pd

# Configure Matplotlib before importing Matheel's evaluation stack.
_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

from matheel.datasets import write_pair_dataset, write_retrieval_dataset  # noqa: E402
from matheel.leaderboard import available_leaderboard_metrics, load_leaderboard_manifest, run_leaderboard  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("benchmark_outputs") / "synthetic_leaderboard"


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        _replace_output_dir(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} already exists and is not empty. Pass --overwrite to replace it.")

    report, artifacts = run_demo(output_dir)
    print("Available metrics:", available_leaderboard_metrics())
    print()
    print("Aggregate ranking")
    print(report["aggregate"].round(4).to_string(index=False))
    print()
    print("Per-dataset ranking")
    print(report["per_dataset"].round(4).to_string(index=False))
    print()
    for name, path in artifacts.items():
        print(f"{name}: {path}")


def run_demo(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    algorithm_dir = output_dir / "algorithms"
    artifact_dir = output_dir / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    _write_pair_dataset(data_dir / "pairs")
    _write_retrieval_dataset(data_dir / "retrieval")
    _write_exact_algorithm(algorithm_dir / "exact_string.py")

    manifest_path = output_dir / "leaderboard.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "synthetic_leaderboard",
                "seed": 7,
                "pair_metrics": ["f1", "accuracy", "auroc"],
                "retrieval_metrics": ["mean_average_precision", "ndcg_at_k"],
                "datasets": [
                    {
                        "name": "tiny_pairs",
                        "task": "pair",
                        "path": "./data/pairs",
                        "threshold": 0.5,
                    },
                    {
                        "name": "tiny_retrieval",
                        "task": "retrieval",
                        "path": "./data/retrieval",
                        "k": 2,
                    },
                ],
                "algorithms": [
                    {
                        "name": "levenshtein",
                        "feature_weights": {"levenshtein": 1.0},
                    },
                    {
                        "name": "exact_string",
                        "algorithm_path": "./algorithms/exact_string.py",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = load_leaderboard_manifest(manifest_path)
    return run_leaderboard(manifest, output_dir=artifact_dir, basename="synthetic")


def _write_pair_dataset(target):
    write_pair_dataset(
        target,
        files=pd.DataFrame(
            [
                {"file_id": "sum_original", "text": "def total(values):\n    return sum(values)\n", "suffix": ".py"},
                {"file_id": "sum_same", "text": "def total(items):\n    return sum(items)\n", "suffix": ".py"},
                {
                    "file_id": "product_different",
                    "text": "def product(values):\n    result = 1\n    for value in values:\n        result *= value\n    return result\n",
                    "suffix": ".py",
                },
            ]
        ),
        pairs=pd.DataFrame(
            [
                {"left_id": "sum_original", "right_id": "sum_same", "label": 1},
                {"left_id": "sum_original", "right_id": "product_different", "label": 0},
            ]
        ),
        metadata={"name": "tiny_pairs", "license": "synthetic"},
    )
    return target


def _write_retrieval_dataset(target):
    write_retrieval_dataset(
        target,
        files=pd.DataFrame(
            [
                {"file_id": "query_sum", "text": "def total(values):\n    return sum(values)\n", "suffix": ".py"},
                {"file_id": "doc_sum", "text": "def total(items):\n    return sum(items)\n", "suffix": ".py"},
                {"file_id": "doc_sort", "text": "def ordered(values):\n    return sorted(values)\n", "suffix": ".py"},
            ]
        ),
        queries=pd.DataFrame([{"query_id": "q_sum", "file_id": "query_sum"}]),
        corpus=pd.DataFrame(
            [
                {"document_id": "d_sum", "file_id": "doc_sum"},
                {"document_id": "d_sort", "file_id": "doc_sort"},
            ]
        ),
        qrels=pd.DataFrame([{"query_id": "q_sum", "document_id": "d_sum", "relevance": 1}]),
        metadata={"name": "tiny_retrieval", "license": "synthetic"},
    )
    return target


def _write_exact_algorithm(target):
    target.write_text(
        "def score_pair(code_a, code_b, row=None):\n"
        "    _ = row\n"
        "    return 1.0 if code_a.strip() == code_b.strip() else 0.0\n",
        encoding="utf-8",
    )
    return target


def _replace_output_dir(output_dir):
    resolved = Path(output_dir).expanduser().resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve()}
    if resolved.parent == resolved or resolved in protected:
        raise SystemExit(f"Refusing to overwrite protected output directory: {output_dir}")
    shutil.rmtree(resolved)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.fspath(DEFAULT_OUTPUT_DIR),
        help=f"Directory for generated datasets and leaderboard artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
