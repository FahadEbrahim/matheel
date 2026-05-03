"""Run a tiny reproducible Matheel pair-classification benchmark."""

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

from matheel.datasets import load_pair_datasets_from_manifest  # noqa: E402
from matheel.evaluation import evaluate_pair_dataset, evaluate_pair_resamples  # noqa: E402
from matheel.reproducibility import (  # noqa: E402
    collect_reproducibility_snapshot,
    write_reproducibility_snapshot,
)
from matheel.resampling import kfold_splits  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("benchmark_outputs") / "synthetic_pair_benchmark"
THRESHOLD = 0.65
RESAMPLING_SEED = 7
RESAMPLING_FOLDS = 4
CONFIDENCE = 0.95
SIMILARITY_OPTIONS = {
    "feature_weights": {"levenshtein": 1.0},
    "preprocess_mode": "basic",
    "code_language": "python",
    "vector_backend": "auto",
}


SYNTHETIC_PAIRS = [
    {
        "left_id": "sum_original",
        "right_id": "sum_renamed",
        "left_code": "def total(values):\n    return sum(values)\n",
        "right_code": "def total(items):\n    return sum(items)\n",
        "label": 1,
    },
    {
        "left_id": "max_original",
        "right_id": "max_renamed",
        "left_code": "def largest(values):\n    return max(values)\n",
        "right_code": "def largest(numbers):\n    return max(numbers)\n",
        "label": 1,
    },
    {
        "left_id": "tax_original",
        "right_id": "tax_reformatted",
        "left_code": "def total_with_tax(price, tax):\n    return price + tax\n",
        "right_code": "def total_with_tax(price,tax):\n    result = price + tax\n    return result\n",
        "label": 1,
    },
    {
        "left_id": "filter_original",
        "right_id": "filter_loop",
        "left_code": "def positives(values):\n    return [value for value in values if value > 0]\n",
        "right_code": (
            "def positives(values):\n"
            "    output = []\n"
            "    for value in values:\n"
            "        if value > 0:\n"
            "            output.append(value)\n"
            "    return output\n"
        ),
        "label": 1,
    },
    {
        "left_id": "sum_original",
        "right_id": "multiply_different",
        "left_code": "def total(values):\n    return sum(values)\n",
        "right_code": "def product(values):\n    result = 1\n    for value in values:\n        result *= value\n    return result\n",
        "label": 0,
    },
    {
        "left_id": "max_original",
        "right_id": "duplicates_different",
        "left_code": "def largest(values):\n    return max(values)\n",
        "right_code": (
            "def has_duplicates(values):\n"
            "    seen = set()\n"
            "    for value in values:\n"
            "        if value in seen:\n"
            "            return True\n"
            "        seen.add(value)\n"
            "    return False\n"
        ),
        "label": 0,
    },
    {
        "left_id": "tax_original",
        "right_id": "discount_different",
        "left_code": "def total_with_tax(price, tax):\n    return price + tax\n",
        "right_code": "def total_after_discount(price, discount):\n    return price - discount\n",
        "label": 0,
    },
    {
        "left_id": "filter_original",
        "right_id": "sort_different",
        "left_code": "def positives(values):\n    return [value for value in values if value > 0]\n",
        "right_code": "def ordered(values):\n    return sorted(values)\n",
        "label": 0,
    },
]


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        _replace_output_dir(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} already exists and is not empty. Pass --overwrite to replace it.")

    artifacts = run_benchmark(output_dir)
    print("Reproducible synthetic pair benchmark")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


def run_benchmark(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    results_dir = output_dir / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs_path = raw_dir / "pairs.csv"
    pd.DataFrame(SYNTHETIC_PAIRS).to_csv(raw_pairs_path, index=False)

    manifest_path = output_dir / "dataset_manifest.json"
    manifest = _dataset_manifest()
    _write_json(manifest_path, manifest)

    config_path = output_dir / "benchmark_config.json"
    config = _benchmark_config(manifest_path.name)
    _write_json(config_path, config)

    dataset = load_pair_datasets_from_manifest(manifest_path)
    scored_pairs, metrics = evaluate_pair_dataset(
        dataset,
        threshold=THRESHOLD,
        similarity_options=SIMILARITY_OPTIONS,
    )

    scores_path = results_dir / "scored_pairs.csv"
    metrics_path = results_dir / "pair_metrics.json"
    scored_pairs.to_csv(scores_path, index=False)
    _write_json(metrics_path, metrics)

    splits = kfold_splits(
        len(scored_pairs),
        n_splits=RESAMPLING_FOLDS,
        shuffle=True,
        seed=RESAMPLING_SEED,
        labels=scored_pairs["label"].tolist(),
    )
    fold_metrics, fold_summary = evaluate_pair_resamples(
        scored_pairs,
        splits,
        threshold=THRESHOLD,
        confidence=CONFIDENCE,
    )
    resample_metrics_path = results_dir / "resample_metrics.csv"
    resample_summary_path = results_dir / "resample_summary.csv"
    fold_metrics.to_csv(resample_metrics_path, index=False)
    fold_summary.to_csv(resample_summary_path, index=False)

    reproducibility_path = results_dir / "reproducibility.json"
    snapshot = collect_reproducibility_snapshot(
        source_path=dataset.root,
        run_configs=[config],
        result_attrs={
            "metrics": metrics,
            "resampling": {
                "method": "kfold",
                "n_splits": RESAMPLING_FOLDS,
                "seed": RESAMPLING_SEED,
                "confidence": CONFIDENCE,
            },
        },
    )
    write_reproducibility_snapshot(snapshot, reproducibility_path)

    return {
        "raw_pairs": raw_pairs_path,
        "dataset_manifest": manifest_path,
        "benchmark_config": config_path,
        "scores": scores_path,
        "metrics": metrics_path,
        "resample_metrics": resample_metrics_path,
        "resample_summary": resample_summary_path,
        "reproducibility": reproducibility_path,
    }


def _replace_output_dir(output_dir):
    resolved = Path(output_dir).expanduser().resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve()}
    if resolved.parent == resolved or resolved in protected:
        raise SystemExit(f"Refusing to overwrite protected output directory: {output_dir}")
    shutil.rmtree(resolved)


def _dataset_manifest():
    return {
        "version": 1,
        "task": "pair",
        "datasets": [
            {
                "name": "synthetic_pair_benchmark",
                "source": "local",
                "identifier": "./raw",
                "adapter": "auto_pair_tabular",
                "adapted_destination": "./dataset",
                "adapter_options": {
                    "pair_table": "pairs.csv",
                    "left_id_column": "left_id",
                    "right_id_column": "right_id",
                    "left_text_column": "left_code",
                    "right_text_column": "right_code",
                    "label_column": "label",
                    "suffix": ".py",
                },
            }
        ],
    }


def _benchmark_config(manifest_name):
    return {
        "dataset_manifest": manifest_name,
        "scoring": {
            "threshold": THRESHOLD,
            "similarity_options": SIMILARITY_OPTIONS,
        },
        "resampling": {
            "method": "kfold",
            "n_splits": RESAMPLING_FOLDS,
            "seed": RESAMPLING_SEED,
            "confidence": CONFIDENCE,
        },
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where synthetic inputs and benchmark outputs are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
