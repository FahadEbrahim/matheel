#!/usr/bin/env python3
"""Run the pinned local JPlag and Dolos baselines on SimilCode v2.4.0."""

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from matheel import __version__
from matheel.datasets import adapt_pair_dataset, load_pair_dataset
from matheel.evaluation import pair_classification_metrics, score_pair_dataset
from matheel.similarity_benchmark import _gzip_csv, _hash, _normalized_content_sha256
from matheel.similarity_datasets import similcode_score_slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted original SimilCode release directory.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True, choices=("java", "csharp"))
    parser.add_argument("--jplag-jar", type=Path, required=True)
    parser.add_argument("--java-path", type=Path, required=True)
    parser.add_argument("--dolos-prefix", type=Path)
    parser.add_argument("--node-path", type=Path)
    args = parser.parse_args()
    if bool(args.dolos_prefix) != bool(args.node_path):
        parser.error("Specify both --dolos-prefix and --node-path to enable Dolos.")
    if args.language != "java" and args.dolos_prefix:
        parser.error("This release's Dolos adapter supports the SimilCode Java view.")
    args.output.mkdir(parents=True, exist_ok=True)
    options = {"language": args.language}
    methods = [("jplag", {**options, "jar_path": str(args.jplag_jar), "java_path": str(args.java_path)})]
    if args.dolos_prefix:
        methods.append(("dolos", {**options, "dolos_path": str(args.dolos_prefix), "node_path": str(args.node_path)}))

    with TemporaryDirectory(prefix="similcode-detectors-") as work, TemporaryDirectory(dir=args.output) as staging:
        prepared = adapt_pair_dataset(args.source, adapter="similcode_pair", destination=Path(work) / "dataset",
                                      adapter_options=options)
        dataset = load_pair_dataset(prepared)
        results, scored_frames, sliced_frames = [], [], []
        for method, configuration in methods:
            scored = score_pair_dataset(dataset, algorithm=f"matheel.external_detectors.{method}",
                                        algorithm_options=configuration)
            results.append({"name": method, "configuration": scored.attrs["algorithm_options"],
                            "metrics": pair_classification_metrics(scored, threshold=0.5)})
            raw = scored.assign(method=method)
            raw.attrs = {}
            scored_frames.append(raw)
            sliced_frames.append(similcode_score_slices(raw).assign(method=method))
        score_path = Path(staging) / "scores.csv.gz"
        slices = Path(staging) / "slices.csv"
        with _gzip_csv(score_path) as handle:
            pd.concat(scored_frames, ignore_index=True).to_csv(handle, index=False)
        pd.concat(sliced_frames, ignore_index=True).to_csv(slices, index=False)
        report = {
            "benchmark_schema_version": 1, "task_family": "pair", "dataset": dataset.metadata,
            "dataset_sha256": _normalized_content_sha256(dataset, "pair"),
            "normalized_fingerprint_schema": "manifest-bytes-and-files-v1",
            "matheel_version": __version__, "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
            "implementation_sha256": {Path(__file__).name: _hash(Path(__file__))},
            "results": results,
            "artifacts": {path.name: {"sha256": _hash(path), "bytes": path.stat().st_size}
                          for path in (score_path, slices)},
        }
        report_path = Path(staging) / "results.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for path in (score_path, slices, report_path):
            path.replace(args.output / path.name)


if __name__ == "__main__":
    main()
