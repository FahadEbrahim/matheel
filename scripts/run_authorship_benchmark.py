#!/usr/bin/env python3
"""Run or verify the pinned Matheel-AIGCodeSet-v1 benchmark."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from matheel.authorship import (
    AIGCODESET_REVISION,
    AIGCODESET_SOURCES,
    benchmark_authorship,
    prepare_aigcodeset,
    validate_authorship_report,
    write_authorship_report,
)
from matheel.datasets import _download_url_to_file


def _source_paths(source_dir, download):
    paths = {}
    for kind, source in AIGCODESET_SOURCES.items():
        path = source_dir / source["filename"]
        if not path.is_file() and download:
            source_dir.mkdir(parents=True, exist_ok=True)
            url = (
                "https://huggingface.co/datasets/basakdemirok/AIGCodeSet/resolve/"
                f"{AIGCODESET_REVISION}/data/{source['filename']}"
            )
            with TemporaryDirectory(dir=source_dir, prefix=".download-") as temporary:
                pending = Path(temporary) / source["filename"]
                _download_url_to_file(url, pending, max_bytes=source["size"])
                if pending.stat().st_size != source["size"] or hashlib.sha256(
                    pending.read_bytes()
                ).hexdigest() != source["sha256"]:
                    raise ValueError(f"Downloaded AIGCodeSet source differs from {source['filename']}.")
                pending.replace(path)
        paths[kind] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run", help="Train and evaluate both baselines.")
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--download", action="store_true", help="Download missing pinned tables.")
    run.add_argument("--compare-to", type=Path, help="Compare regenerated scores to a verified results.json.")
    validate = modes.add_parser("validate", help="Verify and recompute metrics from saved predictions.")
    validate.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        result = validate_authorship_report(args.results)
        print(f"Verified {result['prediction_artifact']['rows']} predictions for {result['benchmark_id']}.")
        return
    source = _source_paths(args.source_dir, args.download)
    dataset = prepare_aigcodeset(source["human"], source["generated"], verify_source=True)
    report = benchmark_authorship(dataset)
    paths = write_authorship_report(report, args.output_dir)
    generated = validate_authorship_report(paths["results"])
    if args.compare_to is not None:
        expected = validate_authorship_report(args.compare_to)
        for key in ("dataset_fingerprint", "threshold_rule", "prediction_artifact"):
            if generated[key] != expected[key]:
                raise ValueError(f"Regenerated authorship benchmark differs on {key}.")
        print("Regenerated predictions and dataset fingerprint match the reference.")
    print(paths["results"])


if __name__ == "__main__":
    main()
