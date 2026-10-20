#!/usr/bin/env python3
"""Generate an uncertainty companion from audited, existing score CSVs."""

import argparse
import json
from pathlib import Path

from matheel.benchmark_analysis import build_benchmark_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = build_benchmark_analysis(args.result, args.scores, rounds=args.rounds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['metrics'])} metric diagnostics to {args.output}")


if __name__ == "__main__":
    main()
