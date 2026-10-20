"""Create offline paired-error and uncertainty reports from synthetic scores."""

import argparse
from pathlib import Path

import pandas as pd

from matheel.result_analysis import analyze_scored_results, write_result_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs/result_analysis"))
    args = parser.parse_args()
    # Diagnostic scores exercise error inspection; they are not research results.
    scores = pd.DataFrame([
        {"left_id": f"a{i}", "right_id": f"b{i}", "label": i % 2,
         "similarity_score": score, "assignment": f"fixture_{i // 2}"}
        for i, score in enumerate([.1, .9, .7, .3, .2, .8, .6, .4])
    ])
    reference = scores.copy()
    reference["similarity_score"] = [.2, .8, .6, .7, .3, .9, .4, .5]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    reference.to_csv(args.output_dir / "reference.csv", index=False)
    report = analyze_scored_results(scores, reference_rows=reference, group_column="assignment", rounds=200)
    artifacts = write_result_analysis(report, args.output_dir)
    print(f"Inspected {report['rows']} pairs in {report['group_count']} assignment groups.")
    print(f"HTML report: {artifacts['html']}")


if __name__ == "__main__":
    main()
