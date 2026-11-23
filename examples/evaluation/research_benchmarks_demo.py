"""Generate offline examples of typed clone recall and code authorship evaluation."""

import argparse
import csv
import json
from pathlib import Path

from matheel.clone_discovery import (
    CloneFinding,
    CloneReference,
    SourceSpan,
    evaluate_clone_discovery,
    write_clone_findings,
    write_clone_references,
)


def _clone_example(output):
    original = SourceSpan("src/Original.java", 10, 19)
    moved = SourceSpan("src/Second.java", 30, 39)
    references = [
        CloneReference(original, moved, "type1", project_scope="intra_project"),
        CloneReference(original, SourceSpan("other/Third.java", 1, 10), "weak_type3_type4",
                       project_scope="inter_project"),
    ]
    findings = [CloneFinding(SourceSpan("src/Original.java", 11, 19), moved)]
    write_clone_references(output / "references.csv", references)
    write_clone_findings(output / "findings.csv", findings)
    report = evaluate_clone_discovery(references, findings)
    (output / "clone_recall.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Clone recall {report['overall']['recall']:.0%} on two synthetic reference pairs")


def _authorship_example(output):
    from matheel.authorship import benchmark_authorship, prepare_aigcodeset, write_authorship_report

    fields = ("problem_id", "submission_id", "LLM", "status_in_folder", "code", "label")
    for label, source, status, code in (
        (0, "Human", "Accepted", "value_{i} = {i}"),
        (1, "ExampleModel", "Generate", "print('example {i}')"),
    ):
        with (output / f"origin_{label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for i in range(10):
                writer.writerow((f"p{i}", f"s{label}_{i}", source, status, code.format(i=i), label))
    dataset = prepare_aigcodeset(output / "origin_0.csv", output / "origin_1.csv")
    report = benchmark_authorship(dataset, models=("class_prior",))
    write_authorship_report(report, output / "authorship")
    print(f"Prepared {len(dataset.samples)} synthetic authorship examples")


def run_research_benchmarks(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _clone_example(output)
    _authorship_example(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs/research_demo"))
    args = parser.parse_args()
    output = run_research_benchmarks(args.output_dir)
    print(f"Reports and example inputs are in {output}")


if __name__ == "__main__":
    main()
