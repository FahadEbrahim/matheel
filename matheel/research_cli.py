"""Command-line entry points for benchmark tasks beyond pair scoring."""

import csv
import json
from pathlib import Path

import click


_INPUT_FILE = click.Path(path_type=Path, dir_okay=False, exists=True)
_OUTPUT_DIRECTORY = click.Path(path_type=Path, file_okay=False)
_OUTPUT_FILE = click.Path(path_type=Path, dir_okay=False)
_INPUT_DIRECTORY = click.Path(path_type=Path, file_okay=False, exists=True)


def _read_map(path):
    if path is None:
        return None
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("A JSON source map must be an object keyed by source ID.")
    return mapping


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_output_paths(outputs, inputs):
    destinations = [path.resolve() for path in outputs if path is not None]
    input_paths = {path.resolve() for path in inputs if path is not None}
    if len(set(destinations)) != len(destinations) or input_paths.intersection(destinations):
        raise ValueError("Use distinct paths for inputs and output artifacts.")


def register_research_commands(main):
    @main.command(name="benchmark-similarity")
    @click.argument("adapter", type=click.Choice([
        "similcode_pair", "poj104_retrieval", "xlcost_retrieval", "codexglue_bcb_pair",
    ]))
    @click.argument("source", type=click.Path(path_type=Path, exists=True, file_okay=False))
    @click.argument("output_dir", type=_OUTPUT_DIRECTORY)
    @click.option("--split", type=click.Choice(["train", "validation", "test"]), default="test")
    @click.option("--source-revision", help="Pinned upstream revision to record in the run.")
    @click.option("--language", type=click.Choice(["java", "csharp"]))
    @click.option("--source-language", type=click.Choice(["java", "python"]), default="java")
    @click.option("--target-language", type=click.Choice(["java", "python"]), default="python")
    @click.option("--label-view", default="related", show_default=True)
    @click.option("--threshold", type=click.FloatRange(0, 1), default=0.5, show_default=True)
    @click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
    def benchmark_similarity_command(adapter, source, output_dir, **options):
        """Reproduce a code similarity benchmark on locally acquired data."""
        from .similarity_benchmark import run_similarity_benchmark

        try:
            run_similarity_benchmark(adapter, source, output_dir, **options)
        except (ImportError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote similarity evaluation artifacts to {output_dir}.")

    @main.group(name="authorship")
    def authorship():
        """Prepare and benchmark individual code samples with origin labels."""

    @authorship.command(name="prepare-aigcodeset")
    @click.argument("human_csv", type=_INPUT_FILE)
    @click.argument("generated_csv", type=_INPUT_FILE)
    @click.argument("output", type=click.Path(path_type=Path))
    @click.option("--seed", type=int, default=23, show_default=True)
    @click.option("--verify-source", is_flag=True, help="Require the published source CSV hashes.")
    def prepare_aigcodeset_command(human_csv, generated_csv, output, seed, verify_source):
        """Prepare a private AIGCodeSet snapshot from the two source tables."""
        from .authorship import prepare_aigcodeset, write_authorship_dataset

        try:
            _check_output_paths((output,), (human_csv, generated_csv))
            dataset = prepare_aigcodeset(human_csv, generated_csv, seed=seed, verify_source=verify_source)
            write_authorship_dataset(dataset, output)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Prepared {len(dataset.samples)} samples in {output}.")

    @authorship.command(name="benchmark")
    @click.argument("dataset", type=_INPUT_FILE)
    @click.argument("output_dir", type=_OUTPUT_DIRECTORY)
    @click.option("--target-fpr", type=click.FloatRange(0, 1), default=0.05, show_default=True)
    @click.option(
        "--model", "models", multiple=True,
        type=click.Choice(["class_prior", "char_tfidf_logistic"]),
        help="Baseline to evaluate. Repeat to select multiple baselines.",
    )
    def benchmark_authorship_command(dataset, output_dir, target_fpr, models):
        """Run authorship baselines and write source-free metrics and predictions."""
        from .authorship import benchmark_authorship, load_authorship_dataset, write_authorship_report

        try:
            options = {"models": models} if models else {}
            report = benchmark_authorship(
                load_authorship_dataset(dataset), target_fpr=target_fpr, **options,
            )
            write_authorship_report(report, output_dir)
        except (ImportError, OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote authorship evaluation artifacts to {output_dir}.")

    @authorship.command(name="verify")
    @click.argument("report", type=_INPUT_FILE)
    def verify_authorship_command(report):
        """Verify an exported report and replay its metrics offline."""
        from .authorship import validate_authorship_report

        try:
            validate_authorship_report(report)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo("Authorship artifact and replayed metrics verified.")

    @main.command(name="evaluate-clones")
    @click.argument("references", type=_INPUT_FILE)
    @click.argument("findings", type=_INPUT_FILE)
    @click.option("--output", "output_path", type=_OUTPUT_FILE, required=True)
    @click.option("--coverage", type=click.FloatRange(0, 1, min_open=True), default=0.70, show_default=True)
    @click.option("--minimum-lines", type=click.IntRange(min=1), default=1, show_default=True)
    @click.option("--minimum-pretty-lines", type=click.IntRange(min=1))
    @click.option("--minimum-tokens", type=click.IntRange(min=1))
    @click.option("--similarity-type", type=click.Choice(["both", "line", "token"]), default="both")
    @click.option("--minimum-syntax-similarity", type=click.FloatRange(0, 1))
    def evaluate_clones_command(references, findings, output_path, coverage, **settings):
        """Evaluate reference clone recall by source span and clone type."""
        from .clone_discovery import evaluate_clone_discovery, load_clone_findings, load_clone_references
        from .external_detectors.common import file_sha256

        try:
            _check_output_paths((output_path,), (references, findings))
            report = evaluate_clone_discovery(
                load_clone_references(references), load_clone_findings(findings),
                coverage_threshold=coverage, **settings,
            )
            report["input_sha256"] = {
                "references": file_sha256(references), "findings": file_sha256(findings),
            }
            _write_json(output_path, report)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        result = report["overall"]
        click.echo(f"Matched {result['matched']}/{result['eligible']} eligible references.")

    @main.command(name="import-clone-report")
    @click.argument("report_file", type=_INPUT_FILE)
    @click.argument("output", type=_OUTPUT_FILE)
    @click.option("--format", "report_format", type=click.Choice(["nicad", "pmd-cpd", "sourcerercc"]), required=True)
    @click.option("--source-map", type=_INPUT_FILE, help="JSON path-to-ID map or SourcererCC mapping CSV.")
    @click.option("--metadata-out", type=_OUTPUT_FILE)
    def import_clone_report_command(report_file, output, report_format, source_map, metadata_out):
        """Convert a recorded detector report to Matheel's span-pair CSV."""
        from .clone_discovery import write_clone_findings
        from .external_detectors import nicad, pmd_cpd, sourcerercc

        try:
            _check_output_paths((output, metadata_out), (report_file, source_map))
            if report_format == "sourcerercc":
                if source_map is None:
                    raise ValueError("SourcererCC reports require a mapping CSV via --source-map.")
                report = sourcerercc.parse_report(report_file, source_map)
            else:
                parser = nicad if report_format == "nicad" else pmd_cpd
                report = parser.parse_report(report_file, source_map=_read_map(source_map))
            output.parent.mkdir(parents=True, exist_ok=True)
            write_clone_findings(output, report.findings)
            if metadata_out is not None:
                _write_json(metadata_out, report.metadata)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote {len(report.findings)} clone findings to {output}.")

    @main.command(name="run-clone-detector")
    @click.argument("detector", type=click.Choice(["nicad", "pmd-cpd"]))
    @click.argument("corpus", type=_INPUT_DIRECTORY)
    @click.argument("executable", type=_INPUT_FILE)
    @click.argument("output", type=_OUTPUT_FILE)
    @click.option("--options", "options_path", type=_INPUT_FILE, help="JSON object of native runner options.")
    @click.option("--metadata-out", type=_OUTPUT_FILE)
    def run_clone_detector_command(detector, corpus, executable, output, options_path, metadata_out):
        """Run a pinned local detector once on a staged Java corpus."""
        from .clone_discovery import write_clone_findings
        from .external_detectors import nicad, pmd_cpd

        runner = nicad if detector == "nicad" else pmd_cpd
        metadata_out = metadata_out or output.with_name(f"{output.name}.json")
        try:
            _check_output_paths((output, metadata_out), (executable, options_path))
            options = _read_map(options_path) or {}
            report = runner.run_detector(corpus, executable, **options)
            output.parent.mkdir(parents=True, exist_ok=True)
            write_clone_findings(output, report.findings)
            _write_json(metadata_out, report.metadata)
        except (OSError, TypeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote {len(report.findings)} clone findings to {output}.")

    @main.command(name="export-bigcloneeval")
    @click.argument("findings", type=_INPUT_FILE)
    @click.argument("output", type=_OUTPUT_FILE)
    @click.option("--source-map", type=_INPUT_FILE, help="JSON map from stable IDs to BigCloneEval paths.")
    def export_bigcloneeval_command(findings, output, source_map):
        """Export span findings for the BigCloneEval reference evaluator."""
        from .clone_discovery import load_clone_findings, write_bigcloneeval_findings

        try:
            _check_output_paths((output,), (findings, source_map))
            output.parent.mkdir(parents=True, exist_ok=True)
            write_bigcloneeval_findings(output, load_clone_findings(findings), source_map=_read_map(source_map))
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote BigCloneEval findings to {output}.")

    @main.command(name="extract-bigcloneeval-references")
    @click.argument("database", type=_INPUT_FILE)
    @click.argument("output", type=_OUTPUT_FILE)
    @click.option("--java", "java_path", type=_INPUT_FILE, required=True)
    @click.option("--h2-jar", type=_INPUT_FILE, required=True)
    @click.option("--max-references", type=click.IntRange(min=1), help="Limit extraction to a bounded reference scope.")
    @click.option("--timeout-seconds", type=click.IntRange(min=1), default=600, show_default=True)
    def extract_bigcloneeval_references_command(database, output, java_path, h2_jar, **options):
        """Export typed references from a local BigCloneEval v1.1 H2 database."""
        from .external_detectors.bigcloneeval import extract_references

        try:
            _check_output_paths((output,), (database, java_path, h2_jar))
            output.parent.mkdir(parents=True, exist_ok=True)
            extract_references(database, output, java_path=java_path, h2_jar=h2_jar, **options)
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote BigCloneEval references to {output}.")

    @main.command(name="import-moss-report")
    @click.argument("report_file", type=_INPUT_FILE)
    @click.argument("output", type=_OUTPUT_FILE)
    @click.option("--source-map", type=_INPUT_FILE, help="JSON map from report paths to submission IDs.")
    @click.option("--match-directory", type=_INPUT_DIRECTORY)
    @click.option("--expected-pairs", type=_INPUT_FILE, help="Optional CSV with left_id and right_id columns.")
    @click.option("--metadata-out", type=_OUTPUT_FILE)
    def import_moss_report_command(report_file, output, source_map, match_directory, expected_pairs, metadata_out):
        """Import directional matches from a local MOSS archive."""
        from .external_detectors.moss import compare_coverage, parse_report, write_moss_matches

        metadata_out = metadata_out or output.with_name(f"{output.name}.json")

        try:
            _check_output_paths((output, metadata_out), (report_file, source_map, expected_pairs))
            report = parse_report(
                report_file, source_map=_read_map(source_map), match_directory=match_directory,
            )
            metadata = dict(report.metadata)
            if expected_pairs is not None:
                with expected_pairs.open(encoding="utf-8-sig", newline="") as stream:
                    rows = csv.DictReader(stream)
                    if not {"left_id", "right_id"}.issubset(rows.fieldnames or ()):
                        raise ValueError("Expected pairs must have left_id and right_id columns.")
                    metadata["coverage"] = compare_coverage(
                        report, ((row["left_id"], row["right_id"]) for row in rows),
                    )
            output.parent.mkdir(parents=True, exist_ok=True)
            write_moss_matches(output, report)
            _write_json(metadata_out, metadata)
        except (OSError, TypeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote recorded MOSS matches to {output}.")
