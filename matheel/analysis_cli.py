"""CLI registration kept separate from the core comparison commands."""

import click

from .result_analysis import analyze_scored_results, write_result_analysis


def register_analysis_commands(main):
    @main.command(name="analyze-results")
    @click.argument("scores", type=click.Path(exists=True, dir_okay=False))
    @click.option("--reference", type=click.Path(exists=True, dir_okay=False), help="Scores from another method on exactly the same candidates.")
    @click.option("--task", type=click.Choice(["auto", "pair", "retrieval"]), default="auto")
    @click.option("--threshold", type=float, default=0.5, show_default=True)
    @click.option("--k", type=click.IntRange(min=1), default=10, show_default=True)
    @click.option("--group-column", help="Independent assignment/author groups for pair resampling.")
    @click.option("--dataset", "dataset_root", type=click.Path(exists=True, file_okay=False), help="Optional normalized source dataset for language and character-length slices.")
    @click.option("--rounds", type=click.IntRange(1, 10000), default=500, show_default=True)
    @click.option("--seed", type=int, default=7, show_default=True)
    @click.option("--confidence", type=click.FloatRange(0, 1, min_open=True, max_open=True), default=0.95)
    @click.option("--output-dir", type=click.Path(file_okay=False), default="result_analysis", show_default=True)
    def analyze_results(scores, reference, task, threshold, k, group_column, dataset_root, rounds, seed, confidence, output_dir):
        """Inspect errors and grouped uncertainty without rerunning a method."""
        try:
            report = analyze_scored_results(scores, task=task, reference_rows=reference, threshold=threshold,
                                            k=k, group_column=group_column, rounds=rounds, seed=seed,
                                            confidence=confidence, dataset_root=dataset_root)
            artifacts = write_result_analysis(report, output_dir)
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        for name, path in artifacts.items():
            click.echo(f"{name}={path}")
