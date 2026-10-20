"""CLI registration kept separate from the main comparison/evaluation commands."""

import json
from pathlib import Path

import click


@click.group(name="experiments")
def experiments_group():
    """Run controlled robustness and long-file input diagnostics."""


def _runs(config_file):
    if config_file is None:
        return None
    from .comparison_suite import load_run_configs

    return load_run_configs(config_file)


def _finish(report, artifacts):
    errors = sum(row["status"] != "ok" for row in report["scores"])
    click.echo(json.dumps({"experiment_id": report["experiment_id"],
                           "scored_rows": len(report["scores"]), "error_rows": errors,
                           "artifacts": artifacts}, indent=2, sort_keys=True))
    if errors:
        raise click.ClickException("Some scoring runs failed; error rows and null scores are in the artifacts.")


@experiments_group.command(name="robustness")
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--source", "source_file", type=click.Path(exists=True, dir_okay=False),
              help="Optional Python source; only AST-checked formatting/comments are generated.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def robustness_command(config_file, source_file, seed, output_dir):
    """Measure score changes on six explicit fixtures or AST-checked user source."""
    from .robustness import python_robustness_case, run_robustness_experiment

    try:
        runs = _runs(config_file)
        if runs is None:
            runs = [
                {"run_name": "levenshtein", "feature_weights": {"levenshtein": 1}},
                {"run_name": "winnowing", "feature_weights": {"winnowing": 1}},
                {"run_name": "static_hash", "feature_weights": {"semantic": 1},
                 "vector_backend": "static_hash"},
            ]
        cases = None
        if source_file:
            source = Path(source_file).read_text(encoding="utf-8")
            cases = [python_robustness_case(source, name, seed=seed)
                     for name in ("formatting", "comments")]
        report, artifacts = run_robustness_experiment(
            runs, cases=cases, seed=seed, output_dir=output_dir,
        )
    except (ValueError, SyntaxError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    _finish(report, artifacts)


@experiments_group.command(name="preflight")
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--options-file", type=click.Path(exists=True, dir_okay=False),
              help="JSON object with the same options as calculate_similarity.")
@click.option("--inspect-model", is_flag=True,
              help="Explicitly load the configured model/tokenizer; may contact its model hub.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False))
def preflight_command(source_file, options_file, inspect_model, output_path):
    """Report byte/line/chunk coverage and measured tokenizer truncation when supplied."""
    from .long_files import preflight_long_file
    from .model_routing import load_hf_model_info, resolve_vector_backend
    from .similarity import DEFAULT_MODEL_NAME, load_backend_model

    try:
        options = json.loads(Path(options_file).read_text()) if options_file else {}
        if not isinstance(options, dict):
            raise ValueError("Preflight options must be a JSON object.")
        model = None
        backend = options.get("vector_backend", "static_hash")
        if inspect_model and backend != "static_hash":
            model_name = options.get("model_name", DEFAULT_MODEL_NAME)
            if backend == "auto":
                backend = resolve_vector_backend(backend, model_name=model_name,
                                                 model_info=load_hf_model_info(model_name))
            options = {**options, "vector_backend": backend, "model_name": model_name}
            model = load_backend_model(
                model_name, vector_backend=backend, device=options.get("device", "cpu"),
                max_token_length=options.get("max_token_length"),
            )
        report = preflight_long_file(Path(source_file).read_text(encoding="utf-8"),
                                     options=options, model=model)
        raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(raw, encoding="utf-8")
        click.echo(raw.rstrip())
    except (ValueError, SyntaxError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiments_group.command(name="long-files")
@click.argument("left_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("right_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--inspect-models/--no-inspect-models", default=True, show_default=True,
              help="Inspect the explicit semantic model as well as running its scorer.")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def long_files_command(left_file, right_file, config_file, inspect_models, output_dir):
    """Compare existing raw/chunked/aggregation strategies on one declared file pair."""
    from .long_files import compare_long_files

    try:
        report, artifacts = compare_long_files(
            Path(left_file).read_text(encoding="utf-8"),
            Path(right_file).read_text(encoding="utf-8"), _runs(config_file),
            inspect_models=inspect_models, output_dir=output_dir,
        )
    except (ValueError, SyntaxError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    _finish(report, artifacts)


def register_experiment_commands(main):
    main.add_command(experiments_group)
