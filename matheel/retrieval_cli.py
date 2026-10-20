"""CLI registration for bounded retrieval experiments."""

import json
from pathlib import Path

import click

from .feature_weights import parse_feature_weights
from .retrieval import evaluate_two_stage_retrieval_dataset
from .reproducibility import collect_reproducibility_snapshot


def _options(path, weights, default):
    options = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(options, dict):
        raise ValueError("Scorer options files must contain a JSON object.")
    if weights:
        options["feature_weights"] = parse_feature_weights(",".join(weights))
    elif "feature_weights" not in options:
        options["feature_weights"] = default
    return options


@click.command("evaluate-two-stage-retrieval")
@click.argument("dataset", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--candidate-count", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--candidate-feature-weight", multiple=True, help="Existing cheap feature, e.g. lcs=1.")
@click.option("--feature-weight", multiple=True, help="Existing reranker feature, e.g. gst=1.")
@click.option("--candidate-options", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--reranker-options", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--k", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--exclude-self-matches", is_flag=True)
@click.option("--scores-out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--metrics-out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def evaluate_two_stage_command(dataset, candidate_count, candidate_feature_weight, feature_weight,
                               candidate_options, reranker_options, k, exclude_self_matches,
                               scores_out, metrics_out):
    """Rerank a bounded shortlist and audit recall against all dataset judgments."""
    try:
        if scores_out.resolve() == metrics_out.resolve():
            raise ValueError("Scores and metrics must use different output paths.")
        candidate = _options(candidate_options, candidate_feature_weight, {"lcs": 1.0})
        reranker = _options(reranker_options, feature_weight, {"levenshtein": 1.0})
        scored, metrics = evaluate_two_stage_retrieval_dataset(
            dataset, candidate_count, k=k, exclude_self_matches=exclude_self_matches,
            candidate_similarity_options=candidate, similarity_options=reranker,
        )
    except (ValueError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scores_out, index=False)
    metadata = collect_reproducibility_snapshot(run_configs=[{"candidate_options": candidate, "reranker_options": reranker,
                                                              "candidate_count": candidate_count, "k": k,
                                                              "exclude_self_matches": exclude_self_matches}])
    metrics_out.write_text(json.dumps({"metrics": metrics, "retrieval": scored.attrs["retrieval"], "reproducibility": metadata},
                                      indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    click.echo(f"Scored {len(scored)} candidates; candidate recall={metrics['candidate_recall']}; {metrics_out}")


def register_retrieval_commands(group):
    group.add_command(evaluate_two_stage_command)
