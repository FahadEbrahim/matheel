import json
import sys
from pathlib import Path

import click

from .comparison_suite import load_run_configs, run_comparison_suite
from .code_metrics import available_code_metrics
from ._progress import should_show_progress
from .datasets import load_pair_datasets, load_retrieval_datasets
from .evaluation import evaluate_pair_dataset, evaluate_retrieval_dataset
from .similarity import DEFAULT_MODEL_NAME, available_runtime_devices, get_sim_list
from .chunking import available_chunk_aggregations, available_chunking_methods
from .model_routing import available_vector_backends
from .preprocessing import available_preprocess_modes
from .vectors import (
    available_pooling_methods,
    available_similarity_functions,
)

@click.group()
def main():
    """Matheel CLI - Compute Code Similarity"""
    pass


def _parse_dataset_adapter_options(entries):
    options = {}
    for entry in entries or ():
        key, separator, value = str(entry).partition("=")
        key = key.strip()
        if not separator or not key:
            raise click.UsageError("--adapter-option values must use name=value.")
        options[key] = value
    return options


def _dataset_spec_from_cli(
    dataset_spec,
    *,
    task_family,
    preset,
    source,
    identifier,
    dataset_name,
    adapter,
    adapter_options,
    destination,
    adapted_destination,
    revision,
    split,
    path_in_archive,
):
    parsed_adapter_options = _parse_dataset_adapter_options(adapter_options)
    if preset and dataset_spec:
        raise click.UsageError("Use either DATASET_SPEC or --preset, not both.")
    if dataset_spec and identifier:
        raise click.UsageError("Use either DATASET_SPEC or --identifier, not both.")
    if parsed_adapter_options and adapter is None and preset is None:
        raise click.UsageError("--adapter-option requires --adapter or --preset.")

    adapter_key = "adapter" if task_family == "pair" else "retrieval_adapter"
    optional_values = {
        "source": source,
        "identifier": identifier,
        "name": dataset_name,
        adapter_key: adapter,
        "adapter_options": parsed_adapter_options or None,
        "destination": destination,
        "adapted_destination": adapted_destination,
        "revision": revision,
        "split": split,
        "path_in_archive": path_in_archive,
    }

    if preset:
        payload = {"preset": preset}
        payload.update({key: value for key, value in optional_values.items() if value is not None})
        return payload

    if dataset_spec is None and identifier is None:
        raise click.UsageError("Provide DATASET_SPEC, --identifier, or --preset.")

    resolved_identifier = identifier if identifier is not None else dataset_spec
    has_dataset_controls = any(value is not None for value in optional_values.values())
    if not has_dataset_controls:
        return resolved_identifier

    payload = {"identifier": resolved_identifier}
    payload.update({key: value for key, value in optional_values.items() if value is not None})
    return payload


@main.command()
@click.argument('source_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option(
    '--feature-weight',
    'feature_weights',
    multiple=True,
    help='Normalized feature weights as name=value. Example: --feature-weight semantic=0.5 --feature-weight code_metric=0.5',
)
@click.option('--model', default=DEFAULT_MODEL_NAME, show_default=True, help='Embedding model name.')
@click.option(
    '--preprocess-mode',
    type=click.Choice(available_preprocess_modes()),
    default='none',
    show_default=True,
    help='Preprocess code before scoring.',
)
@click.option(
    '--chunking-method',
    type=click.Choice(available_chunking_methods()),
    default='none',
    show_default=True,
    help='Split code into chunks before embedding. Non-none methods require Chonkie.',
)
@click.option('--chunk-size', default=200, show_default=True, help='Chunk size hint for chunkers that support it.')
@click.option('--chunk-overlap', default=0, show_default=True, help='Chunk overlap hint for chunkers that support it.')
@click.option('--max-chunks', default=0, show_default=True, help='Limit chunks per file. 0 keeps all chunks.')
@click.option('--chunk-language', default='text', show_default=True, help='Language hint for language-aware chunkers such as Chonkie CodeChunker.')
@click.option(
    '--chunker-option',
    'chunker_options',
    multiple=True,
    help='Extra chunker option as name=value. Repeat to pass multiple options.',
)
@click.option(
    '--chunk-aggregation',
    type=click.Choice(available_chunk_aggregations()),
    default='mean',
    show_default=True,
    help='How chunk embeddings are reduced to one file embedding.',
)
@click.option(
    '--code-metric',
    type=click.Choice(available_code_metrics()),
    default='none',
    show_default=True,
    help='Optional code-level metric added to the final score.',
)
@click.option('--code-metric-weight', default=0.0, show_default=True, help='Weight applied to the selected code metric.')
@click.option('--code-language', default='java', show_default=True, help='Language hint for code-level metrics. Native CodeBLEU, RUBY, and TSED support: java, python, c, cpp, go, javascript, typescript, kotlin, scala, swift, solidity, dart, php, ruby, rust, csharp, lua, julia, r, objc.')
@click.option('--codebleu-component-weights', default='0.25,0.25,0.25,0.25', show_default=True, help='Comma-separated weights for ngram, weighted_ngram, syntax, dataflow.')
@click.option('--crystalbleu-max-order', default=4, show_default=True, help='Maximum n-gram order for CrystalBLEU.')
@click.option('--crystalbleu-trivial-ngram-count', default=50, show_default=True, help='How many frequent n-grams to ignore in CrystalBLEU.')
@click.option('--ruby-max-order', default=4, show_default=True, help='Maximum n-gram order for RUBY.')
@click.option('--ruby-epsilon', default=1e-12, show_default=True, help='Small epsilon used by RUBY F1 smoothing.')
@click.option(
    '--ruby-mode',
    type=click.Choice(("auto", "graph", "tree", "string", "ngram")),
    default='auto',
    show_default=True,
    help='RUBY strategy selection.',
)
@click.option(
    '--ruby-tokenizer',
    type=click.Choice(("tranx", "regex")),
    default='tranx',
    show_default=True,
    help='Tokenizer used by RUBY string mode.',
)
@click.option(
    '--ruby-denominator',
    type=click.Choice(("max", "mean")),
    default='max',
    show_default=True,
    help='Length normalization denominator used by RUBY string mode.',
)
@click.option('--ruby-graph-timeout-seconds', default=1.0, show_default=True, help='Per-step timeout (seconds) for RUBY graph edit search.')
@click.option('--ruby-graph-use-edge-cost/--no-ruby-graph-use-edge-cost', default=True, show_default=True, help='Whether RUBY graph mode includes edge insertion/deletion cost.')
@click.option('--ruby-graph-include-leaf-edges/--no-ruby-graph-include-leaf-edges', default=True, show_default=True, help='Whether RUBY graph mode adds sequential leaf edges.')
@click.option('--ruby-tree-max-nodes', default=180, show_default=True, help='Maximum parsed tree nodes for RUBY tree/graph modes.')
@click.option('--ruby-tree-max-depth', default=10, show_default=True, help='Maximum parsed tree depth for RUBY tree/graph modes.')
@click.option('--ruby-tree-max-children', default=8, show_default=True, help='Maximum parsed children per node for RUBY tree/graph modes.')
@click.option('--tsed-delete-cost', default=1.0, show_default=True, help='Delete edit cost for TSED.')
@click.option('--tsed-insert-cost', default=1.0, show_default=True, help='Insert edit cost for TSED.')
@click.option('--tsed-rename-cost', default=1.0, show_default=True, help='Rename edit cost for TSED.')
@click.option('--tsed-max-nodes', default=180, show_default=True, help='Maximum AST nodes sampled per file for TSED.')
@click.option('--tsed-max-depth', default=10, show_default=True, help='Maximum AST depth for TSED.')
@click.option('--tsed-max-children', default=8, show_default=True, help='Maximum AST children sampled per node for TSED.')
@click.option('--codebertscore-model', default='microsoft/codebert-base', show_default=True, help='Model name used for CodeBERTScore.')
@click.option('--codebertscore-num-layers', default=0, show_default=True, help='Hidden layer count for CodeBERTScore. 0 uses model defaults.')
@click.option('--codebertscore-batch-size', default=16, show_default=True, help='Batch size for CodeBERTScore.')
@click.option('--codebertscore-max-length', default=0, show_default=True, help='Max tokenizer length for CodeBERTScore. 0 uses model limits.')
@click.option(
    '--codebertscore-device',
    type=click.Choice(("auto",) + available_runtime_devices()),
    default='auto',
    show_default=True,
    help='Runtime device for CodeBERTScore.',
)
@click.option('--codebertscore-lang', default='', show_default=True, help='Optional language code passed to CodeBERTScore.')
@click.option('--codebertscore-idf/--no-codebertscore-idf', default=False, show_default=True, help='Enable IDF weighting for CodeBERTScore.')
@click.option(
    '--codebertscore-rescale-with-baseline/--no-codebertscore-rescale-with-baseline',
    default=False,
    show_default=True,
    help='Enable baseline rescaling for CodeBERTScore.',
)
@click.option(
    '--codebertscore-use-fast-tokenizer/--no-codebertscore-use-fast-tokenizer',
    default=False,
    show_default=True,
    help='Use fast tokenizer for CodeBERTScore.',
)
@click.option('--codebertscore-nthreads', default=4, show_default=True, help='Thread count for CodeBERTScore.')
@click.option('--codebertscore-verbose/--no-codebertscore-verbose', default=False, show_default=True, help='Verbose CodeBERTScore output.')
@click.option('--levenshtein-weights', default='1,1,1', show_default=True, help='Comma-separated insert,delete,substitute costs for Levenshtein.')
@click.option('--jaro-winkler-prefix-weight', default=0.1, show_default=True, help='Prefix bonus weight for Jaro-Winkler (0.0 to 0.25).')
@click.option('--winnowing-kgram', default=5, show_default=True, help='Token k-gram size used by Winnowing.')
@click.option('--winnowing-window', default=4, show_default=True, help='Window size used by Winnowing fingerprint selection.')
@click.option('--gst-min-match-length', default=5, show_default=True, help='Minimum token tile length used by Greedy String Tiling.')
@click.option(
    '--vector-backend',
    type=click.Choice(available_vector_backends()),
    default='auto',
    show_default=True,
    help='Vector backend. Auto inspects Hugging Face model metadata and routes to sentence-transformers, model2vec, or PyLate.',
)
@click.option(
    '--similarity-function',
    type=click.Choice(available_similarity_functions()),
    default='cosine',
    show_default=True,
    help='Semantic similarity function for single-vector backends.',
)
@click.option(
    '--normalize-semantic-scores/--raw-semantic-scores',
    default=False,
    show_default=True,
    help='Normalize single-vector semantic scores to 0-1 before weighted blending.',
)
@click.option('--static-vector-dim', default=256, show_default=True, help='Dimension for local static-vector compatibility fallback.')
@click.option('--max-token-length', default=0, show_default=True, help='Maximum token length to use for the selected model. 0 keeps the model default.')
@click.option(
    '--pooling-method',
    type=click.Choice(available_pooling_methods()),
    default='mean',
    show_default=True,
    help='Sentence-transformers pooling mode. Ignored by model2vec and PyLate.',
)
@click.option(
    '--device',
    type=click.Choice(("auto",) + available_runtime_devices()),
    default='auto',
    show_default=True,
    help='Embedding runtime device. Auto prefers CUDA, then MPS, then CPU.',
)
@click.option('--threshold', default=0.0, help='Similarity Threshold')
@click.option('--num', default=10, help='Number of Results to Display')
@click.option(
    '--progress/--no-progress',
    default=None,
    help='Show progress bars on stderr. Defaults to auto for interactive terminals.',
)
def compare(
    source_path,
    feature_weights,
    model,
    preprocess_mode,
    chunking_method,
    chunk_size,
    chunk_overlap,
    max_chunks,
    chunk_language,
    chunker_options,
    chunk_aggregation,
    code_metric,
    code_metric_weight,
    code_language,
    codebleu_component_weights,
    crystalbleu_max_order,
    crystalbleu_trivial_ngram_count,
    ruby_max_order,
    ruby_epsilon,
    ruby_mode,
    ruby_tokenizer,
    ruby_denominator,
    ruby_graph_timeout_seconds,
    ruby_graph_use_edge_cost,
    ruby_graph_include_leaf_edges,
    ruby_tree_max_nodes,
    ruby_tree_max_depth,
    ruby_tree_max_children,
    tsed_delete_cost,
    tsed_insert_cost,
    tsed_rename_cost,
    tsed_max_nodes,
    tsed_max_depth,
    tsed_max_children,
    codebertscore_model,
    codebertscore_num_layers,
    codebertscore_batch_size,
    codebertscore_max_length,
    codebertscore_device,
    codebertscore_lang,
    codebertscore_idf,
    codebertscore_rescale_with_baseline,
    codebertscore_use_fast_tokenizer,
    codebertscore_nthreads,
    codebertscore_verbose,
    levenshtein_weights,
    jaro_winkler_prefix_weight,
    winnowing_kgram,
    winnowing_window,
    gst_min_match_length,
    vector_backend,
    similarity_function,
    normalize_semantic_scores,
    static_vector_dim,
    max_token_length,
    pooling_method,
    device,
    threshold,
    num,
    progress,
):
    """Bulk similarity from a ZIP archive or a directory."""
    results = get_sim_list(
        source_path,
        model_name=model,
        threshold=threshold,
        number_results=num,
        feature_weights=feature_weights,
        preprocess_mode=preprocess_mode,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
        chunk_language=chunk_language,
        chunker_options=chunker_options,
        chunk_aggregation=chunk_aggregation,
        code_metric=code_metric,
        code_metric_weight=code_metric_weight,
        code_language=code_language,
        codebleu_component_weights=codebleu_component_weights,
        crystalbleu_max_order=crystalbleu_max_order,
        crystalbleu_trivial_ngram_count=crystalbleu_trivial_ngram_count,
        ruby_max_order=ruby_max_order,
        ruby_epsilon=ruby_epsilon,
        ruby_mode=ruby_mode,
        ruby_tokenizer=ruby_tokenizer,
        ruby_denominator=ruby_denominator,
        ruby_graph_timeout_seconds=ruby_graph_timeout_seconds,
        ruby_graph_use_edge_cost=ruby_graph_use_edge_cost,
        ruby_graph_include_leaf_edges=ruby_graph_include_leaf_edges,
        ruby_tree_max_nodes=ruby_tree_max_nodes,
        ruby_tree_max_depth=ruby_tree_max_depth,
        ruby_tree_max_children=ruby_tree_max_children,
        tsed_delete_cost=tsed_delete_cost,
        tsed_insert_cost=tsed_insert_cost,
        tsed_rename_cost=tsed_rename_cost,
        tsed_max_nodes=tsed_max_nodes,
        tsed_max_depth=tsed_max_depth,
        tsed_max_children=tsed_max_children,
        codebertscore_model=codebertscore_model,
        codebertscore_num_layers=(None if int(codebertscore_num_layers or 0) <= 0 else int(codebertscore_num_layers)),
        codebertscore_batch_size=codebertscore_batch_size,
        codebertscore_max_length=codebertscore_max_length,
        codebertscore_device=codebertscore_device,
        codebertscore_lang=(str(codebertscore_lang).strip() or None),
        codebertscore_idf=codebertscore_idf,
        codebertscore_rescale_with_baseline=codebertscore_rescale_with_baseline,
        codebertscore_use_fast_tokenizer=codebertscore_use_fast_tokenizer,
        codebertscore_nthreads=codebertscore_nthreads,
        codebertscore_verbose=codebertscore_verbose,
        levenshtein_weights=levenshtein_weights,
        jaro_winkler_prefix_weight=jaro_winkler_prefix_weight,
        winnowing_kgram=winnowing_kgram,
        winnowing_window=winnowing_window,
        gst_min_match_length=gst_min_match_length,
        vector_backend=vector_backend,
        similarity_function=similarity_function,
        normalize_semantic_scores=normalize_semantic_scores,
        static_vector_dim=static_vector_dim,
        max_token_length=max_token_length,
        pooling_method=pooling_method,
        device=device,
        progress=should_show_progress(progress, stream=sys.stderr),
    )
    click.echo(results.to_string(index=False))
    click.echo(_elapsed_summary_text(results), err=True)


@main.command(name="compare-suite")
@click.argument("source_path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--summary-out",
    type=click.Path(),
    default="comparison_summary.csv",
    show_default=True,
    help="Where to write the summary table.",
)
@click.option(
    "--details-dir",
    type=click.Path(),
    default="comparison_runs",
    show_default=True,
    help="Directory where each run's detailed CSV is written.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("csv", "json")),
    default="csv",
    show_default=True,
    help="Summary file format.",
)
@click.option(
    "--progress/--no-progress",
    default=None,
    help="Show progress bars on stderr. Defaults to auto for interactive terminals.",
)
def compare_suite(source_path, config_file, summary_out, details_dir, output_format, progress):
    """Run multiple similarity configurations from a JSON config file."""
    run_configs = load_run_configs(config_file)
    summary, _ = run_comparison_suite(
        source_path,
        run_configs,
        summary_out=summary_out,
        details_dir=details_dir,
        output_format=output_format,
        progress=should_show_progress(progress, stream=sys.stderr),
    )
    if summary.empty:
        click.echo("No runs were executed.")
        return
    click.echo(summary.to_string(index=False))


@main.command(name="evaluate-pairs")
@click.argument("dataset_spec", required=False, metavar="DATASET_SPEC")
@click.option("--preset", help="Registered dataset preset to load.")
@click.option("--source", help="Dataset source resolver to use with DATASET_SPEC or --identifier.")
@click.option("--identifier", help="Dataset source identifier. Use this instead of DATASET_SPEC when clearer.")
@click.option("--dataset-name", help="Name to assign to the loaded or adapted dataset.")
@click.option("--adapter", help="Dataset adapter to apply before evaluation.")
@click.option(
    "--adapter-option",
    "adapter_options",
    multiple=True,
    help="Adapter option as name=value. Repeat to pass multiple options.",
)
@click.option(
    "--destination",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where a source resolver should place raw data.",
)
@click.option(
    "--adapted-destination",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where an adapter should write the normalized dataset.",
)
@click.option("--revision", help="Source revision, branch, tag, or version when the resolver supports it.")
@click.option("--split", help="Dataset split name forwarded to the resolver or adapter.")
@click.option("--path-in-archive", help="Relative nested path inside the resolved source root.")
@click.option("--scores-out", type=click.Path(), default="scored_pairs.csv", show_default=True)
@click.option("--metrics-out", type=click.Path(), default="pair_metrics.json", show_default=True)
@click.option("--threshold", default=0.5, show_default=True, help="Classification threshold.")
@click.option(
    "--feature-weight",
    "feature_weights",
    multiple=True,
    help=(
        "Normalized feature weights as name=value. "
        "Defaults to levenshtein=1.0 for offline-friendly evaluation."
    ),
)
@click.option("--model", default=DEFAULT_MODEL_NAME, show_default=True, help="Embedding model name.")
@click.option(
    "--preprocess-mode",
    type=click.Choice(available_preprocess_modes()),
    default="none",
    show_default=True,
)
@click.option("--code-language", default="python", show_default=True)
@click.option(
    "--vector-backend",
    type=click.Choice(available_vector_backends()),
    default="auto",
    show_default=True,
)
@click.option(
    "--similarity-function",
    type=click.Choice(available_similarity_functions()),
    default="cosine",
    show_default=True,
)
@click.option(
    "--normalize-semantic-scores/--raw-semantic-scores",
    default=False,
    show_default=True,
)
@click.option("--max-token-length", default=0, show_default=True)
@click.option(
    "--pooling-method",
    type=click.Choice(available_pooling_methods()),
    default="mean",
    show_default=True,
)
@click.option(
    "--device",
    type=click.Choice(("auto",) + available_runtime_devices()),
    default="auto",
    show_default=True,
)
def evaluate_pairs(
    dataset_spec,
    preset,
    source,
    identifier,
    dataset_name,
    adapter,
    adapter_options,
    destination,
    adapted_destination,
    revision,
    split,
    path_in_archive,
    scores_out,
    metrics_out,
    threshold,
    feature_weights,
    model,
    preprocess_mode,
    code_language,
    vector_backend,
    similarity_function,
    normalize_semantic_scores,
    max_token_length,
    pooling_method,
    device,
):
    """Evaluate a pair-classification dataset from a path, source spec, or preset."""
    resolved_dataset = load_pair_datasets(
        _dataset_spec_from_cli(
            dataset_spec,
            task_family="pair",
            preset=preset,
            source=source,
            identifier=identifier,
            dataset_name=dataset_name,
            adapter=adapter,
            adapter_options=adapter_options,
            destination=destination,
            adapted_destination=adapted_destination,
            revision=revision,
            split=split,
            path_in_archive=path_in_archive,
        )
    )
    selected_weights = feature_weights or ("levenshtein=1.0",)
    scored_pairs, metrics = evaluate_pair_dataset(
        resolved_dataset,
        threshold=threshold,
        similarity_options={
            "feature_weights": selected_weights,
            "model_name": model,
            "preprocess_mode": preprocess_mode,
            "code_language": code_language,
            "vector_backend": vector_backend,
            "similarity_function": similarity_function,
            "normalize_semantic_scores": normalize_semantic_scores,
            "max_token_length": max_token_length,
            "pooling_method": pooling_method,
            "device": device,
        },
    )
    scores_path = Path(scores_out)
    metrics_path = Path(metrics_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    scored_pairs.to_csv(scores_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    click.echo(f"Wrote {len(scored_pairs)} scored pair(s) to {scores_path}.")
    click.echo(f"Wrote metrics to {metrics_path}.")
    click.echo(
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


@main.command(name="evaluate-retrieval")
@click.argument("dataset_spec", required=False, metavar="DATASET_SPEC")
@click.option("--preset", help="Registered dataset preset to load.")
@click.option("--source", help="Dataset source resolver to use with DATASET_SPEC or --identifier.")
@click.option("--identifier", help="Dataset source identifier. Use this instead of DATASET_SPEC when clearer.")
@click.option("--dataset-name", help="Name to assign to the loaded or adapted dataset.")
@click.option("--adapter", help="Dataset adapter to apply before evaluation.")
@click.option(
    "--adapter-option",
    "adapter_options",
    multiple=True,
    help="Adapter option as name=value. Repeat to pass multiple options.",
)
@click.option(
    "--destination",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where a source resolver should place raw data.",
)
@click.option(
    "--adapted-destination",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where an adapter should write the normalized dataset.",
)
@click.option("--revision", help="Source revision, branch, tag, or version when the resolver supports it.")
@click.option("--split", help="Dataset split name forwarded to the resolver or adapter.")
@click.option("--path-in-archive", help="Relative nested path inside the resolved source root.")
@click.option("--scores-out", type=click.Path(), default="scored_retrieval.csv", show_default=True)
@click.option("--metrics-out", type=click.Path(), default="retrieval_metrics.json", show_default=True)
@click.option("--k", default=10, show_default=True, help="Ranking cutoff for top-k metrics.")
@click.option(
    "--feature-weight",
    "feature_weights",
    multiple=True,
    help=(
        "Normalized feature weights as name=value. "
        "Defaults to levenshtein=1.0 for offline-friendly evaluation."
    ),
)
@click.option("--model", default=DEFAULT_MODEL_NAME, show_default=True, help="Embedding model name.")
@click.option(
    "--preprocess-mode",
    type=click.Choice(available_preprocess_modes()),
    default="none",
    show_default=True,
)
@click.option("--code-language", default="python", show_default=True)
@click.option(
    "--vector-backend",
    type=click.Choice(available_vector_backends()),
    default="auto",
    show_default=True,
)
@click.option(
    "--similarity-function",
    type=click.Choice(available_similarity_functions()),
    default="cosine",
    show_default=True,
)
@click.option(
    "--normalize-semantic-scores/--raw-semantic-scores",
    default=False,
    show_default=True,
)
@click.option("--max-token-length", default=0, show_default=True)
@click.option(
    "--pooling-method",
    type=click.Choice(available_pooling_methods()),
    default="mean",
    show_default=True,
)
@click.option(
    "--device",
    type=click.Choice(("auto",) + available_runtime_devices()),
    default="auto",
    show_default=True,
)
def evaluate_retrieval(
    dataset_spec,
    preset,
    source,
    identifier,
    dataset_name,
    adapter,
    adapter_options,
    destination,
    adapted_destination,
    revision,
    split,
    path_in_archive,
    scores_out,
    metrics_out,
    k,
    feature_weights,
    model,
    preprocess_mode,
    code_language,
    vector_backend,
    similarity_function,
    normalize_semantic_scores,
    max_token_length,
    pooling_method,
    device,
):
    """Evaluate a retrieval dataset from a path, source spec, or preset."""
    resolved_dataset = load_retrieval_datasets(
        _dataset_spec_from_cli(
            dataset_spec,
            task_family="retrieval",
            preset=preset,
            source=source,
            identifier=identifier,
            dataset_name=dataset_name,
            adapter=adapter,
            adapter_options=adapter_options,
            destination=destination,
            adapted_destination=adapted_destination,
            revision=revision,
            split=split,
            path_in_archive=path_in_archive,
        )
    )
    selected_weights = feature_weights or ("levenshtein=1.0",)
    scored_results, metrics = evaluate_retrieval_dataset(
        resolved_dataset,
        k=k,
        similarity_options={
            "feature_weights": selected_weights,
            "model_name": model,
            "preprocess_mode": preprocess_mode,
            "code_language": code_language,
            "vector_backend": vector_backend,
            "similarity_function": similarity_function,
            "normalize_semantic_scores": normalize_semantic_scores,
            "max_token_length": max_token_length,
            "pooling_method": pooling_method,
            "device": device,
        },
    )
    scores_path = Path(scores_out)
    metrics_path = Path(metrics_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    scored_results.to_csv(scores_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    click.echo(f"Wrote {len(scored_results)} scored retrieval result(s) to {scores_path}.")
    click.echo(f"Wrote metrics to {metrics_path}.")
    click.echo(
        f"map={metrics['mean_average_precision']:.4f} "
        f"mrr={metrics['mean_reciprocal_rank']:.4f} "
        f"precision_at_{metrics['k']}={metrics['precision_at_k']:.4f} "
        f"recall_at_{metrics['k']}={metrics['recall_at_k']:.4f} "
        f"ndcg_at_{metrics['k']}={metrics['ndcg_at_k']:.4f}"
    )


def _elapsed_summary_text(results):
    elapsed_seconds = float(results.attrs.get("elapsed_seconds", 0.0))
    feature_set = results.attrs.get("feature_set", "none")
    vector_backend = results.attrs.get("vector_backend", "auto")
    code_metric = results.attrs.get("code_metric", "none")
    chunking_method = results.attrs.get("chunking_method", "none")
    return (
        f"Elapsed: {elapsed_seconds:.4f}s | "
        f"features={feature_set} | "
        f"backend={vector_backend} | "
        f"code_metric={code_metric} | "
        f"chunking={chunking_method}"
    )
