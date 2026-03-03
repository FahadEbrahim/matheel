import click
from .comparison_suite import load_run_configs, run_comparison_suite
from .code_metrics import available_code_metrics
from .similarity import available_runtime_devices, get_sim_list
from .chunking import available_chunk_aggregations, available_chunking_methods
from .preprocessing import available_preprocess_modes
from .vectors import available_vector_backends

@click.group()
def main():
    """Matheel CLI - Compute Code Similarity"""
    pass


@main.command()
@click.argument('source_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option('--ws', default=0.7, help='Semantic Similarity Weight')
@click.option('--wl', default=0.3, help='Levenshtein Distance Weight')
@click.option('--wj', default=0.0, help='Jaro-Winkler Distance Weight')
@click.option(
    '--feature-weight',
    'feature_weights',
    multiple=True,
    help='Optional normalized feature weights as name=value. Example: --feature-weight semantic=0.5 --feature-weight code_metric=0.5',
)
@click.option('--model', default='uclanlp/plbart-java-cs', help='Sentence Transformer Model')
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
    help='Split code into chunks before embedding. Chonkie-backed methods are used automatically when installed.',
)
@click.option('--chunk-size', default=200, show_default=True, help='Chunk size for line/token/character chunking.')
@click.option('--chunk-overlap', default=0, show_default=True, help='Chunk overlap.')
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
@click.option('--code-language', default='java', show_default=True, help='Language hint for code-level metrics. Official CodeBLEU scope: java, python, c, cpp.')
@click.option('--codebleu-component-weights', default='0.25,0.25,0.25,0.25', show_default=True, help='Comma-separated weights for ngram, weighted_ngram, syntax, dataflow.')
@click.option('--crystalbleu-max-order', default=4, show_default=True, help='Maximum n-gram order for CrystalBLEU.')
@click.option('--crystalbleu-trivial-ngram-count', default=500, show_default=True, help='How many frequent n-grams to ignore in CrystalBLEU.')
@click.option(
    '--vector-backend',
    type=click.Choice(available_vector_backends()),
    default='auto',
    show_default=True,
    help='Vector backend. Auto inspects Hugging Face model metadata and routes to sentence-transformers, model2vec, or PyLate.',
)
@click.option('--static-vector-dim', default=256, show_default=True, help='Fallback hashed static vector dimension.')
@click.option('--no-static-vector-lowercase', is_flag=True, help='Keep original token casing for static vectors.')
@click.option('--no-multivector-bidirectional', is_flag=True, help='Use one-way late interaction instead of averaging both directions.')
@click.option(
    '--device',
    type=click.Choice(("auto",) + available_runtime_devices()),
    default='auto',
    show_default=True,
    help='Embedding runtime device. Auto prefers CUDA, then MPS, then CPU.',
)
@click.option('--threshold', default=0.0, help='Similarity Threshold')
@click.option('--num', default=10, help='Number of Results to Display')
def compare(
    source_path,
    ws,
    wl,
    wj,
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
    vector_backend,
    static_vector_dim,
    no_static_vector_lowercase,
    no_multivector_bidirectional,
    device,
    threshold,
    num,
):
    """Bulk similarity from a ZIP archive or a directory."""
    results = get_sim_list(
        source_path,
        ws,
        wl,
        wj,
        model,
        threshold,
        num,
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
        vector_backend=vector_backend,
        static_vector_dim=static_vector_dim,
        static_vector_lowercase=(not no_static_vector_lowercase),
        multivector_bidirectional=(not no_multivector_bidirectional),
        device=device,
    )
    click.echo(results.to_string(index=False))


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
def compare_suite(source_path, config_file, summary_out, details_dir, output_format):
    """Run multiple similarity configurations from a JSON config file."""
    run_configs = load_run_configs(config_file)
    summary, _ = run_comparison_suite(
        source_path,
        run_configs,
        summary_out=summary_out,
        details_dir=details_dir,
        output_format=output_format,
    )
    if summary.empty:
        click.echo("No runs were executed.")
        return
    click.echo(summary.to_string(index=False))
