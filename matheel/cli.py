import json
import sys
from pathlib import Path

import click

from . import __version__
from .algorithms import normalize_algorithm_options, score_source_pairs_with_algorithm
from .comparison_suite import load_run_configs, run_comparison_suite
from .code_metrics import available_code_metrics
from ._progress import should_show_progress
from .datasets import (
    adapt_pair_dataset,
    adapt_retrieval_dataset,
    available_dataset_adapters,
    available_dataset_presets,
    available_dataset_presets_by_task,
    available_dataset_sources,
    dataset_preset_task_families,
    load_pair_dataset,
    load_pair_datasets,
    load_pair_datasets_from_manifest,
    load_retrieval_dataset,
    load_retrieval_datasets,
    load_retrieval_datasets_from_manifest,
)
from .evaluation import evaluate_pair_dataset, evaluate_retrieval_dataset
from .reproducibility import collect_reproducibility_snapshot, write_reproducibility_snapshot
from .similarity import (
    DEFAULT_MODEL_NAME,
    available_lexical_tokenizers,
    available_runtime_devices,
    extract_and_read_source,
    get_sim_list,
)
from .chunking import available_chunk_aggregations, available_chunking_methods
from .model_routing import available_vector_backends
from .preprocessing import available_preprocess_modes
from .vectors import (
    available_pooling_methods,
    available_similarity_functions,
)
from .visualization import (
    available_pair_explanation_segment_modes,
    available_projection_methods,
    write_dataset_embedding_map,
    write_pair_dataset_explanation,
    write_pair_explanation,
)

@click.group()
@click.version_option(version=__version__, prog_name="matheel")
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


def _coerce_cli_option_value(raw_value):
    text = str(raw_value).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_algorithm_options(entries):
    options = {}
    for entry in entries or ():
        key, separator, value = str(entry).partition("=")
        key = key.strip()
        if not separator or not key:
            raise click.UsageError("--algorithm-option values must use name=value.")
        options[key] = _coerce_cli_option_value(value)
    try:
        return normalize_algorithm_options(options)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _write_cli_reproducibility(output_path, source_path, run_name, options, results):
    if not output_path:
        return None
    snapshot = collect_reproducibility_snapshot(
        source_path,
        run_configs=[{"run_name": run_name, "options": options}],
        result_attrs=getattr(results, "attrs", {}),
    )
    return write_reproducibility_snapshot(snapshot, output_path)


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


def _ensure_manifest_has_no_dataset_cli_overrides(
    manifest,
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
):
    if manifest is None:
        return
    overrides = {
        "DATASET_SPEC": dataset_spec,
        "--preset": preset,
        "--source": source,
        "--identifier": identifier,
        "--dataset-name": dataset_name,
        "--adapter": adapter,
        "--destination": destination,
        "--adapted-destination": adapted_destination,
        "--revision": revision,
        "--split": split,
        "--path-in-archive": path_in_archive,
    }
    used = [name for name, value in overrides.items() if value is not None]
    if adapter_options:
        used.append("--adapter-option")
    if used:
        joined = ", ".join(used)
        raise click.UsageError(f"--manifest cannot be combined with dataset source options: {joined}.")


def _echo_json(payload):
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def _dataset_list_payload(task):
    if task is None:
        preset_names = available_dataset_presets()
    else:
        preset_names = available_dataset_presets_by_task(task)
    return {
        "sources": list(available_dataset_sources()),
        "adapters": list(available_dataset_adapters()),
        "presets": [
            {
                "name": name,
                "task_families": list(dataset_preset_task_families(name)),
            }
            for name in preset_names
        ],
    }


def _dataset_kind_from_path(dataset_path, requested_kind):
    if requested_kind != "auto":
        return requested_kind

    root = Path(dataset_path)
    if all((root / name).exists() for name in ("files.csv", "queries.csv", "corpus.csv", "qrels.csv")):
        return "retrieval"
    if (root / "files.csv").exists() and (root / "pairs.csv").exists():
        return "pair"
    raise click.UsageError("Could not detect dataset kind. Use --kind pair or --kind retrieval.")


def _pair_dataset_summary(dataset):
    positive_count = int(dataset.pairs["label"].sum()) if "label" in dataset.pairs else 0
    pair_count = int(len(dataset.pairs))
    return {
        "dataset_kind": "pair_classification",
        "name": str(dataset.metadata.get("name") or dataset.root.name),
        "counts": {
            "files": int(len(dataset.files)),
            "pairs": pair_count,
            "positive_pairs": positive_count,
            "negative_pairs": pair_count - positive_count,
        },
        "metadata": dataset.metadata,
    }


def _retrieval_dataset_summary(dataset):
    return {
        "dataset_kind": "retrieval",
        "name": str(dataset.metadata.get("name") or dataset.root.name),
        "counts": {
            "files": int(len(dataset.files)),
            "queries": int(len(dataset.queries)),
            "documents": int(len(dataset.corpus)),
            "qrels": int(len(dataset.qrels)),
        },
        "metadata": dataset.metadata,
    }


def _echo_dataset_summary(summary, output_format):
    if output_format == "json":
        _echo_json(summary)
        return

    click.echo(f"name={summary['name']}")
    click.echo(f"dataset_kind={summary['dataset_kind']}")
    for key, value in summary["counts"].items():
        click.echo(f"{key}={value}")
    task_type = summary["metadata"].get("task_type")
    if task_type:
        click.echo(f"task_type={task_type}")


def _echo_visualization_summary(summary, output_format):
    if output_format == "json":
        _echo_json(summary)
        return
    click.echo(f"dataset_name={summary['dataset_name']}")
    click.echo(f"dataset_kind={summary['dataset_kind']}")
    click.echo(f"projection_method={summary['projection_method']}")
    click.echo(f"documents={summary['documents']}")
    for name, path in summary["artifacts"].items():
        click.echo(f"{name}={path}")


def _echo_pair_explanation_summary(summary, output_format):
    if output_format == "json":
        _echo_json(summary)
        return
    click.echo(f"left_id={summary['left_id']}")
    click.echo(f"right_id={summary['right_id']}")
    click.echo(f"segment_mode={summary['segment_mode']}")
    click.echo(f"matches={summary['matches']}")
    for level, count in summary["levels"].items():
        click.echo(f"{level}={count}")
    for name, path in summary["artifacts"].items():
        click.echo(f"{name}={path}")


def _pair_explanation_summary(explanation, artifacts):
    levels = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for side in ("left", "right"):
        for segment in explanation[side]["segments"]:
            level = str(segment.get("level") or "none")
            levels[level if level in levels else "none"] += 1
    metadata = explanation["metadata"]
    return {
        "left_id": metadata["left_id"],
        "right_id": metadata["right_id"],
        "segment_mode": metadata["segment_mode"],
        "matches": int(len(explanation.get("matches", []))),
        "levels": levels,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


def _source_pair_texts(source_path, left_name, right_name):
    if not left_name or not right_name:
        raise click.UsageError("--source requires --left-name and --right-name.")
    file_names, codes = extract_and_read_source(source_path)
    by_name = dict(zip(file_names, codes, strict=True))
    missing = [name for name in (left_name, right_name) if name not in by_name]
    if missing:
        joined = ", ".join(missing)
        raise click.UsageError(f"Source does not contain requested file(s): {joined}")
    return by_name[left_name], by_name[right_name]


@main.group(name="datasets")
def datasets_cli():
    """Inspect, validate, and adapt Matheel datasets."""
    pass


@datasets_cli.command(name="list")
@click.option("--task", type=click.Choice(("pair", "retrieval")), help="Filter presets by task.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def datasets_list(task, output_format):
    """List registered dataset sources, adapters, and presets."""
    payload = _dataset_list_payload(task)
    if output_format == "json":
        _echo_json(payload)
        return

    click.echo("Sources:")
    for source in payload["sources"]:
        click.echo(f"- {source}")
    click.echo("Adapters:")
    for adapter in payload["adapters"]:
        click.echo(f"- {adapter}")
    click.echo("Presets:")
    for preset in payload["presets"]:
        tasks = ", ".join(preset["task_families"])
        click.echo(f"- {preset['name']} ({tasks})")


@datasets_cli.command(name="validate")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--kind",
    type=click.Choice(("auto", "pair", "retrieval")),
    default="auto",
    show_default=True,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def datasets_validate(dataset_path, kind, output_format):
    """Validate a normalized pair or retrieval dataset."""
    dataset_kind = _dataset_kind_from_path(dataset_path, kind)
    if dataset_kind == "pair":
        summary = _pair_dataset_summary(load_pair_dataset(dataset_path))
    else:
        summary = _retrieval_dataset_summary(load_retrieval_dataset(dataset_path))
    _echo_dataset_summary(summary, output_format)


@datasets_cli.command(name="adapt")
@click.argument("source_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--kind", type=click.Choice(("pair", "retrieval")), required=True)
@click.option("--adapter", help="Dataset adapter name. Defaults to the generic tabular adapter.")
@click.option(
    "--adapter-option",
    "adapter_options",
    multiple=True,
    help="Adapter option as name=value. Repeat to pass multiple options.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory where the normalized dataset will be written.",
)
@click.option("--dataset-name", help="Name to write into normalized dataset metadata.")
@click.option("--split", help="Dataset split name forwarded to the adapter.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def datasets_adapt(
    source_path,
    kind,
    adapter,
    adapter_options,
    output_path,
    dataset_name,
    split,
    output_format,
):
    """Adapt a local raw dataset into Matheel's normalized manifests."""
    parsed_options = _parse_dataset_adapter_options(adapter_options)
    if split is not None and "split" not in parsed_options:
        parsed_options["split"] = split

    if kind == "pair":
        adapter_name = adapter or "auto_pair_tabular"
        adapted_root = adapt_pair_dataset(
            source_path,
            adapter=adapter_name,
            destination=output_path,
            dataset_name=dataset_name,
            adapter_options=parsed_options,
        )
        summary = _pair_dataset_summary(load_pair_dataset(adapted_root))
    else:
        adapter_name = adapter or "auto_retrieval_tabular"
        adapted_root = adapt_retrieval_dataset(
            source_path,
            adapter=adapter_name,
            destination=output_path,
            dataset_name=dataset_name,
            adapter_options=parsed_options,
        )
        summary = _retrieval_dataset_summary(load_retrieval_dataset(adapted_root))
    summary["adapter"] = adapter_name
    _echo_dataset_summary(summary, output_format)


@main.command(name="visualize-dataset")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--kind",
    type=click.Choice(("auto", "pair", "retrieval")),
    default="auto",
    show_default=True,
    help="Normalized dataset kind. Auto detects pair or retrieval manifests.",
)
@click.option(
    "--method",
    type=click.Choice(available_projection_methods()),
    default="auto",
    show_default=True,
    help="Projection method. Auto uses UMAP when installed, otherwise deterministic PCA.",
)
@click.option("--seed", default=7, show_default=True, help="Projection seed for stochastic reducers.")
@click.option("--static-vector-dim", default=256, show_default=True, help="Static hash embedding dimension.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory where dataset map artifacts will be written.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def visualize_dataset(dataset_path, kind, method, seed, static_vector_dim, output_dir, output_format):
    """Write dataset-level embedding map artifacts for a normalized dataset."""
    projection, artifacts = write_dataset_embedding_map(
        dataset_path,
        output_dir,
        kind=kind,
        method=method,
        seed=seed,
        static_vector_dim=static_vector_dim,
    )
    summary = {
        "dataset_name": projection.attrs.get("dataset_name"),
        "dataset_kind": projection.attrs.get("dataset_kind"),
        "projection_method": projection.attrs.get("projection_method"),
        "requested_projection_method": projection.attrs.get("requested_projection_method"),
        "embedding_source": projection.attrs.get("embedding_source"),
        "documents": int(len(projection)),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    _echo_visualization_summary(summary, output_format)


@main.command(name="explain-pair")
@click.argument("left_path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("right_path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Normalized pair dataset to read the selected pair from.",
)
@click.option("--pair-index", default=0, show_default=True, help="Zero-based pair row for --dataset.")
@click.option("--left-id", help="Left file id for selecting a dataset pair.")
@click.option("--right-id", help="Right file id for selecting a dataset pair.")
@click.option(
    "--source",
    "source_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    help="Directory or ZIP archive used by compare results.",
)
@click.option("--left-name", help="Left relative file name inside --source.")
@click.option("--right-name", help="Right relative file name inside --source.")
@click.option(
    "--segment-mode",
    type=click.Choice(available_pair_explanation_segment_modes()),
    default="line",
    show_default=True,
    help="Granularity for local matching.",
)
@click.option("--high-threshold", default=0.85, show_default=True, help="Minimum score for high matches.")
@click.option("--medium-threshold", default=0.6, show_default=True, help="Minimum score for medium matches.")
@click.option("--low-threshold", default=0.3, show_default=True, help="Minimum score for low matches.")
@click.option("--chunk-size", default=5, show_default=True, help="Line count for chunk segment mode.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory where pair explanation artifacts will be written.",
)
@click.option("--basename", help="Artifact filename stem. Defaults to the selected pair ids.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json")),
    default="text",
    show_default=True,
)
def explain_pair(
    left_path,
    right_path,
    dataset_path,
    pair_index,
    left_id,
    right_id,
    source_path,
    left_name,
    right_name,
    segment_mode,
    high_threshold,
    medium_threshold,
    low_threshold,
    chunk_size,
    output_dir,
    basename,
    output_format,
):
    """Write pair-level similarity explanation artifacts."""
    using_files = left_path is not None or right_path is not None
    selected_modes = sum(bool(value) for value in (using_files, dataset_path, source_path))
    if selected_modes != 1:
        raise click.UsageError("Use exactly one input mode: LEFT_PATH RIGHT_PATH, --dataset, or --source.")
    if using_files and (left_path is None or right_path is None):
        raise click.UsageError("Provide both LEFT_PATH and RIGHT_PATH.")
    if dataset_path and (left_name or right_name):
        raise click.UsageError("--dataset uses --left-id/--right-id, not --left-name/--right-name.")
    if source_path and (left_id or right_id):
        raise click.UsageError("--source uses --left-name/--right-name, not --left-id/--right-id.")

    if dataset_path:
        explanation, artifacts = write_pair_dataset_explanation(
            dataset_path,
            output_dir,
            pair_index=pair_index,
            left_id=left_id,
            right_id=right_id,
            segment_mode=segment_mode,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
            low_threshold=low_threshold,
            chunk_size=chunk_size,
            basename=basename,
        )
    elif source_path:
        left_code, right_code = _source_pair_texts(source_path, left_name, right_name)
        explanation, artifacts = write_pair_explanation(
            left_code,
            right_code,
            output_dir,
            left_id=left_name,
            right_id=right_name,
            segment_mode=segment_mode,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
            low_threshold=low_threshold,
            chunk_size=chunk_size,
            basename=basename,
        )
    else:
        left_file = Path(left_path)
        right_file = Path(right_path)
        explanation, artifacts = write_pair_explanation(
            left_file.read_text(encoding="utf-8", errors="ignore"),
            right_file.read_text(encoding="utf-8", errors="ignore"),
            output_dir,
            left_id=left_file.name,
            right_id=right_file.name,
            segment_mode=segment_mode,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
            low_threshold=low_threshold,
            chunk_size=chunk_size,
            basename=basename,
        )

    _echo_pair_explanation_summary(
        _pair_explanation_summary(explanation, artifacts),
        output_format,
    )


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
    '--lexical-tokenizer',
    type=click.Choice(available_lexical_tokenizers()),
    default='raw',
    show_default=True,
    help='Token stream for Winnowing and GST. Use parser for tree-sitter leaf token types.',
)
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
    '--algorithm-path',
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help='Optional Python module/file path that defines score_pair for custom scoring.',
)
@click.option(
    '--algorithm-option',
    'algorithm_options',
    multiple=True,
    help='Custom algorithm option as name=value. JSON values are supported.',
)
@click.option(
    '--reproducibility-out',
    type=click.Path(),
    default=None,
    help='Optional path to write reproducibility metadata JSON.',
)
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
    lexical_tokenizer,
    vector_backend,
    similarity_function,
    normalize_semantic_scores,
    static_vector_dim,
    max_token_length,
    pooling_method,
    device,
    threshold,
    num,
    algorithm_path,
    algorithm_options,
    reproducibility_out,
    progress,
):
    """Bulk similarity from a ZIP archive or a directory."""
    show_progress = should_show_progress(progress, stream=sys.stderr)
    parsed_algorithm_options = _parse_algorithm_options(algorithm_options)
    if parsed_algorithm_options and not algorithm_path:
        raise click.UsageError("--algorithm-option requires --algorithm-path.")
    if algorithm_path:
        results = score_source_pairs_with_algorithm(
            source_path,
            algorithm=algorithm_path,
            preprocess_mode=preprocess_mode,
            code_language=code_language,
            algorithm_options=parsed_algorithm_options,
            threshold=threshold,
            number_results=num,
            progress=show_progress,
        )
    else:
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
            codebertscore_num_layers=(
                None if int(codebertscore_num_layers or 0) <= 0 else int(codebertscore_num_layers)
            ),
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
            lexical_tokenizer=lexical_tokenizer,
            vector_backend=vector_backend,
            similarity_function=similarity_function,
            normalize_semantic_scores=normalize_semantic_scores,
            static_vector_dim=static_vector_dim,
            max_token_length=max_token_length,
            pooling_method=pooling_method,
            device=device,
            progress=show_progress,
        )
    _write_cli_reproducibility(
        reproducibility_out,
        source_path,
        "compare",
        {
            "algorithm_path": algorithm_path,
            "algorithm_options": parsed_algorithm_options,
            "model_name": model,
            "threshold": threshold,
            "number_results": num,
            "feature_weights": feature_weights,
            "preprocess_mode": preprocess_mode,
            "code_language": code_language,
            "lexical_tokenizer": lexical_tokenizer,
        },
        results,
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
    "--reproducibility-out",
    type=click.Path(),
    default=None,
    help="Optional path to write reproducibility metadata JSON.",
)
@click.option(
    "--progress/--no-progress",
    default=None,
    help="Show progress bars on stderr. Defaults to auto for interactive terminals.",
)
def compare_suite(source_path, config_file, summary_out, details_dir, output_format, reproducibility_out, progress):
    """Run multiple similarity configurations from a JSON config file."""
    run_configs = load_run_configs(config_file)
    summary, _ = run_comparison_suite(
        source_path,
        run_configs,
        summary_out=summary_out,
        details_dir=details_dir,
        output_format=output_format,
        reproducibility_out=reproducibility_out,
        progress=should_show_progress(progress, stream=sys.stderr),
    )
    if summary.empty:
        click.echo("No runs were executed.")
        return
    click.echo(summary.to_string(index=False))


@main.command(name="evaluate-pairs")
@click.argument("dataset_spec", required=False, metavar="DATASET_SPEC")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), help="Dataset loading manifest.")
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
    "--algorithm-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional Python module/file path that defines score_pair for custom scoring.",
)
@click.option(
    "--algorithm-option",
    "algorithm_options",
    multiple=True,
    help="Custom algorithm option as name=value. JSON values are supported.",
)
@click.option(
    "--reproducibility-out",
    type=click.Path(),
    default=None,
    help="Optional path to write reproducibility metadata JSON.",
)
@click.option(
    "--feature-weight",
    "feature_weights",
    multiple=True,
    help=(
        "Normalized feature weights as name=value. "
        "Defaults to levenshtein=1.0 for offline-friendly evaluation."
    ),
)
@click.option(
    "--lexical-tokenizer",
    type=click.Choice(available_lexical_tokenizers()),
    default="raw",
    show_default=True,
    help="Token stream for Winnowing and GST. Use parser for tree-sitter leaf token types.",
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
    manifest,
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
    algorithm_path,
    algorithm_options,
    reproducibility_out,
    feature_weights,
    lexical_tokenizer,
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
    _ensure_manifest_has_no_dataset_cli_overrides(
        manifest,
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
    )
    if manifest is not None:
        resolved_dataset = load_pair_datasets_from_manifest(manifest)
    else:
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
    parsed_algorithm_options = _parse_algorithm_options(algorithm_options)
    if algorithm_path:
        scored_pairs, metrics = evaluate_pair_dataset(
            resolved_dataset,
            threshold=threshold,
            algorithm=algorithm_path,
            algorithm_options=parsed_algorithm_options,
            similarity_options={
                "preprocess_mode": preprocess_mode,
                "code_language": code_language,
            },
        )
    else:
        if parsed_algorithm_options:
            raise click.UsageError("--algorithm-option requires --algorithm-path.")
        scored_pairs, metrics = evaluate_pair_dataset(
            resolved_dataset,
            threshold=threshold,
            similarity_options={
                "feature_weights": selected_weights,
                "model_name": model,
                "preprocess_mode": preprocess_mode,
                "code_language": code_language,
                "lexical_tokenizer": lexical_tokenizer,
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
    _write_cli_reproducibility(
        reproducibility_out,
        resolved_dataset.root,
        "evaluate_pairs",
        {
            "algorithm_path": algorithm_path,
            "algorithm_options": parsed_algorithm_options,
            "threshold": threshold,
            "feature_weights": selected_weights,
            "model_name": model,
            "preprocess_mode": preprocess_mode,
            "code_language": code_language,
            "lexical_tokenizer": lexical_tokenizer,
        },
        scored_pairs,
    )
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
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), help="Dataset loading manifest.")
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
    "--algorithm-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional Python module/file path that defines score_pair for custom scoring.",
)
@click.option(
    "--algorithm-option",
    "algorithm_options",
    multiple=True,
    help="Custom algorithm option as name=value. JSON values are supported.",
)
@click.option(
    "--reproducibility-out",
    type=click.Path(),
    default=None,
    help="Optional path to write reproducibility metadata JSON.",
)
@click.option(
    "--feature-weight",
    "feature_weights",
    multiple=True,
    help=(
        "Normalized feature weights as name=value. "
        "Defaults to levenshtein=1.0 for offline-friendly evaluation."
    ),
)
@click.option(
    "--lexical-tokenizer",
    type=click.Choice(available_lexical_tokenizers()),
    default="raw",
    show_default=True,
    help="Token stream for Winnowing and GST. Use parser for tree-sitter leaf token types.",
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
    manifest,
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
    algorithm_path,
    algorithm_options,
    reproducibility_out,
    feature_weights,
    lexical_tokenizer,
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
    _ensure_manifest_has_no_dataset_cli_overrides(
        manifest,
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
    )
    if manifest is not None:
        resolved_dataset = load_retrieval_datasets_from_manifest(manifest)
    else:
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
    parsed_algorithm_options = _parse_algorithm_options(algorithm_options)
    if algorithm_path:
        scored_results, metrics = evaluate_retrieval_dataset(
            resolved_dataset,
            k=k,
            algorithm=algorithm_path,
            algorithm_options=parsed_algorithm_options,
            similarity_options={
                "preprocess_mode": preprocess_mode,
                "code_language": code_language,
            },
        )
    else:
        if parsed_algorithm_options:
            raise click.UsageError("--algorithm-option requires --algorithm-path.")
        scored_results, metrics = evaluate_retrieval_dataset(
            resolved_dataset,
            k=k,
            similarity_options={
                "feature_weights": selected_weights,
                "model_name": model,
                "preprocess_mode": preprocess_mode,
                "code_language": code_language,
                "lexical_tokenizer": lexical_tokenizer,
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
    _write_cli_reproducibility(
        reproducibility_out,
        resolved_dataset.root,
        "evaluate_retrieval",
        {
            "algorithm_path": algorithm_path,
            "algorithm_options": parsed_algorithm_options,
            "k": k,
            "feature_weights": selected_weights,
            "model_name": model,
            "preprocess_mode": preprocess_mode,
            "code_language": code_language,
            "lexical_tokenizer": lexical_tokenizer,
        },
        scored_results,
    )
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
    summary = (
        f"Elapsed: {elapsed_seconds:.4f}s | "
        f"features={feature_set} | "
        f"backend={vector_backend} | "
        f"code_metric={code_metric} | "
        f"chunking={chunking_method}"
    )
    algorithm = results.attrs.get("algorithm")
    if algorithm:
        summary = f"{summary} | algorithm={algorithm.get('algorithm_name')}"
    return summary
