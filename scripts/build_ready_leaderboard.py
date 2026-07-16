#!/usr/bin/env python3
"""Build the sampled public leaderboard displayed by the Gradio app."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from matheel.datasets import (
    available_dataset_presets,
    get_dataset_preset,
    load_code_texts,
    load_pair_datasets,
    load_retrieval_datasets,
    write_pair_dataset,
    write_retrieval_dataset,
)
from matheel.leaderboard import (
    available_leaderboard_metrics,
    run_leaderboard,
    write_leaderboard_artifacts,
)
from matheel.leaderboard_presets import available_leaderboard_algorithm_presets


DEFAULT_OUTPUT = Path("gradio_app/assets/ready_leaderboard.json")
DATASET_LANGUAGES = {
    "conplag": "java",
    "criminal_minds": "java",
    "ipca": "cpp",
    "irplag": "java",
    "soco14": "cpp",
    "student_code_similarity": "python",
}
DATASET_LICENSES = {
    "conplag": "CC-BY-4.0",
    "criminal_minds": "CC-BY-4.0",
    "ipca": "GPL-3.0",
    "irplag": "Apache-2.0",
    "soco14": "unknown",
    "student_code_similarity": "unknown",
}
PAIR_METRICS = available_leaderboard_metrics("pair")
RETRIEVAL_METRICS = available_leaderboard_metrics("retrieval")


def build_ready_leaderboard(
    output_path,
    work_dir,
    *,
    seed=7,
    pair_sample_limit=128,
    retrieval_query_limit=24,
    retrieval_corpus_limit=64,
):
    output = Path(output_path)
    workspace = Path(work_dir)
    raw_root = workspace / "raw"
    normalized_root = workspace / "normalized"
    sampled_root = workspace / "sampled"
    artifact_root = workspace / "artifacts"
    for path in (raw_root, normalized_root, sampled_root, artifact_root):
        path.mkdir(parents=True, exist_ok=True)

    dataset_configs = []
    dataset_sources = {}
    for preset_name in available_dataset_presets():
        preset = get_dataset_preset(preset_name)
        for task_family in preset["task_families"]:
            dataset_name = _dataset_task_name(preset_name, task_family, preset["task_families"])
            print(f"Preparing {dataset_name}...", flush=True)
            normalized = _load_preset_dataset(
                preset_name,
                task_family,
                raw_root=raw_root,
                normalized_root=normalized_root,
            )
            sample_destination = sampled_root / dataset_name
            metadata = _sample_metadata(
                preset_name,
                preset,
                task_family,
                seed=seed,
                pair_sample_limit=pair_sample_limit,
                retrieval_query_limit=retrieval_query_limit,
                retrieval_corpus_limit=retrieval_corpus_limit,
            )
            if task_family == "pair":
                sample_path = _sample_pair_dataset(
                    normalized,
                    sample_destination,
                    limit=pair_sample_limit,
                    seed=seed,
                    metadata=metadata,
                )
                task_defaults = {"threshold": 0.5}
            else:
                sample_path = _sample_retrieval_dataset(
                    normalized,
                    sample_destination,
                    query_limit=retrieval_query_limit,
                    corpus_limit=retrieval_corpus_limit,
                    seed=seed,
                    metadata=metadata,
                )
                task_defaults = {"k": 10}
            dataset_configs.append(
                {
                    "name": dataset_name,
                    "task": task_family,
                    "path": str(sample_path),
                    "similarity_options": {
                        "code_language": DATASET_LANGUAGES[preset_name],
                    },
                    **task_defaults,
                }
            )
            dataset_sources[dataset_name] = {
                "preset": preset_name,
                "source": preset["source"],
                "identifier": str(preset["identifier"]),
                "url": str(preset.get("url") or ""),
            }

    algorithm_names = available_leaderboard_algorithm_presets()
    algorithms = [
        {
            "preset": name,
            "similarity_options": {
                "vector_backend": "static_hash",
                "static_vector_dim": 256,
                "device": "cpu",
            },
        }
        for name in algorithm_names
    ]
    manifest = {
        "name": "Matheel Ready-made Public Leaderboard",
        "seed": int(seed),
        "pair_metrics": list(PAIR_METRICS),
        "retrieval_metrics": list(RETRIEVAL_METRICS),
        "datasets": dataset_configs,
        "algorithms": algorithms,
    }
    print(
        f"Scoring {len(dataset_configs)} dataset tasks with {len(algorithms)} algorithms...",
        flush=True,
    )
    report, _ = run_leaderboard(manifest)
    _describe_snapshot(
        report,
        dataset_sources,
        algorithm_names,
        seed=seed,
        pair_sample_limit=pair_sample_limit,
        retrieval_query_limit=retrieval_query_limit,
        retrieval_corpus_limit=retrieval_corpus_limit,
    )
    artifacts = write_leaderboard_artifacts(report, artifact_root, basename="ready_leaderboard")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifacts["json"], output)
    return output


def _load_preset_dataset(preset_name, task_family, *, raw_root, normalized_root):
    spec = {
        "preset": preset_name,
        "destination": raw_root / preset_name,
        "adapted_destination": normalized_root / f"{preset_name}_{task_family}",
    }
    if task_family == "pair":
        return load_pair_datasets(spec)
    return load_retrieval_datasets(spec)


def _dataset_task_name(preset_name, task_family, task_families):
    if len(task_families) == 1:
        return preset_name
    return f"{preset_name}_{task_family}"


def _sample_metadata(
    preset_name,
    preset,
    task_family,
    *,
    seed,
    pair_sample_limit,
    retrieval_query_limit,
    retrieval_corpus_limit,
):
    return {
        "name": _dataset_task_name(preset_name, task_family, preset["task_families"]),
        "task_type": "plagiarism",
        "license": DATASET_LICENSES[preset_name],
        "source_preset": preset_name,
        "source_url": str(preset.get("url") or ""),
        "benchmark_sample": {
            "seed": int(seed),
            "pair_limit": int(pair_sample_limit),
            "query_limit": int(retrieval_query_limit),
            "corpus_limit": int(retrieval_corpus_limit),
        },
    }


def _sample_pair_dataset(dataset, destination, *, limit, seed, metadata):
    pairs = dataset.pairs.copy()
    target = min(len(pairs), int(limit))
    selected_indices = []
    labels = sorted(pairs["label"].dropna().unique().tolist())
    per_label = max(1, target // max(1, len(labels)))
    for offset, label in enumerate(labels):
        group = pairs[pairs["label"] == label]
        count = min(len(group), per_label)
        if count:
            selected_indices.extend(
                group.sample(n=count, random_state=int(seed) + offset).index.tolist()
            )
    remaining_count = target - len(selected_indices)
    if remaining_count > 0:
        remaining = pairs.drop(index=selected_indices)
        selected_indices.extend(
            remaining.sample(n=remaining_count, random_state=int(seed) + 101).index.tolist()
        )
    sampled_pairs = pairs.loc[selected_indices].reset_index(drop=True)
    referenced_ids = set(sampled_pairs["left_id"].astype(str)) | set(
        sampled_pairs["right_id"].astype(str)
    )
    files = _sampled_file_rows(dataset, referenced_ids)
    payload = dict(metadata)
    payload["benchmark_sample"] = {
        **payload["benchmark_sample"],
        "source_pairs": int(len(pairs)),
        "sampled_pairs": int(len(sampled_pairs)),
    }
    return write_pair_dataset(
        destination,
        files=pd.DataFrame(files),
        pairs=sampled_pairs,
        metadata=payload,
    ).root


def _sample_retrieval_dataset(
    dataset,
    destination,
    *,
    query_limit,
    corpus_limit,
    seed,
    metadata,
):
    query_count = min(len(dataset.queries), int(query_limit))
    queries = dataset.queries.sample(n=query_count, random_state=int(seed)).reset_index(drop=True)
    query_ids = set(queries["query_id"].astype(str))
    qrels = dataset.qrels[dataset.qrels["query_id"].astype(str).isin(query_ids)].copy()
    required_documents = set(qrels["document_id"].astype(str))
    required_corpus = dataset.corpus[
        dataset.corpus["document_id"].astype(str).isin(required_documents)
    ]
    remaining_corpus = dataset.corpus[
        ~dataset.corpus["document_id"].astype(str).isin(required_documents)
    ]
    fill_count = min(
        max(0, int(corpus_limit) - len(required_corpus)),
        len(remaining_corpus),
    )
    if fill_count:
        remaining_corpus = remaining_corpus.sample(n=fill_count, random_state=int(seed) + 211)
    else:
        remaining_corpus = remaining_corpus.iloc[0:0]
    corpus = pd.concat([required_corpus, remaining_corpus], ignore_index=True)
    corpus_ids = set(corpus["document_id"].astype(str))
    qrels = qrels[qrels["document_id"].astype(str).isin(corpus_ids)].reset_index(drop=True)
    referenced_ids = set(queries["file_id"].astype(str)) | set(corpus["file_id"].astype(str))
    files = _sampled_file_rows(dataset, referenced_ids)
    payload = dict(metadata)
    payload["benchmark_sample"] = {
        **payload["benchmark_sample"],
        "source_queries": int(len(dataset.queries)),
        "source_documents": int(len(dataset.corpus)),
        "sampled_queries": int(len(queries)),
        "sampled_documents": int(len(corpus)),
    }
    return write_retrieval_dataset(
        destination,
        files=pd.DataFrame(files),
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        metadata=payload,
    ).root


def _sampled_file_rows(dataset, referenced_ids):
    texts = load_code_texts(dataset)
    source_files = dataset.files.set_index(dataset.files["file_id"].astype(str), drop=False)
    rows = []
    for file_id in sorted(referenced_ids):
        source = source_files.loc[file_id]
        suffix = Path(str(source["file_path"])).suffix or ".txt"
        rows.append({"file_id": file_id, "text": texts[file_id], "suffix": suffix})
    return rows


def _describe_snapshot(
    report,
    dataset_sources,
    algorithm_names,
    *,
    seed,
    pair_sample_limit,
    retrieval_query_limit,
    retrieval_corpus_limit,
):
    profile = {
        "profile": "sampled_public_snapshot",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_presets": list(available_dataset_presets()),
        "dataset_task_count": len(dataset_sources),
        "algorithm_presets": list(algorithm_names),
        "pair_sample_limit": int(pair_sample_limit),
        "retrieval_query_limit": int(retrieval_query_limit),
        "retrieval_corpus_limit": int(retrieval_corpus_limit),
        "seed": int(seed),
        "vector_backend": "static_hash",
        "static_vector_dim": 256,
    }
    report["metadata"]["benchmark_profile"] = profile
    source_by_dataset = {name: source["source"] for name, source in dataset_sources.items()}
    report["per_dataset"]["dataset_source"] = report["per_dataset"]["dataset_name"].map(
        source_by_dataset
    )
    for dataset_config in report["manifest"]["datasets"]:
        source = dataset_sources[dataset_config["name"]]
        dataset_config["spec"] = {
            "name": dataset_config["name"],
            "source": source["source"],
            "identifier": source["identifier"],
            "preset": source["preset"],
            "url": source["url"],
            "task_families": [dataset_config["task_family"]],
        }
    for card in report["cards"]["datasets"]:
        source = dataset_sources[card["name"]]
        card["source"] = source


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--work-dir",
        default=str(Path(tempfile.gettempdir()) / "matheel_ready_leaderboard"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pair-sample-limit", type=int, default=128)
    parser.add_argument("--retrieval-query-limit", type=int, default=24)
    parser.add_argument("--retrieval-corpus-limit", type=int, default=64)
    return parser.parse_args()


def main():
    args = _parse_args()
    output = build_ready_leaderboard(
        args.output,
        args.work_dir,
        seed=args.seed,
        pair_sample_limit=args.pair_sample_limit,
        retrieval_query_limit=args.retrieval_query_limit,
        retrieval_corpus_limit=args.retrieval_corpus_limit,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        f"Wrote {len(payload['aggregate'])} aggregate rows and "
        f"{len(payload['per_dataset'])} per-dataset rows to {output}."
    )


if __name__ == "__main__":
    main()
