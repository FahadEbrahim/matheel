#!/usr/bin/env python3
"""Prepare checksum-verified local inputs for Matheel-Plagiarism-v1.

Nothing is downloaded without --download. Archives and normalized source code
belong in a user cache outside this Git repository, never in published results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from matheel.benchmark_protocol import dataset_content_fingerprint
from matheel.datasets import (
    _download_url_to_file,
    _find_soco14_split_root,
    _safe_extract_zip,
    adapt_pair_dataset,
    adapt_retrieval_dataset,
    load_code_texts,
    load_pair_dataset,
    load_retrieval_dataset,
    write_pair_dataset,
    write_retrieval_dataset,
)

DEFAULT_SOURCES = Path(__file__).resolve().parents[1] / "benchmarks" / "plagiarism-v1" / "sources.json"
_SCENARIOS = ("A1", "A2", "B1", "B2", "C1", "C2")
_PAIR_ADAPTERS = {
    "irplag": "irplag_pair", "ipca": "ipca_pair",
    "criminal_minds": "criminal_minds_pair", "student_code_similarity": "student_code_similarity_pair",
}


def _digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path, archive):
    """Fail closed on missing/truncated/changed source archives before extraction."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source archive: {path}. Supply it or rerun with --download.")
    if path.stat().st_size != int(archive["size"]):
        raise ValueError(f"Source archive size mismatch: {path.name}")
    for algorithm in ("sha256", "md5"):
        if archive.get(algorithm) and _digest_file(path, algorithm) != archive[algorithm]:
            raise ValueError(f"Source archive {algorithm} mismatch: {path.name}")
    return path


def _archive_path(cache_dir, name, archive, download):
    path = cache_dir / name / archive["filename"]
    if not path.exists() and download:
        # Download to a sibling temporary file; failed requests never become cache hits.
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=path.parent) as directory:
            pending = Path(directory) / archive["filename"]
            _download_url_to_file(archive["url"], pending, max_bytes=int(archive["size"]) + 1024, timeout=90)
            verify_archive(pending, archive)
            pending.replace(path)
    return verify_archive(path, archive)


def _metadata(name, source, **extra):
    return {
        "name": name, "task_type": "plagiarism",
        **{key: source[key] for key in ("license", "languages", "granularity", "ground_truth", "source_url", "source_revision") if key in source},
        "source_checksum": ["sha256:" + archive["sha256"] for archive in source["archives"]],
        **extra,
    }


def _with_metadata(dataset, name, source):
    metadata = {**dataset.metadata, **_metadata(name, source)}
    # Only metadata changes. Source-relative file IDs and existing file tables stay intact.
    (dataset.root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_pair_dataset(dataset.root) if hasattr(dataset, "pairs") else load_retrieval_dataset(dataset.root)


def prepare_conplag(source_root, destination, source):
    """Select raw pair directories explicitly, never BPlag's rewritten copies."""
    versions = next(iter(sorted(Path(source_root).rglob("versions"))))
    labels = pd.read_csv(versions / "labels.csv", dtype=str, keep_default_na=False)
    files, pairs, seen = [], [], {}
    for row in labels.to_dict(orient="records"):
        pair_dir = versions / "version_1" / f"{row['sub1']}_{row['sub2']}"
        if not pair_dir.is_dir():
            pair_dir = versions / "version_1" / f"{row['sub2']}_{row['sub1']}"
        ids = []
        for submission in (row["sub1"], row["sub2"]):
            path = pair_dir / f"{submission}.java"
            if not path.is_file():
                raise ValueError(f"Missing raw ConPlag submission: {path.relative_to(versions)}")
            # The archive contains differing raw copies of some submission IDs in
            # different pair directories. Keep the exact annotated pair's copy.
            key = (row["problem"], pair_dir.name, submission)
            identity = "conplag_" + hashlib.sha256(":".join(key).encode()).hexdigest()[:16]
            checksum = _digest_file(path)
            if key in seen and seen[key] != checksum:
                raise ValueError(f"ConPlag submission has conflicting raw copies: {key}")
            if key not in seen:
                seen[key] = checksum
                files.append({"file_id": identity, "source_path": path, "suffix": ".java", "language": "java", "problem_id": row["problem"]})
            ids.append(identity)
        pairs.append({"left_id": ids[0], "right_id": ids[1], "label": int(row["verdict"]), "problem": row["problem"]})
    return write_pair_dataset(destination, files, pairs, _metadata("conplag_pair", source, source_variant="raw_version_1"))


def _positive_irplag(source_root, destination, source):
    adapted = adapt_retrieval_dataset(source_root, adapter="irplag_retrieval", destination=destination, dataset_name="irplag_retrieval")
    dataset = load_retrieval_dataset(adapted)
    positive = set(dataset.qrels.loc[dataset.qrels["relevance"] > 0, "query_id"])
    queries = dataset.queries.loc[dataset.queries["query_id"].isin(positive)].copy()
    qrels = dataset.qrels.loc[dataset.qrels["query_id"].isin(positive)].copy()
    files = dataset.files.copy()
    texts = load_code_texts(dataset)
    files["text"] = files["file_id"].map(texts)
    # Keep all original files in the source inventory while the query policy is explicit.
    metadata = _metadata("irplag_retrieval", source, excluded_queries_without_positive_qrels=len(dataset.queries) - len(queries))
    return write_retrieval_dataset(destination, files, queries, dataset.corpus, qrels, metadata)


def prepare_soco14(source_root, output_dir, source):
    """Build one judged-endpoint ranking task per nonempty language/scenario."""
    root = _find_soco14_split_root(source_root, "test")
    if root is None:
        raise ValueError("SOCO14 test split is missing.")
    datasets, excluded, missing = [], [], []
    for folder, language in (("c", "cpp"), ("java", "java")):
        edges = defaultdict(set)
        qrel_path = root / f"soco14-test-{folder}-update.qrel"
        for line_number, line in enumerate(qrel_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            left, right = line.split()
            scenario = left[:2]
            if scenario not in _SCENARIOS or scenario != right[:2]:
                raise ValueError(f"Unexpected SOCO14 cross-scenario qrel: {line}")
            if left == right:
                raise ValueError(f"Unexpected SOCO14 self-relevance: {line}")
            absent = [token for token in (left, right) if not (root / folder / scenario / token).is_file()]
            if absent:
                missing.append({"language": language, "scenario": scenario, "line": line_number, "left": left, "right": right, "missing": absent})
                continue
            edges[scenario].add(tuple(sorted((left, right))))
        for scenario in _SCENARIOS:
            name = f"soco14_{language}_{scenario.lower()}_judged_retrieval"
            if not edges[scenario]:
                excluded.append({"name": name, "reason": "no valid positive qrels in this language/scenario"})
                continue
            endpoints = sorted({token for pair in edges[scenario] for token in pair})
            files = [{"file_id": token, "source_path": root / folder / scenario / token,
                      "suffix": ".cpp" if folder == "c" else ".java", "language": language,
                      "scenario": scenario} for token in endpoints]
            queries = [{"query_id": token, "file_id": token} for token in endpoints]
            corpus = [{"document_id": token, "file_id": token} for token in endpoints]
            directed = sorted((left, right) for pair in edges[scenario] for left, right in (pair, pair[::-1]))
            qrels = [{"query_id": left, "document_id": right, "relevance": 1} for left, right in directed]
            metadata = _metadata(name, source, languages=[language], scenario=scenario, split="test",
                                 candidate_pool="all valid judged endpoints within this language/scenario",
                                 qrel_direction="reciprocal (source reuse pairs are undirected)")
            dataset = write_retrieval_dataset(Path(output_dir) / name, files, queries, corpus, qrels, metadata)
            datasets.append(dataset)
    return datasets, {"missing_qrel_rows": missing, "excluded_unjudged_views": excluded}


def _entry(dataset, name, task, source, selection, language):
    return {
        "name": name, "task": task, "path": str(dataset.root.resolve()), "domain": "plagiarism",
        "source_url": source["source_url"], "source_revision": source["source_revision"],
        "source_checksum": ["sha256:" + archive["sha256"] for archive in source["archives"]],
        "split": "source_labeled_pairs" if task == "pair" else "source_judged_retrieval",
        "selection": selection, "similarity_options": {"code_language": language}, "k": 10,
    }


def prepare_benchmark(cache_dir, output_dir, *, sources_path=DEFAULT_SOURCES, download=False):
    """Verify pinned archives and materialize all explicitly selected benchmark rows."""
    cache, output = Path(cache_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    for path in (cache, output):
        if path == repository or repository in path.parents:
            raise ValueError("Benchmark caches and source-code outputs must be outside this Git repository.")
    lock = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported source lock schema.")
    output.mkdir(parents=True, exist_ok=True)
    entries, summary = [], {"benchmark_id": lock["benchmark_id"], "datasets": []}
    for name, source in lock["sources"].items():
        archives = [_archive_path(cache, name, archive, download) for archive in source["archives"]]
        with tempfile.TemporaryDirectory(prefix=f"matheel-{name}-") as temporary:
            raw = Path(temporary)
            for archive in archives:
                _safe_extract_zip(archive, raw)
            if name == "soco14":
                prepared, omissions = prepare_soco14(raw, output, source)
                summary["soco14_omissions"] = omissions
                for dataset in prepared:
                    language = dataset.metadata["languages"][0]
                    selection = {"mode": "all_valid_judged_endpoints", "query_policy": "positive_judged",
                                 "candidate_policy": "exclude_same_file", "candidate_pool": "judged_endpoints_within_language_and_scenario",
                                 "description": source["selection"], "scenario": dataset.metadata["scenario"]}
                    entries.append(_entry(dataset, dataset.metadata["name"], "retrieval", source, selection, language))
            else:
                task_name = name + "_pair"
                if name == "conplag":
                    dataset = prepare_conplag(raw, output / task_name, source)
                else:
                    adapted = adapt_pair_dataset(raw, adapter=_PAIR_ADAPTERS[name], destination=output / task_name, dataset_name=task_name)
                    dataset = _with_metadata(load_pair_dataset(adapted), task_name, source)
                entries.append(_entry(dataset, task_name, "pair", source,
                                      {"mode": "all_labeled_pairs", "description": source["selection"]}, source["languages"][0]))
                if name == "irplag":
                    retrieval = _positive_irplag(raw, output / "irplag_retrieval", source)
                    entries.append(_entry(retrieval, "irplag_retrieval", "retrieval", source,
                                          {"mode": "all_positive_judged_queries", "query_policy": "positive_judged",
                                           "candidate_policy": "all", "candidate_pool": "all_seven_original_submissions", "excluded_queries_without_positive_qrels": retrieval.metadata["excluded_queries_without_positive_qrels"],
                                           "description": source["selection"]}, "java"))
    for entry in entries:
        summary["datasets"].append({"name": entry["name"], "task": entry["task"], **dataset_content_fingerprint(entry["path"], entry["task"])})
    manifest = {
        "name": lock["benchmark_id"], "domain": "plagiarism", "seed": 7,
        "pair_metrics": ["f1", "precision", "recall", "accuracy", "auroc", "average_precision"],
        "retrieval_metrics": ["mean_average_precision", "mean_reciprocal_rank", "ndcg_at_k", "precision_at_k", "recall_at_k"],
        "split_policy": {"type": "provided_test", "group_fields": [],
                         "description": "Fixed source labels and explicit judged-pool task views; no learned threshold fitting, random sampling, or train/test claim."},
        "threshold_policy": {"mode": "fixed", "value": 0.5}, "datasets": entries,
    }
    manifest_path = output / "prepared-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "preparation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return manifest_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path, help="Archive cache outside this repository.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Normalized data destination outside this repository.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--download", action="store_true", help="Download missing pinned archives; existing archives are always verified.")
    args = parser.parse_args()
    manifest, summary = prepare_benchmark(args.cache_dir, args.output_dir, sources_path=args.sources, download=args.download)
    print(json.dumps({"manifest": str(manifest), **summary}, indent=2))


if __name__ == "__main__":
    main()
