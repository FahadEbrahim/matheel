"""Reproducible, bounded local runs for similarity and clone datasets."""

import csv
import gzip
import hashlib
import io
import json
import math
import platform
from collections import defaultdict
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from . import __version__
from .calibration import _threshold_metrics
from .datasets import adapt_pair_dataset, adapt_retrieval_dataset, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .evaluation import score_pair_dataset
from .similarity_datasets import similcode_score_slices

SIMILARITY_BENCHMARK_ADAPTERS = ("similcode_pair", "codexglue_bcb_pair", "poj104_retrieval", "xlcost_retrieval")
_PAIR_ADAPTERS = SIMILARITY_BENCHMARK_ADAPTERS[:2]
_TFIDF_OPTIONS = {
    "analyzer": "char", "ngram_range": (3, 5), "min_df": 1, "max_features": 50000,
    "lowercase": False, "norm": "l2", "sublinear_tf": True,
}


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _normalized_content_sha256(dataset, task):
    manifest_names = ("pairs.csv",) if task == "pair" else ("queries.csv", "corpus.csv", "qrels.csv")
    names = sorted(["metadata.json", "files.csv", *manifest_names, *set(dataset.files["file_path"])])
    digest = hashlib.sha256()
    for name in names:
        digest.update(json.dumps([name, _hash(dataset.root / name)], ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _gzip_csv(path):
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            yield text


def _pair_results(dataset, output, threshold, batch_size, is_similcode):
    texts = load_code_texts(dataset)
    counts = [0, 0, 0, 0]
    extra = [column for column in dataset.pairs if column not in {"left_id", "right_id", "label"}]
    names = ["method", "left_id", "right_id", "label", *extra, "similarity_score"]
    score_path = output / "scores.csv.gz"
    results = []
    with _gzip_csv(score_path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for start in range(0, len(dataset.pairs), batch_size):
            part = dataset.pairs.iloc[start:start + batch_size]
            for left, right, label, *details in part[["left_id", "right_id", "label", *extra]].itertuples(index=False, name=None):
                score = Levenshtein.normalized_similarity(texts[left], texts[right])
                # The threshold is inclusive. Retain source pair ordering and release labels.
                counts[(0 if label else 1) if score >= threshold else (3 if label else 2)] += 1
                writer.writerow(("levenshtein", left, right, label, *details, score))
        results.append({"name": "levenshtein", "configuration": {"preprocess_mode": "none", "weights": [1, 1, 1]},
                        "metrics": _threshold_metrics(threshold, *counts)})
        if is_similcode:
            other = score_pair_dataset(dataset, similarity_options={"feature_weights": {"winnowing": 1.0}})
            for row in other[names[1:]].itertuples(index=False, name=None):
                writer.writerow(("winnowing", *row))
            labels = other["label"].astype(bool)
            match = other["similarity_score"].ge(threshold)
            counts = [int(mask.sum()) for mask in (match & labels, match & ~labels, ~match & ~labels, ~match & labels)]
            results.append({"name": "winnowing", "configuration": {"preprocess_mode": "none", "kgram": 5, "window": 4},
                            "metrics": _threshold_metrics(threshold, *counts)})
    paths = [score_path]
    if is_similcode:
        lexical = pd.read_csv(score_path, dtype={"left_id": str, "right_id": str})
        slices = pd.concat([similcode_score_slices(group).assign(method=method)
                            for method, group in lexical.groupby("method")], ignore_index=True)
        slices_path = output / "slices.csv"
        slices.to_csv(slices_path, index=False)
        paths.append(slices_path)
    return results, paths


def _relevant_documents(dataset, query_rows, corpus_rows, grouped, exclude_self):
    query_lookup = {row.query_id: row for row in query_rows.itertuples(index=False)}
    positions = {row.document_id: pos for pos, row in enumerate(corpus_rows.itertuples(index=False))}
    if grouped:
        query_groups = query_rows.set_index("query_id")["problem_id"]
        corpus_groups = corpus_rows.set_index("document_id")["problem_id"]
        if set(query_lookup) != set(positions) or not query_groups.sort_index().equals(corpus_groups.sort_index()):
            raise ValueError("POJ-104 must use the same documents and problem IDs for queries and corpus.")
        groups = {group: np.asarray(indices) for group, indices in corpus_rows.groupby("problem_id").indices.items()}
        qrels = dataset.qrels
        positive_count = sum(len(indices) * (len(indices) - 1) for indices in groups.values())
        if (len(qrels) != positive_count or not qrels["relevance"].eq(1).all()
                or qrels["query_id"].eq(qrels["document_id"]).any()
                or not qrels["query_id"].map(query_groups).eq(qrels["document_id"].map(corpus_groups)).all()):
            raise ValueError("POJ-104 qrels must be complete same-problem, non-self judgments.")
        return {key: groups[row.problem_id][groups[row.problem_id] != positions[key]]
                for key, row in query_lookup.items()}
    qrels = defaultdict(list)
    document_files = corpus_rows.set_index("document_id")["file_id"].to_dict()
    for row in dataset.qrels.itertuples(index=False):
        if row.relevance <= 0:
            continue
        if exclude_self and document_files[row.document_id] == query_lookup[row.query_id].file_id:
            raise ValueError("Self-excluded retrieval cannot contain a relevant self match.")
        qrels[row.query_id].append(positions[row.document_id])
    return qrels


def _retrieval_results(dataset, output, batch_size, k):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ValueError("Exact TF-IDF benchmark runs require the Matheel benchmark extra.") from exc

    grouped = dataset.metadata.get("adapter") == "poj104_retrieval"
    exclude_self = grouped or dataset.metadata.get("exclude_self_matches") is True
    texts = load_code_texts(dataset)
    queries = dataset.queries.sort_values("query_id", ignore_index=True)
    corpus = dataset.corpus.sort_values("document_id", ignore_index=True)
    relevant = _relevant_documents(dataset, queries, corpus, grouped, exclude_self)
    document_ids = corpus["document_id"].to_numpy()
    document_files = corpus["file_id"].to_numpy()
    query_ids = queries["query_id"].to_numpy()
    query_files = queries["file_id"].to_numpy()
    vectorizer = TfidfVectorizer(**_TFIDF_OPTIONS)
    documents = vectorizer.fit_transform(texts[file_id] for file_id in document_files)
    query_vectors = vectorizer.transform(texts[file_id] for file_id in query_files)
    document_matrix = documents.T.tocsr()
    score_path = output / "scores.csv.gz"
    per_query_path = output / "queries.csv"
    totals = defaultdict(float)
    comparisons, retained, relevant_total = 0, 0, 0
    with _gzip_csv(score_path) as handle, per_query_path.open("w", encoding="utf-8", newline="") as query_handle:
        writer = csv.writer(handle, lineterminator="\n")
        query_writer = csv.writer(query_handle, lineterminator="\n")
        writer.writerow(("query_id", "document_id", "rank", "similarity_score", "relevance"))
        query_writer.writerow(("query_id", "eligible_count", "r", "first_relevant_rank", "average_precision_at_r", "recall_at_k"))
        for start in range(0, len(queries), batch_size):
            similarities = (query_vectors[start:start + batch_size] @ document_matrix).toarray()
            for offset, scores in enumerate(similarities):
                position = start + offset
                query_id = query_ids[position]
                if exclude_self:
                    scores[document_files == query_files[position]] = -np.inf
                ranking = np.argsort(-scores, kind="stable")
                ranking = ranking[np.isfinite(scores[ranking])]
                eligible = len(ranking)
                comparisons += eligible
                relevant_indices = relevant.get(query_id, ())
                r = len(relevant_indices)
                relevant_total += r
                hits = np.isin(ranking, relevant_indices)
                hit_ranks = np.flatnonzero(hits) + 1
                first_hit = int(hit_ranks[0]) if len(hit_ranks) else None
                ranks_at_r = hit_ranks[hit_ranks <= r]
                average_precision = float(np.sum(np.arange(1, len(hit_ranks) + 1) / hit_ranks) / r) if r else 0.0
                ap_at_r = float(np.sum(np.arange(1, len(ranks_at_r) + 1) / ranks_at_r) / r) if r else 0.0
                recall = float(np.count_nonzero(hits[:k]) / r) if r else 0.0
                totals["mean_average_precision"] += average_precision
                totals["mean_average_precision_at_r"] += ap_at_r
                totals["mean_reciprocal_rank"] += 1.0 / first_hit if first_hit else 0.0
                totals["recall_at_k"] += recall
                query_writer.writerow((query_id, eligible, r, first_hit, ap_at_r, recall))
                limit = min(max(k, r), eligible) if grouped else min(k, eligible)
                retained += limit
                writer.writerows((query_id, document_ids[index], rank, float(scores[index]), int(hits[rank - 1]))
                                 for rank, index in enumerate(ranking[:limit], 1))
    metrics = {key: value / len(queries) for key, value in totals.items()}
    metrics.update(k=k, query_count=len(queries), relevant_count=relevant_total,
                   candidate_count=len(corpus), scored_comparisons=comparisons, retained_score_rows=retained)
    configuration = {**_TFIDF_OPTIONS, "fit_scope": "fixed unlabeled target corpus", "batch_size": batch_size,
                     "tie_break": "ascending document ID", "exclude_self_matches": exclude_self}
    return [{"name": "character_tfidf_cosine", "configuration": configuration, "metrics": metrics}], [score_path, per_query_path]


def run_similarity_benchmark(adapter, source, output_dir, *, split="test", source_revision=None,
                             language=None, source_language="java", target_language="python", label_view="related",
                             threshold=0.5, batch_size=32, k=10):
    """Prepare an adapter and write reproducible source-free scores and metrics.

    Retrieval scans every eligible candidate in bounded query blocks. For POJ-104
    the output includes at least R ranks per query. Per-query first-hit rank is
    retained even when this rank is below the score export cutoff.
    """
    if adapter not in SIMILARITY_BENCHMARK_ADAPTERS:
        raise ValueError(f"Unsupported benchmark adapter: {adapter}.")
    for name, value in (("batch_size", batch_size), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1].")
    is_pair = adapter in _PAIR_ADAPTERS
    is_similcode = adapter == "similcode_pair"
    adapt = adapt_pair_dataset if is_pair else adapt_retrieval_dataset
    load = load_pair_dataset if is_pair else load_retrieval_dataset
    options = {"split": split}
    if source_revision is not None:
        options["source_revision"] = source_revision
    if is_similcode:
        options.update(language=language, label_view=label_view)
    elif adapter == "xlcost_retrieval":
        options.update(source_language=source_language, target_language=target_language)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = load(adapt(source, adapter=adapter, destination=output / "dataset", adapter_options=options))
    if dataset.metadata.get("adapter") != adapter:
        raise ValueError(f"Prepared dataset does not declare the {adapter!r} adapter.")
    with TemporaryDirectory(prefix=".results-", dir=output) as temp:
        staged = Path(temp)
        if is_pair:
            results, paths = _pair_results(dataset, staged, threshold, max(1024, batch_size), is_similcode)
        else:
            results, paths = _retrieval_results(dataset, staged, batch_size, k)
        task = "pair" if is_pair else "retrieval"
        package_versions = ["numpy", "pandas", "rapidfuzz"] if is_pair else ["numpy", "pandas", "scikit-learn", "scipy"]
        if dataset.metadata.get("input_format") == "parquet":
            package_versions.append("pyarrow")
        report = {
            "benchmark_schema_version": 1,
            "task_family": task,
            "dataset": dataset.metadata,
            "dataset_sha256": _normalized_content_sha256(dataset, task),
            "normalized_fingerprint_schema": "manifest-bytes-and-files-v1",
            "matheel_version": __version__,
            "runtime": {"python": platform.python_version(), **{name: version(name) for name in package_versions}},
            "implementation_sha256": {name: _hash(Path(__file__).with_name(name)) for name in (
                "similarity_benchmark.py", "similarity_datasets.py", "datasets.py", "evaluation.py", "similarity.py",
            )},
            "results": results,
            "artifacts": {path.name: {"sha256": _hash(path), "bytes": path.stat().st_size} for path in paths},
        }
        report_path = staged / "results.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for path in [*paths, report_path]:
            path.replace(output / path.name)
    return report
