"""Bound expensive retrieval scoring with explicit, auditable candidate recall.

The first stage scans the corpus using an existing scorer and retains a bounded
heap per query. It reduces expensive reranker work, not first-stage complexity.
Missing candidates are absent from the ranking, never synthesized as zero ties.
"""

from collections import Counter, defaultdict
import heapq
import math
from time import perf_counter

import pandas as pd

from .algorithms import prepare_algorithm_dataset, resolve_pair_algorithm, score_pair_with_algorithm
from .datasets import RetrievalDataset, load_code_texts, load_retrieval_dataset
from .evaluation import (
    _prepare_builtin_scoring_context,
    _preprocess_algorithm_texts,
    _qrels_lookup,
    _score_builtin_pair,
    retrieval_ranking_metrics,
)


def normalize_retrieval_options(options):
    """Validate the predeclared shortlist policy used by custom leaderboards."""
    if not isinstance(options, dict) or set(options) - {"candidate_count", "candidate_similarity_options"}:
        raise ValueError("retrieval_options accepts candidate_count and candidate_similarity_options.")
    count = _candidate_count(options.get("candidate_count", 100))
    scorer_options = options.get("candidate_similarity_options")
    if scorer_options is not None and not isinstance(scorer_options, dict):
        raise ValueError("candidate_similarity_options must be an options object.")
    return {"candidate_count": count,
            "candidate_similarity_options": dict(scorer_options or {"feature_weights": {"lcs": 1.0}})}


def _candidate_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("candidate_count must be a positive integer.")
    return value


def _load(dataset):
    return dataset if isinstance(dataset, RetrievalDataset) else load_retrieval_dataset(dataset)


def _universe(dataset, exclude_self_matches):
    queries = dataset.queries.to_dict("records")
    documents = dataset.corpus.to_dict("records")
    query_files = {str(row["query_id"]): str(row["file_id"]) for row in queries}
    document_files = {str(row["document_id"]): str(row["file_id"]) for row in documents}
    if exclude_self_matches and any(
        relevance > 0 and query_files[query] == document_files[document]
        for (query, document), relevance in _qrels_lookup(dataset.qrels).items()
    ):
        raise ValueError("Self-excluded retrieval cannot contain positive self judgments.")
    counts = Counter(document_files.values())
    full_count = sum(len(documents) - (counts[file_id] if exclude_self_matches else 0)
                     for file_id in query_files.values())
    return queries, documents, query_files, document_files, full_count


class _Stage:
    def __init__(self, dataset, texts, options, query_files, document_files,
                 scorer=None, algorithm=None, algorithm_options=None):
        if scorer is not None and algorithm is not None:
            raise ValueError("Use either scorer or algorithm, not both.")
        self.scorer = scorer
        self.algorithm = resolve_pair_algorithm(algorithm) if algorithm is not None else None
        self.options = dict(algorithm_options or {})
        self.texts = _preprocess_algorithm_texts(texts, options) if self.algorithm is not None else texts
        self.context = (prepare_algorithm_dataset(self.algorithm, dataset, algorithm_options=self.options,
                                                 prepared_texts=self.texts)
                        if self.algorithm is not None else None)
        if scorer is None and self.algorithm is None:
            self.context = _prepare_builtin_scoring_context(
                texts, options, query_file_ids=set(query_files), document_file_ids=set(document_files),
                precompute_multivector=False,
            )

    def score(self, row):
        left, right = row["query_file_id"], row["document_file_id"]
        if self.algorithm is not None:
            score = score_pair_with_algorithm(self.texts[left], self.texts[right], self.algorithm,
                                              algorithm_options=self.options, dataset_context=self.context, row=row)
        elif self.scorer is not None:
            score = self.scorer(self.texts[left], self.texts[right], row)
        else:
            score = _score_builtin_pair(self.context, left, right, left_is_query=True)
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Retrieval stages must return finite scores.")
        return score


class _Candidate:
    __slots__ = ("score", "document_id", "row")

    def __init__(self, score, row):
        self.score, self.document_id, self.row = score, row["document_id"], row

    def __lt__(self, other):
        # The heap root is the worst retained candidate: low score, then large ID.
        return self.score < other.score or (self.score == other.score and self.document_id > other.document_id)


def _select(dataset, texts, count, options, scorer, exclude_self_matches):
    queries, documents, query_files, document_files, full_count = _universe(dataset, exclude_self_matches)
    started = perf_counter()
    stage = (_Stage(dataset, texts, options, query_files.values(), document_files.values(), scorer=scorer)
             if queries and documents and full_count else None)
    selected = []
    for query in queries:
        heap = []
        for document in documents:
            if exclude_self_matches and str(query["file_id"]) == str(document["file_id"]):
                continue
            row = {"query_id": str(query["query_id"]), "document_id": str(document["document_id"]),
                   "query_file_id": str(query["file_id"]), "document_file_id": str(document["file_id"])}
            # Candidate scorers receive identifiers, never relevance labels.
            score = stage.score(row)
            candidate = _Candidate(score, row)
            if len(heap) < count:
                heapq.heappush(heap, candidate)
            elif heap[0] < candidate:
                heapq.heapreplace(heap, candidate)
        for rank, candidate in enumerate(sorted(heap, key=lambda item: (-item.score, item.document_id)), 1):
            selected.append({**candidate.row, "candidate_score": candidate.score, "candidate_rank": rank})
    frame = pd.DataFrame(selected, columns=["query_id", "document_id", "query_file_id", "document_file_id",
                                            "candidate_score", "candidate_rank"])
    frame.attrs["retrieval"] = {
        "retrieval_mode": "two_stage", "candidate_count": count,
        "candidate_similarity_options": options, "exclude_self_matches": bool(exclude_self_matches),
        "first_stage_comparisons": full_count, "first_stage_seconds": perf_counter() - started,
    }
    return frame


def select_retrieval_candidates(dataset, candidate_count=100, *, similarity_options=None,
                                scorer=None, exclude_self_matches=False):
    """Select exact top-N cheap scores per query, with ascending document-ID ties.

    Defaults to the existing LCS scorer. First-stage storage is O(Q*N), without
    building the full Q*D score table. Custom scorers accept (left, right, row).
    """
    dataset = _load(dataset)
    count = _candidate_count(candidate_count)
    options = dict(similarity_options or {"feature_weights": {"lcs": 1.0}})
    return _select(dataset, load_code_texts(dataset), count, options, scorer, exclude_self_matches)


def candidate_recall_diagnostics(dataset, candidates, *, exclude_self_matches=False):
    """Measure candidate recall against all positive qrels, including missed ones."""
    dataset = _load(dataset)
    _, _, query_files, document_files, full_count = _universe(dataset, exclude_self_matches)
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame(candidates)
    if not {"query_id", "document_id"}.issubset(frame.columns):
        raise ValueError("Candidates require query_id and document_id columns.")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["document_id"] = frame["document_id"].astype(str)
    if frame.duplicated(["query_id", "document_id"]).any():
        raise ValueError("Candidates contain duplicate query/document pairs.")
    if set(frame["query_id"]) - set(query_files) or set(frame["document_id"]) - set(document_files):
        raise ValueError("Candidates refer to unknown queries or documents.")
    selected = set(zip(frame["query_id"], frame["document_id"]))
    if exclude_self_matches and any(query_files[query] == document_files[document] for query, document in selected):
        raise ValueError("Candidates contain excluded self matches.")
    positives = {key for key, relevance in _qrels_lookup(dataset.qrels).items() if relevance > 0}
    relevant_by_query, selected_by_query = defaultdict(set), defaultdict(set)
    for query, document in positives:
        relevant_by_query[query].add(document)
    for query, document in selected:
        selected_by_query[query].add(document)
    rows = []
    total_retained = 0
    for query in sorted(query_files):
        relevant = relevant_by_query[query]
        retained = len(relevant & selected_by_query[query])
        total_retained += retained
        rows.append({"query_id": query, "positive_judgments": len(relevant), "retained_positives": retained,
                     "candidate_count": len(selected_by_query[query]),
                     "candidate_recall": retained / len(relevant) if relevant else None})
    recalls = [row["candidate_recall"] for row in rows if row["candidate_recall"] is not None]
    return {
        "candidate_recall": sum(recalls) / len(recalls) if recalls else None,
        "candidate_recall_micro": total_retained / len(positives) if positives else None,
        "positive_judgments": len(positives), "retained_positive_judgments": total_retained,
        "queries_with_positive_judgments": len(recalls),
        "queries_without_positive_judgments": len(rows) - len(recalls),
        "selected_candidate_count": len(frame), "full_comparison_count": full_count,
        "per_query": rows,
    }


def score_two_stage_retrieval_dataset(
    dataset, candidate_count=100, *, candidate_similarity_options=None, similarity_options=None,
    candidate_scorer=None, scorer=None, algorithm=None, algorithm_options=None,
    exclude_self_matches=False, score_column="similarity_score",
):
    """Rerank only selected candidates using existing built-in or custom scorers.

    Diagnostics and both stage configurations are retained in DataFrame attrs.
    This is a distinct pipeline experiment, not an exhaustive frozen-v1 run.
    """
    dataset = _load(dataset)
    count = _candidate_count(candidate_count)
    texts = load_code_texts(dataset)
    candidate_options = dict(candidate_similarity_options or {"feature_weights": {"lcs": 1.0}})
    candidates = _select(dataset, texts, count, candidate_options, candidate_scorer, exclude_self_matches)
    options = dict(similarity_options or {"feature_weights": {"levenshtein": 1.0}})
    lookup = _qrels_lookup(dataset.qrels)
    rows = []
    started = perf_counter()
    stage = (_Stage(dataset, texts, options, candidates["query_file_id"], candidates["document_file_id"],
                    scorer=scorer, algorithm=algorithm, algorithm_options=algorithm_options)
             if not candidates.empty else None)
    for candidate in candidates.to_dict("records"):
        score = stage.score(candidate)
        row = {**candidate, "relevance": float(lookup.get((candidate["query_id"], candidate["document_id"]), 0.0))}
        row[score_column] = score
        rows.append(row)
    scored = pd.DataFrame(rows, columns=[*candidates.columns, "relevance", score_column])
    scored.attrs["retrieval"] = {
        **candidates.attrs["retrieval"], "similarity_options": options,
        "reranker_comparisons": len(scored), "reranker_seconds": perf_counter() - started,
        "candidate_diagnostics": candidate_recall_diagnostics(dataset, candidates, exclude_self_matches=exclude_self_matches),
    }
    return scored


def evaluate_two_stage_retrieval_dataset(dataset, candidate_count=100, *, k=10, **options):
    """Evaluate the returned ranking against full qrels and every declared query."""
    dataset = _load(dataset)
    scored = score_two_stage_retrieval_dataset(dataset, candidate_count, **options)
    metrics = retrieval_ranking_metrics(scored, qrels=dataset.qrels, k=k,
                                       score_column=options.get("score_column", "similarity_score"),
                                       query_ids=dataset.queries["query_id"])
    diagnostics = scored.attrs["retrieval"]
    metrics.update({key: value for key, value in diagnostics.items() if key not in {
        "candidate_similarity_options", "similarity_options", "candidate_diagnostics"}})
    metrics.update({key: value for key, value in diagnostics["candidate_diagnostics"].items() if key != "per_query"})
    return scored, metrics
