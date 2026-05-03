import math

import pandas as pd

from .calibration import evaluate_threshold
from .datasets import PairDataset, RetrievalDataset, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .similarity import calculate_similarity


def score_pair_dataset(
    dataset,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    if not isinstance(dataset, PairDataset):
        dataset = load_pair_dataset(dataset)
    texts = load_code_texts(dataset)
    options = dict(similarity_options or {})
    rows = []
    for pair in dataset.pairs.to_dict(orient="records"):
        left_id = str(pair["left_id"])
        right_id = str(pair["right_id"])
        left_text = texts[left_id]
        right_text = texts[right_id]
        if scorer is None:
            score = calculate_similarity(left_text, right_text, **options)
        else:
            score = scorer(left_text, right_text, pair)
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"Pair score must be finite. Got: {score}")
        row = dict(pair)
        row[score_column] = numeric_score
        rows.append(row)
    return pd.DataFrame(rows)


def pair_classification_metrics(
    scored_pairs,
    threshold=0.5,
    score_column="similarity_score",
    label_column="label",
):
    frame = scored_pairs.copy() if isinstance(scored_pairs, pd.DataFrame) else pd.DataFrame(scored_pairs)
    for column in (score_column, label_column):
        if column not in frame.columns:
            raise ValueError(f"scored_pairs is missing required column: {column}")
    metrics = evaluate_threshold(
        frame.to_dict(orient="records"),
        threshold=threshold,
        score_key=score_column,
        label_key=label_column,
    )
    metrics["pair_count"] = int(len(frame))
    return metrics


def evaluate_pair_dataset(
    dataset,
    threshold=0.5,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    scored_pairs = score_pair_dataset(
        dataset,
        scorer=scorer,
        similarity_options=similarity_options,
        score_column=score_column,
    )
    metrics = pair_classification_metrics(
        scored_pairs,
        threshold=threshold,
        score_column=score_column,
    )
    return scored_pairs, metrics


def score_retrieval_dataset(
    dataset,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    if not isinstance(dataset, RetrievalDataset):
        dataset = load_retrieval_dataset(dataset)
    texts = load_code_texts(dataset)
    options = dict(similarity_options or {})
    qrels_lookup = _qrels_lookup(dataset.qrels)
    rows = []

    query_rows = dataset.queries.to_dict(orient="records")
    corpus_rows = dataset.corpus.to_dict(orient="records")
    for query in query_rows:
        query_id = str(query["query_id"])
        query_file_id = str(query["file_id"])
        query_text = texts[query_file_id]
        for document in corpus_rows:
            document_id = str(document["document_id"])
            document_file_id = str(document["file_id"])
            document_text = texts[document_file_id]
            row = {
                "query_id": query_id,
                "document_id": document_id,
                "query_file_id": query_file_id,
                "document_file_id": document_file_id,
                "relevance": float(qrels_lookup.get((query_id, document_id), 0.0)),
            }
            if scorer is None:
                score = calculate_similarity(query_text, document_text, **options)
            else:
                score = scorer(query_text, document_text, row)
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"Retrieval score must be finite. Got: {score}")
            row[score_column] = numeric_score
            rows.append(row)

    return pd.DataFrame(rows)


def retrieval_ranking_metrics(
    scored_results,
    qrels=None,
    k=10,
    score_column="similarity_score",
    query_column="query_id",
    document_column="document_id",
    relevance_column="relevance",
):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    frame = scored_results.copy() if isinstance(scored_results, pd.DataFrame) else pd.DataFrame(scored_results)
    for column in (query_column, document_column, score_column):
        if column not in frame.columns:
            raise ValueError(f"scored_results is missing required column: {column}")

    if qrels is None:
        if relevance_column not in frame.columns:
            raise ValueError(f"scored_results is missing required column: {relevance_column}")
        qrels_frame = frame[[query_column, document_column, relevance_column]].copy()
    else:
        qrels_frame = qrels.copy() if isinstance(qrels, pd.DataFrame) else pd.DataFrame(qrels)
        for column in (query_column, document_column, relevance_column):
            if column not in qrels_frame.columns:
                raise ValueError(f"qrels is missing required column: {column}")

    frame = frame.copy()
    frame[score_column] = frame[score_column].map(_normalize_finite_score)
    qrels_lookup = _qrels_lookup(
        qrels_frame,
        query_column=query_column,
        document_column=document_column,
        relevance_column=relevance_column,
    )
    query_ids = sorted(set(frame[query_column].astype(str).tolist()) | {query for query, _ in qrels_lookup})

    average_precisions = []
    reciprocal_ranks = []
    precision_at_k_values = []
    recall_at_k_values = []
    ndcg_at_k_values = []
    relevant_count = 0

    for query_id in query_ids:
        relevant_by_document = {
            document_id: relevance
            for (judged_query_id, document_id), relevance in qrels_lookup.items()
            if judged_query_id == query_id and relevance > 0
        }
        relevant_count += len(relevant_by_document)
        ranked = _rank_query_results(
            frame[frame[query_column].astype(str) == query_id],
            document_column=document_column,
            score_column=score_column,
        )
        ranked_documents = ranked[document_column].astype(str).tolist()

        average_precisions.append(_average_precision(ranked_documents, relevant_by_document))
        reciprocal_ranks.append(_reciprocal_rank(ranked_documents, relevant_by_document))
        precision_at_k_values.append(_precision_at_k(ranked_documents, relevant_by_document, k))
        recall_at_k_values.append(_recall_at_k(ranked_documents, relevant_by_document, k))
        ndcg_at_k_values.append(_ndcg_at_k(ranked_documents, relevant_by_document, k))

    return {
        "query_count": int(len(query_ids)),
        "result_count": int(len(frame)),
        "relevant_count": int(relevant_count),
        "k": int(k),
        "mean_average_precision": _mean(average_precisions),
        "mean_reciprocal_rank": _mean(reciprocal_ranks),
        "precision_at_k": _mean(precision_at_k_values),
        "recall_at_k": _mean(recall_at_k_values),
        "ndcg_at_k": _mean(ndcg_at_k_values),
    }


def evaluate_retrieval_dataset(
    dataset,
    k=10,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
):
    if not isinstance(dataset, RetrievalDataset):
        dataset = load_retrieval_dataset(dataset)
    scored_results = score_retrieval_dataset(
        dataset,
        scorer=scorer,
        similarity_options=similarity_options,
        score_column=score_column,
    )
    metrics = retrieval_ranking_metrics(
        scored_results,
        qrels=dataset.qrels,
        k=k,
        score_column=score_column,
    )
    return scored_results, metrics


def _qrels_lookup(
    qrels,
    query_column="query_id",
    document_column="document_id",
    relevance_column="relevance",
):
    frame = qrels.copy() if isinstance(qrels, pd.DataFrame) else pd.DataFrame(qrels)
    lookup = {}
    for row in frame.to_dict(orient="records"):
        query_id = str(row[query_column])
        document_id = str(row[document_column])
        relevance = _normalize_nonnegative_relevance(row[relevance_column])
        lookup[(query_id, document_id)] = max(lookup.get((query_id, document_id), 0.0), relevance)
    return lookup


def _rank_query_results(frame, document_column, score_column):
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        by=[score_column, document_column],
        ascending=[False, True],
        kind="mergesort",
    )


def _average_precision(ranked_documents, relevant_by_document):
    if not relevant_by_document:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(ranked_documents, start=1):
        if document_id in relevant_by_document:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_by_document)


def _reciprocal_rank(ranked_documents, relevant_by_document):
    for rank, document_id in enumerate(ranked_documents, start=1):
        if document_id in relevant_by_document:
            return 1.0 / rank
    return 0.0


def _precision_at_k(ranked_documents, relevant_by_document, k):
    top_documents = ranked_documents[:k]
    hits = sum(1 for document_id in top_documents if document_id in relevant_by_document)
    return hits / k


def _recall_at_k(ranked_documents, relevant_by_document, k):
    if not relevant_by_document:
        return 0.0
    top_documents = ranked_documents[:k]
    hits = sum(1 for document_id in top_documents if document_id in relevant_by_document)
    return hits / len(relevant_by_document)


def _ndcg_at_k(ranked_documents, relevant_by_document, k):
    ideal_relevances = sorted(relevant_by_document.values(), reverse=True)[:k]
    ideal_dcg = _discounted_cumulative_gain(ideal_relevances)
    if ideal_dcg == 0.0:
        return 0.0
    ranked_relevances = [relevant_by_document.get(document_id, 0.0) for document_id in ranked_documents[:k]]
    return _discounted_cumulative_gain(ranked_relevances) / ideal_dcg


def _discounted_cumulative_gain(relevances):
    return sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(relevances, start=1))


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _normalize_finite_score(value):
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Scores must be finite. Got: {value}")
    return numeric


def _normalize_nonnegative_relevance(value):
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"Relevance values must be finite and non-negative. Got: {value}")
    return numeric
