import math

import pandas as pd

from .algorithms import (
    attach_algorithm_metadata,
    prepare_algorithm_dataset,
    resolve_pair_algorithm,
    score_pair_with_algorithm,
)
from .calibration import evaluate_threshold
from .datasets import PairDataset, RetrievalDataset, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .preprocessing import preprocess_code
from .resampling import DataSplit, summarize_metric_samples
from .similarity import calculate_similarity


def score_pair_dataset(
    dataset,
    scorer=None,
    similarity_options=None,
    score_column="similarity_score",
    *,
    algorithm=None,
    algorithm_options=None,
):
    if not isinstance(dataset, PairDataset):
        dataset = load_pair_dataset(dataset)
    if scorer is not None and algorithm is not None:
        raise ValueError("Use either scorer or algorithm, not both.")
    texts = load_code_texts(dataset)
    options = dict(similarity_options or {})
    resolved_algorithm = resolve_pair_algorithm(algorithm) if algorithm is not None else None
    if resolved_algorithm is not None:
        texts = _preprocess_algorithm_texts(texts, options)
    dataset_context = (
        prepare_algorithm_dataset(
            resolved_algorithm,
            dataset,
            algorithm_options=algorithm_options,
            prepared_texts=texts,
        )
        if resolved_algorithm is not None
        else None
    )
    rows = []
    for pair in dataset.pairs.to_dict(orient="records"):
        left_id = str(pair["left_id"])
        right_id = str(pair["right_id"])
        left_text = texts[left_id]
        right_text = texts[right_id]
        if resolved_algorithm is not None:
            score = score_pair_with_algorithm(
                left_text,
                right_text,
                resolved_algorithm,
                algorithm_options=algorithm_options,
                dataset_context=dataset_context,
                row=pair,
            )
        elif scorer is None:
            score = calculate_similarity(left_text, right_text, **options)
        else:
            score = scorer(left_text, right_text, pair)
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"Pair score must be finite. Got: {score}")
        row = dict(pair)
        row[score_column] = numeric_score
        rows.append(row)
    scored = pd.DataFrame(rows)
    if resolved_algorithm is not None:
        attach_algorithm_metadata(scored, resolved_algorithm, algorithm_options=algorithm_options)
    return scored


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
    *,
    algorithm=None,
    algorithm_options=None,
):
    scored_pairs = score_pair_dataset(
        dataset,
        scorer=scorer,
        similarity_options=similarity_options,
        score_column=score_column,
        algorithm=algorithm,
        algorithm_options=algorithm_options,
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
    *,
    algorithm=None,
    algorithm_options=None,
):
    if not isinstance(dataset, RetrievalDataset):
        dataset = load_retrieval_dataset(dataset)
    if scorer is not None and algorithm is not None:
        raise ValueError("Use either scorer or algorithm, not both.")
    texts = load_code_texts(dataset)
    options = dict(similarity_options or {})
    qrels_lookup = _qrels_lookup(dataset.qrels)
    resolved_algorithm = resolve_pair_algorithm(algorithm) if algorithm is not None else None
    if resolved_algorithm is not None:
        texts = _preprocess_algorithm_texts(texts, options)
    dataset_context = (
        prepare_algorithm_dataset(
            resolved_algorithm,
            dataset,
            algorithm_options=algorithm_options,
            prepared_texts=texts,
        )
        if resolved_algorithm is not None
        else None
    )
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
            if resolved_algorithm is not None:
                score = score_pair_with_algorithm(
                    query_text,
                    document_text,
                    resolved_algorithm,
                    algorithm_options=algorithm_options,
                    dataset_context=dataset_context,
                    row=row,
                )
            elif scorer is None:
                score = calculate_similarity(query_text, document_text, **options)
            else:
                score = scorer(query_text, document_text, row)
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"Retrieval score must be finite. Got: {score}")
            row[score_column] = numeric_score
            rows.append(row)

    scored = pd.DataFrame(rows)
    if resolved_algorithm is not None:
        attach_algorithm_metadata(scored, resolved_algorithm, algorithm_options=algorithm_options)
    return scored


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
    _reject_duplicate_results(
        frame,
        query_column=query_column,
        document_column=document_column,
    )
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
    *,
    algorithm=None,
    algorithm_options=None,
):
    if not isinstance(dataset, RetrievalDataset):
        dataset = load_retrieval_dataset(dataset)
    scored_results = score_retrieval_dataset(
        dataset,
        scorer=scorer,
        similarity_options=similarity_options,
        score_column=score_column,
        algorithm=algorithm,
        algorithm_options=algorithm_options,
    )
    metrics = retrieval_ranking_metrics(
        scored_results,
        qrels=dataset.qrels,
        k=k,
        score_column=score_column,
    )
    return scored_results, metrics


def evaluate_pair_resamples(
    scored_pairs,
    splits,
    threshold=0.5,
    score_column="similarity_score",
    label_column="label",
    metric_columns=None,
    confidence=0.95,
):
    frame = scored_pairs.copy() if isinstance(scored_pairs, pd.DataFrame) else pd.DataFrame(scored_pairs)
    rows = []
    for split in _normalize_splits(splits):
        subset = _test_subset(frame, split, item_label="pair rows")
        row = _split_metadata(split)
        row.update(
            pair_classification_metrics(
                subset,
                threshold=threshold,
                score_column=score_column,
                label_column=label_column,
            )
        )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = summarize_metric_samples(
        metrics,
        metric_columns=metric_columns or ("accuracy", "precision", "recall", "f1"),
        confidence=confidence,
    )
    return metrics, summary


def evaluate_retrieval_resamples(
    scored_results,
    splits,
    qrels=None,
    k=10,
    score_column="similarity_score",
    query_column="query_id",
    document_column="document_id",
    relevance_column="relevance",
    metric_columns=None,
    confidence=0.95,
):
    frame = scored_results.copy() if isinstance(scored_results, pd.DataFrame) else pd.DataFrame(scored_results)
    if query_column not in frame.columns:
        raise ValueError(f"scored_results is missing required column: {query_column}")
    query_ids = tuple(sorted(frame[query_column].astype(str).unique().tolist()))
    qrels_frame = qrels.copy() if isinstance(qrels, pd.DataFrame) else None
    if qrels is not None and qrels_frame is None:
        qrels_frame = pd.DataFrame(qrels)

    rows = []
    for split in _normalize_splits(splits):
        selected_queries = _selected_query_ids(query_ids, split)
        subset = frame[frame[query_column].astype(str).isin(selected_queries)].copy()
        if subset.empty:
            raise ValueError(f"Split {split.name!r} selected no retrieval results.")

        split_qrels = None
        if qrels_frame is not None:
            if query_column not in qrels_frame.columns:
                raise ValueError(f"qrels is missing required column: {query_column}")
            split_qrels = qrels_frame[qrels_frame[query_column].astype(str).isin(selected_queries)].copy()

        row = _split_metadata(split)
        row.update(
            retrieval_ranking_metrics(
                subset,
                qrels=split_qrels,
                k=k,
                score_column=score_column,
                query_column=query_column,
                document_column=document_column,
                relevance_column=relevance_column,
            )
        )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = summarize_metric_samples(
        metrics,
        metric_columns=metric_columns
        or (
            "mean_average_precision",
            "mean_reciprocal_rank",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
        ),
        confidence=confidence,
    )
    return metrics, summary


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


def _preprocess_algorithm_texts(texts, similarity_options):
    preprocess_mode = (similarity_options or {}).get("preprocess_mode", "none")
    code_language = (similarity_options or {}).get("code_language")
    return {
        file_id: preprocess_code(text, mode=preprocess_mode, language=code_language)
        for file_id, text in texts.items()
    }


def _rank_query_results(frame, document_column, score_column):
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        by=[score_column, document_column],
        ascending=[False, True],
        kind="mergesort",
    )


def _reject_duplicate_results(frame, query_column, document_column):
    pairs = frame[[query_column, document_column]].astype(str)
    duplicate_mask = pairs.duplicated(subset=[query_column, document_column], keep=False)
    if not bool(duplicate_mask.any()):
        return
    duplicate_pairs = sorted(
        {
            f"{row[query_column]}->{row[document_column]}"
            for row in pairs.loc[duplicate_mask, [query_column, document_column]].to_dict(orient="records")
        }
    )
    preview = ", ".join(duplicate_pairs[:5])
    if len(duplicate_pairs) > 5:
        preview = f"{preview}, ..."
    raise ValueError(f"scored_results contains duplicate query/document result rows: {preview}")


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


def _normalize_splits(splits):
    if isinstance(splits, DataSplit):
        selected = (splits,)
    else:
        selected = tuple(splits)
    if not selected:
        raise ValueError("At least one split is required.")
    return selected


def _split_metadata(split):
    return {
        "split": str(split.name),
        "split_method": str(split.method),
        "train_count": int(len(split.train_indices)),
        "validation_count": int(len(split.validation_indices)),
        "test_count": int(len(split.test_indices)),
    }


def _test_subset(frame, split, item_label):
    indices = _checked_indices(split.test_indices, len(frame), split.name)
    if not indices:
        raise ValueError(f"Split {split.name!r} has no test {item_label}.")
    return frame.iloc[list(indices)].copy()


def _selected_query_ids(query_ids, split):
    indices = _checked_indices(split.test_indices, len(query_ids), split.name)
    if not indices:
        raise ValueError(f"Split {split.name!r} has no test queries.")
    return {query_ids[index] for index in indices}


def _checked_indices(indices, item_count, split_name):
    checked = tuple(int(index) for index in indices)
    invalid = [index for index in checked if index < 0 or index >= item_count]
    if invalid:
        raise ValueError(f"Split {split_name!r} contains out-of-range test indices: {invalid}")
    return checked
