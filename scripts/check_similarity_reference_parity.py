#!/usr/bin/env python3
"""Compare the preserved rankings and pair predictions against pinned upstream evaluators."""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path


_REFERENCE_REVISION = "ac74a62802a0dd159b3258c78a2df8ad36cdf2b9"
_TASKS = {"poj104": "Clone-detection-POJ-104", "bcb": "Clone-detection-BigCloneBench"}
_SHA256 = {
    "poj104": "9ee0dfc582007d8c90adf8a90e3d69817550a133939033d928e31f91c67ad72a",
    "bcb": "0b1c9e46cf78ecabcce51334a5bcd877e3df2474f43713058f4628f996b538e7",
}


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _evaluator(root, task):
    path = root / "Code-Code" / _TASKS[task] / "evaluator" / "evaluator.py"
    digest = _sha256(path)
    if digest != _SHA256[task]:
        raise ValueError(f"The {task} evaluator does not match the pinned source revision.")
    spec = importlib.util.spec_from_file_location(task, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def _retrieval_lookups(run, scores):
    groups, query_groups, canonical = defaultdict(list), {}, {}
    with (run / "dataset" / "queries.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query, group = row["query_id"], row["problem_id"]
            canonical[query] = query
            groups[group].append(query)
            query_groups[query] = group
    answers = {query: [member for member in groups[group] if member != query]
               for query, group in query_groups.items()}
    predictions = {query: [] for query in query_groups}
    for row in scores:
        query = row["query_id"]
        if len(predictions[query]) < len(answers[query]):
            predictions[query].append(canonical[row["document_id"]])
    return answers, predictions


def _pair_lookups(run, scores, threshold):
    answers, predictions = {}, {}
    with (run / "dataset" / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["left_id"], row["right_id"])
            if key in answers:
                raise ValueError("An ordered test pair appeared more than once.")
            answers[key] = int(row["label"])
    for row in scores:
        key = (row["left_id"], row["right_id"])
        if key in predictions or int(row["label"]) != answers.get(key):
            raise ValueError("Scores do not match one unique labeled test pair per row.")
        predictions[key] = int(float(row["similarity_score"]) >= threshold)
    if predictions.keys() != answers.keys():
        raise ValueError("Predictions do not cover the binary test partition.")
    return answers, predictions


def check_run(reference, run, task):
    evaluator, evaluator_sha256 = _evaluator(reference, task)
    report = json.loads((run / "results.json").read_text(encoding="utf-8"))
    score_path = run / "scores.csv.gz"
    score_sha256 = _sha256(score_path)
    if score_sha256 != report["artifacts"]["scores.csv.gz"]["sha256"]:
        raise ValueError("Stored predictions differ from the recorded score artifact.")
    metrics = report["results"][0]["metrics"]
    with gzip.open(score_path, "rt", encoding="utf-8", newline="") as handle:
        scores = csv.DictReader(handle)
        if task == "poj104":
            answers, predictions = _retrieval_lookups(run, scores)
            metric = "MAP@R"
            expected = round(metrics["mean_average_precision_at_r"], 4)
        else:
            answers, predictions = _pair_lookups(run, scores, metrics["threshold"])
            metric = "F1"
            expected = metrics["f1"]
    observed = float(evaluator.calculate_scores(answers, predictions)[metric])
    if not math.isclose(observed, expected, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"{metric} differs from the upstream evaluator on {task}.")
    return {"metric": metric, "matheel": expected, "upstream": observed, "evaluated": len(answers),
            "scores_sha256": score_sha256, "reference_sha256": evaluator_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Local CodeXGLUE checkout at the documented revision.")
    parser.add_argument("--poj-run", type=Path, required=True)
    parser.add_argument("--bcb-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"reference_revision": _REFERENCE_REVISION,
              "poj104": check_run(args.reference, args.poj_run, "poj104"),
              "codexglue_bcb": check_run(args.reference, args.bcb_run, "bcb")}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
