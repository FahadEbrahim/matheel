"""Single-snippet source-authorship benchmarks with problem-disjoint evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import tempfile
import time
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from .calibration import evaluate_threshold, precision_recall_curve, roc_curve

AIGCODESET_REVISION = "b4d4e69"
AIGCODESET_BENCHMARK_ID = "Matheel-AIGCodeSet-v1"
AIGCODESET_SOURCES = {
    "human": {
        "filename": "human_selected_dataset.csv",
        "size": 2_791_316,
        "sha256": "f02f1099de41434e67f26dd0470e8cce32ac1a80966cea51c8f9a25b95991de8",
    },
    "generated": {
        "filename": "created_dataset_with_llms.csv",
        "size": 1_330_275,
        "sha256": "be858efdb3233dff32b4cf239171dd686f9fb968d26cb0ab44c668fa40b2226e",
    },
}
AUTHORSHIP_MODELS = ("class_prior", "char_tfidf_logistic")
AUTHORSHIP_SPLIT_SEED = 23
_SCHEMA_VERSION = 1
_SOURCE = "https://huggingface.co/datasets/basakdemirok/AIGCodeSet"
_COHORTS = ("scratch", "runtime_repair", "wrong_answer_repair")
_SPLITS = ("train", "validation", "test")
_SOURCE_STATUS = (
    {"accepted": "scratch", "runtime": "runtime_repair", "wrong": "wrong_answer_repair"},
    {"generate": "scratch", "runtime": "runtime_repair", "wrong": "wrong_answer_repair"},
)
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NEAR_THRESHOLD = 95
_NEAR_MIN_TOKENS = 8
_NEAR_BATCH_SIZE = 64
_TFIDF_OPTIONS = dict(analyzer="char", ngram_range=(3, 5), lowercase=False, max_features=30_000)
_LOGISTIC_OPTIONS = dict(solver="liblinear", C=1.0, max_iter=1_000)
_PREDICTION_COLUMNS = (
    "model", "sample_id", "problem_id", "split", "cohort", "generator", "label", "score", "prediction",
)
_THRESHOLD_RULE = "max_validation_recall_then_min_fpr_then_highest_threshold"


@dataclass(frozen=True)
class AuthorshipDataset:
    """Prepared source samples and the audit used to produce their splits."""

    samples: tuple[dict, ...]
    audit: dict


@dataclass(frozen=True)
class AuthorshipClassifier:
    """Fitted local classifier exposing a score for one source snippet."""

    model: str
    configuration: dict
    _prior: float
    _predictor: object = field(repr=False)

    def score(self, code: str) -> float:
        """Return an AI-origin classifier score for one snippet."""
        if not isinstance(code, str):
            raise TypeError("code must be a string.")
        return self._score_many([code])[0]

    def _score_many(self, code):
        if self._predictor is None:
            return [self._prior] * len(code)
        if len(code) == 0:
            return []
        return self._predictor.predict_proba(code)[:, 1].tolist()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_rows(path, source, expected_label, verify):
    path = Path(path)
    if verify and path.stat().st_size != source["size"]:
        raise ValueError(f"AIGCodeSet source size mismatch for {source['filename']}.")
    source_bytes = path.read_bytes()
    checksum = hashlib.sha256(source_bytes).hexdigest()
    matches = checksum == source["sha256"]
    if verify and not matches:
        raise ValueError(f"AIGCodeSet source checksum mismatch for {source['filename']}.")
    rows = []
    required = {"problem_id", "submission_id", "LLM", "status_in_folder", "code", "label"}
    with io.StringIO(source_bytes.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            missing = ", ".join(sorted(required.difference(reader.fieldnames or ())))
            raise ValueError(f"AIGCodeSet {source['filename']} is missing columns: {missing}.")
        for raw in reader:
            identifiers = [raw.get(key) for key in ("problem_id", "submission_id", "LLM")]
            if any(not isinstance(value, str) or not _ID.fullmatch(value) for value in identifiers):
                raise ValueError(f"Invalid AIGCodeSet identifier on CSV line {reader.line_num}.")
            problem_id, submission_id, generator = identifiers
            status = (raw.get("status_in_folder") or "").strip().lower()
            cohort = _SOURCE_STATUS[expected_label].get(status)
            if raw.get("label") != str(expected_label) or cohort is None:
                raise ValueError(f"Invalid AIGCodeSet label or status on CSV line {reader.line_num}.")
            if (generator.lower() == "human") != (expected_label == 0):
                raise ValueError(f"Invalid AIGCodeSet origin on CSV line {reader.line_num}.")
            code = raw.get("code") or ""
            code_hash = _digest(code)
            identity = [problem_id, submission_id, generator, status, expected_label, code_hash]
            rows.append({
                "sample_id": f"aig-{_digest(_canonical(identity))[:24]}",
                "problem_id": problem_id,
                "submission_id": submission_id,
                "generator": generator,
                "status": status,
                "cohort": cohort,
                "language": "python",
                "label": expected_label,
                "code_sha256": code_hash,
                "code": code,
            })
    if not rows:
        raise ValueError(f"AIGCodeSet source {source['filename']} contains no samples.")
    return rows, {
        "filename": source["filename"],
        "size": len(source_bytes),
        "sha256": checksum,
        "pinned_source": matches,
        "rows": len(rows),
    }


def _filter_rows(rows):
    labels_by_text = defaultdict(set)
    for row in rows:
        if row["code"].strip():
            labels_by_text[row["code_sha256"]].add(row["label"])
    conflicts = {digest for digest, labels in labels_by_text.items() if len(labels) > 1}
    retained, excluded, seen = [], [], set()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        key = (row["problem_id"], row["cohort"], row["code_sha256"])
        if not row["code"].strip():
            reason = "empty_code"
        elif row["code_sha256"] in conflicts:
            reason = "exact_origin_conflict"
        elif key in seen:
            reason = "duplicate_within_problem_and_cohort"
        else:
            seen.add(key)
            retained.append(row)
            continue
        excluded.append({
            "sample_id": row["sample_id"], "reason": reason, "cohort": row["cohort"],
            "label": row["label"],
        })
    return retained, excluded, len(conflicts)


def _token_signature(code):
    ignored = {tokenize.COMMENT, tokenize.ENCODING, tokenize.NL, tokenize.ENDMARKER}
    stream = tokenize.generate_tokens(io.StringIO(code).readline)
    try:
        return tuple(
            tokenize.tok_name[token.type] if token.type in {
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE
            } else token.string
            for token in stream if token.type not in ignored
        )
    except (tokenize.TokenError, SyntaxError, UnicodeError):
        return None


def _compile_groups(rows):
    problems = {row["problem_id"]: row["problem_id"] for row in rows}

    def component(problem):
        root = problem
        while problems[root] != root:
            root = problems[root]
        while problem != root:
            parent = problems[problem]
            problems[problem] = root
            problem = parent
        return root

    def join(problem_a, problem_b):
        first, second = sorted((component(problem_a), component(problem_b)))
        problems[second] = first

    representative = {}
    signatures, signature_labels = {}, defaultdict(set)
    signature_cache = {}
    cross_problem_exact, cross_problem_normalized = set(), set()
    unparsed = 0
    for row in rows:
        problem, digest = row["problem_id"], row["code_sha256"]
        same_text = representative.setdefault(digest, problem)
        if same_text != problem:
            join(problem, same_text)
            cross_problem_exact.add(digest)
        if digest not in signature_cache:
            signature_cache[digest] = _token_signature(row["code"])
        tokens = signature_cache[digest]
        if not tokens:
            unparsed += 1
            continue
        same_signature = signatures.setdefault(tokens, problem)
        signature_labels[tokens].add(row["label"])
        if same_signature != problem:
            join(problem, same_signature)
            cross_problem_normalized.add(tokens)

    candidates = sorted(
        (signature for signature in signatures if len(signature) >= _NEAR_MIN_TOKENS),
        key=lambda signature: (len(signature), signature),
    )
    # Indel similarity cannot reach the threshold unless lengths overlap by this ratio.
    ratio = _NEAR_THRESHOLD / (200 - _NEAR_THRESHOLD)
    lengths = np.array([len(signature) for signature in candidates])
    near_pairs = cross_problem_pairs = 0
    for start in range(0, len(candidates), _NEAR_BATCH_SIZE):
        stop = min(start + _NEAR_BATCH_SIZE, len(candidates))
        first = int(lengths.searchsorted(lengths[start] * ratio))
        scores = process.cdist(
            candidates[start:stop], candidates[first:stop],
            scorer=fuzz.ratio, score_cutoff=_NEAR_THRESHOLD, dtype=np.uint8,
        )
        row_ids, col_ids = np.nonzero(scores)
        for index, other in zip(row_ids + start, col_ids + first):
            if other >= index:
                continue
            near_pairs += 1
            left, right = signatures[candidates[index]], signatures[candidates[other]]
            if left != right:
                cross_problem_pairs += 1
                join(left, right)
    audit = {
        "normalized_token_similarity_threshold": _NEAR_THRESHOLD / 100,
        "minimum_near_duplicate_tokens": _NEAR_MIN_TOKENS,
        "cross_problem_exact_texts": len(cross_problem_exact),
        "cross_problem_equal_normalized_token_forms": len(cross_problem_normalized),
        "cross_label_equal_normalized_token_forms": sum(len(labels) > 1 for labels in signature_labels.values()),
        "distinct_token_forms": len(signatures),
        "near_duplicate_pairs_between_distinct_token_forms": near_pairs,
        "cross_problem_near_duplicate_pairs": cross_problem_pairs,
        "samples_without_token_signature": unparsed,
    }
    return {problem: component(problem) for problem in problems}, audit


def _assign_splits(rows, seed, groups):
    roots = sorted(set(groups.values()), key=lambda group: _digest(f"{seed}:{group}"))
    if len(roots) < 3:
        raise ValueError("An authorship benchmark requires at least three independent source groups.")
    held_out = max(1, len(roots) // 5)
    split_by_root = {
        root: "test" if index < held_out else "validation" if index < 2 * held_out else "train"
        for index, root in enumerate(roots)
    }
    return tuple(
        {**row, "group_id": groups[row["problem_id"]], "split": split_by_root[groups[row["problem_id"]]]}
        for row in rows
    )


def _coverage(rows):
    counts = Counter((row["split"], row["cohort"], row["label"]) for row in rows)
    return {
        split: {
            cohort: {"human": counts[split, cohort, 0], "generated": counts[split, cohort, 1]}
            for cohort in _COHORTS
        } for split in _SPLITS
    }


def _problem_splits(rows):
    return {
        row["problem_id"]: {"group_id": row["group_id"], "split": row["split"]}
        for row in rows
    }


def _dataset_fingerprint(rows):
    return _digest(_canonical([
        [
            row[key] for key in (
                "sample_id", "problem_id", "submission_id", "generator", "status", "cohort",
                "language", "label", "code_sha256", "group_id", "split"
            )
        ]
        for row in sorted(rows, key=lambda item: item["sample_id"])
    ]))


def _validate_dataset(dataset):
    if not isinstance(dataset, AuthorshipDataset):
        raise TypeError("dataset must be an AuthorshipDataset.")
    if not isinstance(dataset.audit, dict) or type(dataset.audit.get("split_seed")) is not int:
        raise ValueError("Invalid authorship dataset audit.")
    required_ids = ("sample_id", "problem_id", "submission_id", "generator", "group_id", "language")
    identifiers = set()
    split_by_field = {key: {} for key in ("problem_id", "group_id", "code_sha256")}
    label_by_checksum = {}
    for row in dataset.samples:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not _ID.fullmatch(row[key]) for key in required_ids
        ):
            raise ValueError("Invalid authorship sample identifiers.")
        split, code, label = row.get("split"), row.get("code"), row.get("label")
        status = row.get("status")
        if (
            split not in _SPLITS or type(label) is not int or label not in (0, 1)
            or not isinstance(status, str) or status not in _SOURCE_STATUS[label]
            or _SOURCE_STATUS[label][status] != row.get("cohort")
            or not isinstance(code, str) or not code.strip() or _digest(code) != row.get("code_sha256")
        ):
            raise ValueError("Invalid authorship sample split, status, label, or source checksum.")
        if row["sample_id"] in identifiers:
            raise ValueError("Duplicate authorship sample ID.")
        if label_by_checksum.setdefault(row["code_sha256"], label) != label:
            raise ValueError("Conflicting authorship labels for the same source text.")
        identifiers.add(row["sample_id"])
        for key, split_by_value in split_by_field.items():
            if split_by_value.setdefault(row[key], split) != split:
                raise ValueError(f"Authorship splits overlap on {key}.")
    coverage = _coverage(dataset.samples)
    for split, cohorts in coverage.items():
        if not all(cohorts["scratch"].values()):
            raise ValueError(f"The {split} scratch cohort requires both human and generated examples.")
    if (
        _dataset_fingerprint(dataset.samples) != dataset.audit.get("dataset_fingerprint")
        or coverage != dataset.audit.get("split_coverage")
        or _problem_splits(dataset.samples) != dataset.audit.get("problem_splits")
    ):
        raise ValueError("Authorship dataset fingerprint or split metadata mismatch.")
    return dataset


def prepare_aigcodeset(
    human_csv, generated_csv, *, seed=AUTHORSHIP_SPLIT_SEED, verify_source=False,
) -> AuthorshipDataset:
    """Import the two AIGCodeSet source tables and produce frozen problem-disjoint splits."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    human, human_source = _source_rows(human_csv, AIGCODESET_SOURCES["human"], 0, verify_source)
    generated, generated_source = _source_rows(
        generated_csv, AIGCODESET_SOURCES["generated"], 1, verify_source,
    )
    source_rows = human + generated
    rows, excluded, conflicts = _filter_rows(source_rows)
    groups, similarity = _compile_groups(rows)
    samples = _assign_splits(rows, seed, groups)
    sources = [human_source, generated_source]
    source_verified = all(source["pinned_source"] for source in sources)
    audit = {
        "source_url": _SOURCE,
        "source_revision": AIGCODESET_REVISION if source_verified else None,
        "source_verified": source_verified,
        "data_license": "CDLA-Permissive-2.0",
        "source_artifacts": sources,
        "source_counts": {
            "human": dict(sorted(Counter(row["status"] for row in human).items())),
            "generated": dict(sorted(Counter(row["status"] for row in generated).items())),
        },
        "source_problem_count": len({row["problem_id"] for row in source_rows}),
        "retained_problem_count": len(groups),
        "retained_independent_group_count": len(set(groups.values())),
        "exact_origin_conflict_texts": conflicts,
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "excluded_samples": excluded,
        "near_duplicate_audit": similarity,
        "split_seed": seed,
        "split_algorithm": "sha256_ordered_groups_60_20_20",
        "split_coverage": _coverage(samples),
        "problem_splits": _problem_splits(samples),
        "dataset_fingerprint": _dataset_fingerprint(samples),
    }
    return _validate_dataset(AuthorshipDataset(samples, audit))


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".authorship-") as temporary:
        output = Path(temporary) / "artifact"
        output.write_bytes(data)
        os.replace(output, path)
    return path


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_authorship_dataset(dataset, path):
    """Write a private normalized dataset snapshot containing source text."""
    _validate_dataset(dataset)
    payload = {"schema_version": _SCHEMA_VERSION, "samples": dataset.samples, "audit": dataset.audit}
    return _atomic_write(path, _json_bytes(payload))


def load_authorship_dataset(path):
    """Load and validate a private normalized dataset snapshot."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(payload.get("samples"), list) or not isinstance(payload.get("audit"), dict)
    ):
        raise ValueError("Invalid authorship dataset snapshot.")
    return _validate_dataset(AuthorshipDataset(tuple(payload["samples"]), payload["audit"]))


def _fit(dataset, model):
    if model not in AUTHORSHIP_MODELS:
        raise ValueError(f"Unsupported authorship model {model!r}.")
    rows = [row for row in dataset.samples if row["cohort"] == "scratch" and row["split"] == "train"]
    prior = sum(row["label"] for row in rows) / len(rows)
    if model == "class_prior":
        return AuthorshipClassifier(model, {"strategy": "positive_class_prior"}, prior, None)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as error:
        raise ImportError("Install matheel[authorship] to fit the char_tfidf_logistic baseline.") from error
    options = {**_LOGISTIC_OPTIONS, "random_state": dataset.audit["split_seed"]}
    predictor = make_pipeline(TfidfVectorizer(**_TFIDF_OPTIONS), LogisticRegression(**options))
    predictor.fit([row["code"] for row in rows], [row["label"] for row in rows])
    configuration = {
        "tfidf": {**_TFIDF_OPTIONS, "ngram_range": list(_TFIDF_OPTIONS["ngram_range"])},
        "logistic_regression": options,
        "scikit_learn": version("scikit-learn"),
        "scipy": version("scipy"),
    }
    return AuthorshipClassifier(model, configuration, prior, predictor)


def fit_authorship_classifier(dataset, *, model="char_tfidf_logistic"):
    """Fit a classifier on the scratch cohort of the training split."""
    return _fit(_validate_dataset(dataset), model)


def _prediction_rows(classifier, rows, threshold):
    scores = classifier._score_many([row["code"] for row in rows])
    return [
        {
            **{key: row[key] for key in ("sample_id", "problem_id", "split", "cohort", "generator", "label")},
            "model": classifier.model, "score": score, "prediction": int(score >= threshold),
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def _metrics(rows, threshold):
    if not rows:
        return {"count": 0, "positive_count": 0, "negative_count": 0}
    metrics = evaluate_threshold(rows, threshold=threshold)
    true_negative, false_positive, false_negative = (
        metrics[key] for key in ("true_negative", "false_positive", "false_negative")
    )
    negative_f1_denom = 2 * true_negative + false_positive + false_negative
    negative_f1 = 2 * true_negative / negative_f1_denom if negative_f1_denom else 0.0
    two_classes = metrics["positive_count"] > 0 and metrics["negative_count"] > 0
    return {
        **metrics,
        "count": len(rows),
        "false_positive_rate": false_positive / metrics["negative_count"] if metrics["negative_count"] else None,
        "macro_f1": (metrics["f1"] + negative_f1) / 2 if two_classes else None,
        "average_precision": precision_recall_curve(rows).attrs["average_precision"] if two_classes else None,
        "auroc": roc_curve(rows).attrs["auroc"] if two_classes else None,
    }


def _generator_recall(rows):
    by_generator = defaultdict(list)
    for row in rows:
        if row["label"] == 1:
            by_generator[row["generator"]].append(row["prediction"])
    return [
        {"generator": key, "count": len(values), "recall": sum(values) / len(values)}
        for key, values in sorted(by_generator.items())
    ]


def _validation_threshold(rows, target_fpr):
    curve = roc_curve(rows)
    eligible = curve.loc[curve["false_positive_rate"] <= target_fpr]
    best = eligible.sort_values(
        ["recall", "false_positive_rate", "threshold"], ascending=[False, True, False],
    ).iloc[0]
    return float(best["threshold"])


def _summarize_predictions(validation, test, threshold):
    by_cohort = {cohort: [row for row in test if row["cohort"] == cohort] for cohort in _COHORTS}
    return {
        "validation": _metrics(validation, threshold),
        "test": _metrics(by_cohort["scratch"], threshold),
        "repair_slices": {cohort: _metrics(by_cohort[cohort], threshold) for cohort in _COHORTS[1:]},
        "by_generator": {cohort: _generator_recall(rows) for cohort, rows in by_cohort.items()},
    }


def benchmark_authorship(dataset, *, models=AUTHORSHIP_MODELS, target_fpr=0.05):
    """Fit, calibrate on validation, and evaluate held-out single-snippet predictions."""
    dataset = _validate_dataset(dataset)
    models = (models,) if isinstance(models, str) else tuple(models)
    if not models or any(model not in AUTHORSHIP_MODELS for model in models) or len(set(models)) != len(models):
        raise ValueError("Choose one or more distinct authorship models.")
    target_fpr = float(target_fpr)
    if not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between zero and one.")
    validation = [
        row for row in dataset.samples if row["cohort"] == "scratch" and row["split"] == "validation"
    ]
    test = [row for row in dataset.samples if row["split"] == "test"]
    results, predictions = [], []
    implementation = {
        "authorship_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_source_sha256": hashlib.sha256(Path(__file__).with_name("calibration.py").read_bytes()).hexdigest(),
    }
    for model in models:
        start = time.perf_counter()
        classifier = _fit(dataset, model)
        # These provisional predictions carry scores. Only validation scores choose the threshold.
        validation_predictions = _prediction_rows(classifier, validation, threshold=0.0)
        threshold = _validation_threshold(validation_predictions, target_fpr)
        for row in validation_predictions:
            row["prediction"] = int(row["score"] >= threshold)
        test_predictions = _prediction_rows(classifier, test, threshold)
        results.append({
            "model": model,
            "configuration": classifier.configuration,
            "method_id": _digest(_canonical([model, classifier.configuration, implementation])),
            "threshold": threshold,
            "target_false_positive_rate": target_fpr,
            **_summarize_predictions(validation_predictions, test_predictions, threshold),
            "elapsed_seconds": time.perf_counter() - start,
        })
        predictions.extend(validation_predictions + test_predictions)
    from . import __version__

    return {
        "schema_version": _SCHEMA_VERSION,
        "benchmark_id": AIGCODESET_BENCHMARK_ID,
        "task_type": "ai_authorship",
        "dataset_fingerprint": dataset.audit["dataset_fingerprint"],
        "threshold_rule": _THRESHOLD_RULE,
        "runtime": {
            "python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
            "matheel": __version__, "numpy": version("numpy"), "pandas": version("pandas"),
            "rapidfuzz": version("rapidfuzz"),
            **implementation,
        },
        "audit": dataset.audit,
        "results": results,
        "predictions": predictions,
    }


def write_authorship_report(report, output_dir):
    """Export metrics, validation and test predictions without source text."""
    directory = Path(output_dir)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_PREDICTION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["predictions"])
    data = stream.getvalue().encode("utf-8")
    checksum = hashlib.sha256(data).hexdigest()
    # The result points to immutable score content even if another run writes to this directory.
    predictions_path = _atomic_write(directory / f"predictions-{checksum[:16]}.csv", data)
    payload = {key: value for key, value in report.items() if key != "predictions"}
    payload["prediction_artifact"] = {
        "filename": predictions_path.name, "sha256": checksum, "rows": len(report["predictions"]),
    }
    result_path = _atomic_write(directory / "results.json", _json_bytes(payload))
    return {"results": result_path, "predictions": predictions_path}


def validate_authorship_report(path):
    """Verify artifact identity and reproduce thresholds and metrics from its score table."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict) or report.get("schema_version") != _SCHEMA_VERSION
        or report.get("benchmark_id") != AIGCODESET_BENCHMARK_ID or report.get("threshold_rule") != _THRESHOLD_RULE
    ):
        raise ValueError("Unsupported authorship report schema or protocol.")
    artifact = report.get("prediction_artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str) or not _SHA256.fullmatch(artifact["sha256"]):
        raise ValueError("Invalid authorship prediction artifact.")
    filename = f"predictions-{artifact['sha256'][:16]}.csv"
    if artifact.get("filename") != filename:
        raise ValueError("Invalid authorship prediction filename.")
    data = path.with_name(filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
        raise ValueError("Authorship prediction checksum mismatch.")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if reader.fieldnames != list(_PREDICTION_COLUMNS):
        raise ValueError("Invalid authorship prediction columns.")
    by_model, seen = defaultdict(list), set()
    audit = report.get("audit")
    if (
        not isinstance(audit, dict) or not isinstance(audit.get("problem_splits"), dict)
        or not isinstance(audit.get("split_coverage"), dict)
        or audit.get("dataset_fingerprint") != report.get("dataset_fingerprint")
    ):
        raise ValueError("Invalid authorship audit.")
    for row in reader:
        try:
            record = {**row, "label": int(row["label"]), "prediction": int(row["prediction"]), "score": float(row["score"])}
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid authorship prediction row.") from None
        key = (row["model"], row["sample_id"])
        source = audit["problem_splits"].get(row["problem_id"])
        if (
            row["model"] not in AUTHORSHIP_MODELS or key in seen or row["split"] not in _SPLITS[1:]
            or not isinstance(source, dict) or source.get("split") != row["split"]
            or None in row or row["cohort"] not in _COHORTS
            or (row["split"] == "validation" and row["cohort"] != "scratch")
            or record["label"] not in (0, 1) or record["prediction"] not in (0, 1) or not 0 <= record["score"] <= 1
        ):
            raise ValueError("Invalid or duplicate authorship prediction.")
        seen.add(key)
        by_model[row["model"]].append(record)
    declared = report.get("results")
    if (
        not isinstance(declared, list) or not declared or any(not isinstance(row, dict) for row in declared)
        or len(by_model) != len(declared) or len(seen) != artifact.get("rows")
        or {row.get("model") for row in declared if isinstance(row.get("model"), str)} != by_model.keys()
    ):
        raise ValueError("Authorship prediction coverage mismatch.")
    for result in declared:
        rows = by_model[result["model"]]
        validation = [row for row in rows if row["split"] == "validation"]
        test = [row for row in rows if row["split"] == "test"]
        coverage = _coverage(rows)
        expected = audit["split_coverage"]
        if (
            coverage["test"] != expected.get("test")
            or not isinstance(expected.get("validation"), dict)
            or coverage["validation"]["scratch"] != expected["validation"].get("scratch")
        ):
            raise ValueError("Authorship prediction coverage disagrees with the audited split.")
        target_fpr = result.get("target_false_positive_rate")
        if not isinstance(target_fpr, (float, int)) or not 0 <= target_fpr <= 1:
            raise ValueError("Invalid authorship false-positive rate ceiling.")
        threshold = _validation_threshold(validation, target_fpr)
        metrics = _summarize_predictions(validation, test, threshold)
        if (
            threshold != result.get("threshold")
            or any(row["prediction"] != int(row["score"] >= threshold) for row in rows)
            or any(value != result.get(key) for key, value in metrics.items())
        ):
            raise ValueError("Authorship report disagrees with its prediction artifact.")
    return report
