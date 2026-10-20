"""Frozen, content-addressed benchmark protocols and validated result artifacts.

A protocol describes an evaluation, independently of the methods that enter it.
Local paths locate data; fingerprints and immutable provenance define its identity.
"""

import hashlib
import inspect
import json
import math
import re
from pathlib import Path, PureWindowsPath

import pandas as pd

from .datasets import dataset_task_type, load_code_texts, load_pair_dataset, load_retrieval_dataset
from .task_eligibility import _validate_dataset_exclusion_names, method_exclusion_reason

BENCHMARK_PROTOCOL_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_ID = "Matheel-Plagiarism-v1"
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_MOVING_REVISIONS = {"main", "master", "head", "latest", "develop", "development"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()



def _validate_public_metadata(value):
    """Reject machine-local locations rather than silently change identity metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_public_metadata(str(key))
            _validate_public_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_metadata(item)
    elif isinstance(value, (str, Path)):
        text = str(value).strip()
        embedded_path = re.search(r"(?<![\w:/])/(?:[^/\s]+/)+[^\s]*", text)
        if text.startswith(("file://", "~/")) or Path(text).is_absolute() or PureWindowsPath(text).is_absolute() or embedded_path:
            raise ValueError("Portable benchmark metadata must not contain local absolute paths; keep paths only in dataset path/validation_path fields.")

def method_configuration(configuration):
    """Return the effective method identity, excluding display name and local path."""
    from . import __version__
    from .similarity import calculate_similarity

    config = dict(configuration)
    config.pop("name", None)
    config.pop("_configuration_id", None)
    algorithm_path = config.pop("algorithm_path", None)
    if algorithm_path:
        path = Path(algorithm_path)
        if path.is_file():
            config["algorithm_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            from .algorithms import resolve_pair_algorithm
            resolved = resolve_pair_algorithm(algorithm_path)
            config.update(algorithm_module=resolved.module_name, algorithm_package=resolved.package_name,
                          algorithm_package_version=resolved.package_version,
                          algorithm_sha256=(resolved.source_fingerprint or {}).get("sha256"))
            if not config["algorithm_sha256"]:
                raise ValueError("Benchmark custom methods require fingerprintable source code.")
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(calculate_similarity).parameters.items()
        if parameter.default is not inspect.Parameter.empty and name not in {"progress", "progress_callback"}
    }
    defaults.update(config.get("similarity_options") or {})
    config["similarity_options"] = defaults
    config["matheel_version"] = __version__
    _validate_public_metadata(config)
    return config


def method_configuration_id(configuration):
    """Hash the effective configuration, defaults, package version, and custom code."""
    return _digest(method_configuration(configuration))


def _validate_exhaustive_methods(methods):
    if any("retrieval_options" in method for method in methods):
        raise ValueError("Two-stage retrieval requires separate pipeline experiments; this frozen protocol evaluates exhaustive candidate coverage.")


def dataset_content_fingerprint(root, task_family):
    """Hash normalized records and resolved source text, independent of root paths."""
    dataset = load_pair_dataset(root) if task_family == "pair" else load_retrieval_dataset(root)
    files = dataset.files.drop(columns=["path", "file_path", "text"], errors="ignore").copy()
    texts = load_code_texts(dataset)
    files["text"] = files["file_id"].map(texts)
    tables = {"files": files}
    if task_family == "pair":
        tables["pairs"] = dataset.pairs
    else:
        tables.update(queries=dataset.queries, corpus=dataset.corpus, qrels=dataset.qrels)
    records = {}
    for name, frame in tables.items():
        # Column order and CSV formatting have no semantic effect; record order does.
        # pandas.to_json rounds floats to ten decimal places by default. Preserve
        # native finite values so distinct graded relevance never shares a hash.
        records[name] = frame.reindex(sorted(frame.columns), axis=1).to_dict(orient="records")
    return {"sha256": _digest(records), "counts": {name: len(frame) for name, frame in tables.items()}}


def freeze_benchmark_protocol(manifest, benchmark_id=DEFAULT_BENCHMARK_ID, output_path=None, base_dir=None):
    """Freeze explicit local normalized evaluation data and immutable source provenance.

    Manifest datasets require name, task, path, source_url, source_revision, and
    split. Pair thresholds are fixed before test scoring, or fitted only on an
    explicitly supplied validation_path with disjoint source content/groups.
    """
    from .leaderboard import _credential_paths, normalize_leaderboard_manifest

    if not isinstance(manifest, dict):
        raise ValueError("Benchmark manifest must be a JSON object.")
    if _credential_paths(manifest):
        raise ValueError("Benchmark manifests must not contain credentials.")
    domain = str(manifest.get("domain") or "plagiarism").strip().lower()
    if domain != "plagiarism":
        raise ValueError("This protocol version supports the plagiarism domain.")
    supplied = manifest.get("datasets") or []
    normalized = normalize_leaderboard_manifest({**manifest, "algorithms": [{"name": "freeze", "feature_weights": {"levenshtein": 1.0}}]}, base_dir=base_dir)
    split_policy = dict(manifest.get("split_policy") or {})
    split_policy.setdefault("type", "provided_test")
    split_policy.setdefault("description", "Evaluate the frozen supplied split; never select thresholds on its labels.")
    split_policy.setdefault("group_fields", [])
    if split_policy["type"] not in {"provided_test", "group_disjoint"}:
        raise ValueError("split_policy.type must be provided_test or group_disjoint.")
    if split_policy["type"] == "group_disjoint" and not split_policy["group_fields"]:
        raise ValueError("group_disjoint requires explicit group_fields.")
    threshold_policy = dict(manifest.get("threshold_policy") or {"mode": "fixed", "value": 0.5})
    if threshold_policy.get("mode") not in {"fixed", "validation_f1"}:
        raise ValueError("threshold_policy.mode must be fixed or validation_f1.")
    if threshold_policy["mode"] == "fixed":
        value = float(threshold_policy.get("value", 0.5))
        if not math.isfinite(value):
            raise ValueError("Fixed threshold must be finite.")
        threshold_policy["value"] = value
    execution_policy = dict(manifest.get("execution_policy") or {"max_method_seconds": 1800})
    budget = float(execution_policy.get("max_method_seconds", 1800))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("execution_policy.max_method_seconds must be finite and positive.")
    execution_policy["max_method_seconds"] = budget
    entries = []
    for raw, item in zip(supplied, normalized["datasets"]):
        if item["spec"].get("source", "local") != "local" or item["spec"].get("adapter"):
            raise ValueError("Freeze requires already normalized local datasets; download/adapt first.")
        unsupported_options = set(item.get("similarity_options", {})).difference({"code_language"})
        if unsupported_options:
            raise ValueError("Official dataset similarity_options may only set code_language; model, weights, tokenizer, preprocessing, and backend belong to the method configuration.")
        revision = str(raw.get("source_revision") or "").strip()
        if not revision or revision.casefold() in _MOVING_REVISIONS:
            raise ValueError(f"Dataset {item['name']} requires an immutable source_revision (commit, versioned DOI, or checksum).")
        if not raw.get("source_url") or not raw.get("split"):
            raise ValueError(f"Dataset {item['name']} requires source_url and an explicit split.")
        if item.get("domain") != domain:
            raise ValueError("Dataset domain must match the benchmark domain.")
        root = Path(item["spec"]["identifier"]).resolve()
        loaded = load_pair_dataset(root) if item["task_family"] == "pair" else load_retrieval_dataset(root)
        if dataset_task_type(loaded) != domain:
            raise ValueError("Frozen dataset metadata domain does not match the benchmark domain.")
        entry = {
            "name": item["name"], "domain": domain, "task_family": item["task_family"],
            "path": str(root), "source_url": str(raw["source_url"]), "source_revision": revision,
            "split": str(raw["split"]), "content": dataset_content_fingerprint(root, item["task_family"]),
            "selection": raw.get("selection") or {"mode": "provided_split", "description": "All normalized rows supplied at freeze time."},
            "source_checksum": raw.get("source_checksum"),
            "similarity_options": item.get("similarity_options", {}), "k": int(raw.get("k") or 10),
        }
        if item["task_family"] == "pair":
            entry["expected_result_count"] = len(loaded.pairs)
        else:
            candidates = loaded.corpus["file_id"].value_counts()
            excluded = sum(int(candidates.get(file_id, 0)) for file_id in loaded.queries["file_id"]) if entry["selection"].get("candidate_policy") == "exclude_same_file" else 0
            entry["expected_result_count"] = len(loaded.queries) * len(loaded.corpus) - excluded
        if entry["expected_result_count"] < 1:
            raise ValueError("Official benchmark tasks require at least one evaluated pair or query/document candidate.")
        if entry["k"] < 1:
            raise ValueError("Retrieval k must be positive.")
        if item["task_family"] == "pair" and threshold_policy["mode"] == "validation_f1":
            validation = raw.get("validation_path")
            if not validation or not raw.get("validation_split"):
                raise ValueError("validation_f1 requires validation_path and validation_split for each pair dataset.")
            path = Path(validation)
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir) / path
            entry.update(validation_path=str(path.resolve()), validation_split=str(raw["validation_split"]),
                         validation_content=dataset_content_fingerprint(path, "pair"))
            _check_disjoint(entry, split_policy["group_fields"])
        entries.append(entry)
    protocol = {
        "schema_version": BENCHMARK_PROTOCOL_SCHEMA_VERSION, "benchmark_id": str(benchmark_id),
        "domain": domain, "seed": normalized.get("seed"), "datasets": entries,
        "pair_metrics": list(normalized["pair_metrics"]), "retrieval_metrics": list(normalized["retrieval_metrics"]),
        "split_policy": split_policy, "threshold_policy": threshold_policy,
        "execution_policy": execution_policy,
        "aggregation": "macro_mean_over_all_declared_datasets_per_domain_task_metric",
        "coverage_policy": "all_required_metrics_finite_on_every_declared_dataset",
        "evaluation_rules": {"pair_positive_label": 1, "pair_decision": "score >= threshold",
                             "retrieval_unjudged_relevance": 0, "retrieval_ties": "document_id_ascending",
                             "average_precision_depth": "full_ranking", "aggregation_weighting": "equal_dataset_weight"},
    }
    protocol["protocol_id"] = _digest(public_benchmark_protocol(protocol))
    validate_benchmark_protocol(protocol, verify_data=True)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return protocol


def public_benchmark_protocol(protocol):
    """Portable identity-bearing protocol metadata, with machine-local paths removed."""
    clean = {key: value for key, value in protocol.items() if key != "protocol_id"}
    clean["datasets"] = [{key: value for key, value in entry.items() if key not in {"path", "validation_path"}} for entry in protocol["datasets"]]
    _validate_public_metadata(clean)
    return clean


def load_benchmark_protocol(path):
    target = Path(path)
    protocol = json.loads(target.read_text(encoding="utf-8"))
    for entry in protocol.get("datasets", []):
        for key in ("path", "validation_path"):
            if key in entry and not Path(entry[key]).is_absolute():
                entry[key] = str((target.parent / entry[key]).resolve())
    validate_benchmark_protocol(protocol)
    return protocol


def validate_benchmark_protocol(protocol, verify_data=False):
    if not isinstance(protocol, dict) or protocol.get("schema_version") != BENCHMARK_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark protocol schema.")
    if not protocol.get("datasets") or not protocol.get("benchmark_id"):
        raise ValueError("A benchmark protocol requires an ID and datasets.")
    required = {"domain", "pair_metrics", "retrieval_metrics", "split_policy", "threshold_policy", "coverage_policy", "aggregation", "evaluation_rules", "execution_policy"}
    if not required.issubset(protocol) or protocol["domain"] != "plagiarism":
        raise ValueError("Invalid benchmark protocol definition or domain.")
    budget = float(protocol["execution_policy"].get("max_method_seconds", float("nan")))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Benchmark execution budget must be finite and positive.")
    from .leaderboard import available_leaderboard_metrics
    for task in ("pair", "retrieval"):
        metrics = protocol[f"{task}_metrics"]
        if not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics)) or not set(metrics).issubset(available_leaderboard_metrics(task)):
            raise ValueError("Invalid benchmark protocol metrics.")
    if protocol["threshold_policy"].get("mode") not in {"fixed", "validation_f1"}:
        raise ValueError("Invalid benchmark threshold policy.")
    if protocol["threshold_policy"]["mode"] == "fixed" and not math.isfinite(float(protocol["threshold_policy"].get("value", float("nan")))):
        raise ValueError("Invalid fixed benchmark threshold.")
    expected = _digest(public_benchmark_protocol(protocol))
    if protocol.get("protocol_id") != expected:
        raise ValueError("Benchmark protocol fingerprint does not match its definition.")
    names = set()
    for entry in protocol["datasets"]:
        if not isinstance(entry, dict) or not {"name", "domain", "task_family", "source_url", "source_revision", "split", "content", "selection", "similarity_options", "k", "expected_result_count"}.issubset(entry):
            raise ValueError("Incomplete frozen dataset definition.")
        count = float(entry["expected_result_count"])
        if not math.isfinite(count) or count < 1 or not count.is_integer():
            raise ValueError("Frozen expected result counts must be positive integers.")
        if set(entry["similarity_options"]).difference({"code_language"}):
            raise ValueError("Official dataset similarity_options may only set code_language.")
        if entry["task_family"] not in {"pair", "retrieval"} or entry["domain"] != protocol["domain"] or int(entry["k"]) < 1:
            raise ValueError("Invalid frozen dataset task/domain/k.")
        if entry["name"] in names or not _FINGERPRINT.fullmatch(entry.get("content", {}).get("sha256", "")):
            raise ValueError("Invalid or duplicate frozen dataset entry.")
        names.add(entry["name"])
        if verify_data and entry["task_family"] == "retrieval":
            _check_retrieval_selection(entry)
        if verify_data:
            loaded = load_pair_dataset(entry["path"]) if entry["task_family"] == "pair" else load_retrieval_dataset(entry["path"])
            if dataset_task_type(loaded) != protocol["domain"]:
                raise ValueError("Frozen dataset metadata domain changed or does not match the benchmark.")
            if dataset_content_fingerprint(entry["path"], entry["task_family"]) != entry["content"]:
                raise ValueError(f"Frozen dataset content changed: {entry['name']}")
            if "validation_path" in entry:
                if dataset_content_fingerprint(entry["validation_path"], "pair") != entry["validation_content"]:
                    raise ValueError(f"Frozen validation content changed: {entry['name']}")
                _check_disjoint(entry, protocol["split_policy"]["group_fields"])
    return protocol



def _check_retrieval_selection(entry):
    dataset = load_retrieval_dataset(entry["path"])
    selection = entry.get("selection") or {}
    candidate_policy = selection.get("candidate_policy", "all")
    if candidate_policy not in {"all", "exclude_same_file"}:
        raise ValueError("Unsupported retrieval candidate_policy.")
    queries = dict(zip(dataset.queries["query_id"], dataset.queries["file_id"]))
    documents = dict(zip(dataset.corpus["document_id"], dataset.corpus["file_id"]))
    positives = dataset.qrels[pd.to_numeric(dataset.qrels["relevance"]) > 0]
    positive_queries = set()
    for row in positives.to_dict("records"):
        if candidate_policy == "exclude_same_file" and queries[row["query_id"]] == documents[row["document_id"]]:
            raise ValueError("Positive self-qrels conflict with exclude_same_file candidate policy.")
        positive_queries.add(row["query_id"])
    if selection.get("query_policy") == "positive_judged" and set(queries) != positive_queries:
        raise ValueError("positive_judged retrieval selection contains queries without positive judgments.")

def _check_disjoint(entry, group_fields):
    test = load_pair_dataset(entry["path"])
    validation = load_pair_dataset(entry["validation_path"])
    def hashes(dataset):
        return {hashlib.sha256(str(text).encode()).hexdigest() for text in load_code_texts(dataset).values()}
    if hashes(test).intersection(hashes(validation)):
        raise ValueError("Validation and test source content overlap; use a disjoint held-out split.")
    for field in group_fields:
        if field not in test.files or field not in validation.files:
            raise ValueError(f"Split group field is missing from files.csv: {field}")
        left, right = test.files[field], validation.files[field]
        if left.isna().any() or right.isna().any() or left.astype(str).str.strip().eq("").any() or right.astype(str).str.strip().eq("").any() or set(left.astype(str)).intersection(right.astype(str)):
            raise ValueError(f"Validation and test groups overlap or are missing: {field}")


def run_benchmark_protocol(protocol, algorithms, output_dir=None, basename="benchmark", *, score_artifact_dir=None):
    """Evaluate methods on verified frozen data, with complete-coverage rankings."""
    from .calibration import calibrate_threshold
    from .evaluation import score_pair_dataset
    from .leaderboard import normalize_leaderboard_manifest, run_leaderboard, write_leaderboard_artifacts

    validate_benchmark_protocol(protocol, verify_data=True)
    manifest = {
        "name": protocol["benchmark_id"], "seed": protocol.get("seed"),
        "pair_metrics": protocol["pair_metrics"], "retrieval_metrics": protocol["retrieval_metrics"],
        "strict_coverage": True, "continue_on_error": True, "algorithms": algorithms,
        "datasets": [{"name": entry["name"], "domain": entry["domain"], "task": entry["task_family"],
                      "path": entry["path"], "similarity_options": entry["similarity_options"], "k": entry["k"],
                      "threshold": protocol["threshold_policy"].get("value", 0.5),
                      "exclude_self_matches": entry["selection"].get("candidate_policy") == "exclude_same_file"} for entry in protocol["datasets"]],
    }
    config = normalize_leaderboard_manifest(manifest)
    for method in config["algorithms"]:
        if method.get("threshold") is not None or method.get("k") is not None:
            raise ValueError("Official benchmark thresholds and retrieval k are defined by the protocol, not methods.")
    canonical_methods = config["algorithms"]
    _validate_exhaustive_methods(canonical_methods)
    identities = [method_configuration_id(method) for method in canonical_methods]
    if len(set(identities)) != len(identities):
        raise ValueError("Official method configurations must be unique; display-name aliases are not separate methods.")
    runtime_methods = []
    for method in canonical_methods:
        runtime = {**method, "similarity_options": dict(method["similarity_options"]), "_configuration_id": method_configuration_id(method)}
        if all(method_exclusion_reason(method, entry) for entry in protocol["datasets"]):
            runtime_methods.append(runtime)
            continue
        weights = method["similarity_options"].get("feature_weights")
        semantic = weights is None or float(weights.get("semantic", 0)) > 0
        from .model_routing import normalize_vector_backend_name
        backend = normalize_vector_backend_name(method["similarity_options"].get("vector_backend", "auto"))
        if not method.get("algorithm_path") and semantic and backend == "auto":
            raise ValueError("Official semantic methods require an explicit vector_backend so snapshot paths cannot change model routing.")
        if not method.get("algorithm_path") and semantic and backend != "static_hash":
            revision = str(method.get("model_revision") or "")
            model = method["similarity_options"].get("model_name")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision) or not model or Path(str(model)).exists():
                raise ValueError("Official embedding methods require a canonical model_name and immutable 40-character model_revision.")
            from huggingface_hub import snapshot_download
            try:
                snapshot = snapshot_download(repo_id=model, revision=revision, local_files_only=True)
            except OSError:
                snapshot = snapshot_download(
                    repo_id=model, revision=revision,
                    ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.tflite"],
                )
            runtime["similarity_options"]["model_name"] = snapshot
        runtime_methods.append(runtime)
    config["algorithms"] = runtime_methods
    calibrated = {}
    if protocol["threshold_policy"]["mode"] == "validation_f1":
        # Each dataset can have a distinct held-out validation distribution.
        for entry, dataset in zip(protocol["datasets"], config["datasets"]):
            if entry["task_family"] != "pair":
                continue
            for method in config["algorithms"]:
                if method_exclusion_reason(method, entry):
                    continue
                options = {**method["similarity_options"], **entry["similarity_options"]}
                scored = score_pair_dataset(load_pair_dataset(entry["validation_path"]), similarity_options=options,
                                            algorithm=method.get("algorithm_path"), algorithm_options=method.get("algorithm_options"))
                calibrated[(entry["name"], method["name"])] = calibrate_threshold(scored, score_key="similarity_score")["threshold"]
    if calibrated:
        # Evaluate each dataset/method once, then aggregate against the complete protocol.
        from .leaderboard import _aggregate_leaderboard, _rank_per_dataset
        reports = []
        for dataset in config["datasets"]:
            for method in config["algorithms"]:
                tuned = {**method, "threshold": calibrated.get((dataset["name"], method["name"]))}
                if "dataset_exclusions" in tuned:
                    # This internal child run evaluates only one declared dataset.
                    tuned["dataset_exclusions"] = {name: reason for name, reason in tuned["dataset_exclusions"].items()
                                                    if name == dataset["name"]}
                partial, _ = run_leaderboard({**config, "datasets": [dataset], "algorithms": [tuned]}, score_artifact_dir=score_artifact_dir)
                partial["per_dataset"]["algorithm_id"] = method["_configuration_id"]
                partial["per_dataset"]["selected_threshold"] = tuned["threshold"]
                reports.append(partial)
        report = reports[0]
        report["manifest"] = config
        report["metadata"]["score_artifacts"] = [item for partial in reports for item in partial["metadata"].get("score_artifacts", [])]
        report["per_dataset"] = _rank_per_dataset(pd.concat([item["per_dataset"] for item in reports], ignore_index=True))
        report["aggregate"] = _aggregate_leaderboard(report["per_dataset"], config=config)
        from .benchmark_cards import leaderboard_cards
        report["cards"] = leaderboard_cards(config)
    else:
        report, _ = run_leaderboard(config, score_artifact_dir=score_artifact_dir)
    config["algorithms"] = canonical_methods
    report["manifest"] = config
    from .benchmark_cards import leaderboard_cards
    report["cards"] = leaderboard_cards(config)
    for method in canonical_methods:
        selected = report["per_dataset"]["algorithm_name"] == method["name"]
        report["per_dataset"].loc[selected, "algorithm_id"] = method_configuration_id(method)
    from .leaderboard import _aggregate_leaderboard
    report["aggregate"] = _aggregate_leaderboard(report["per_dataset"], config=config)
    public = {**public_benchmark_protocol(protocol), "protocol_id": protocol["protocol_id"]}
    report["metadata"].update(benchmark_id=protocol["benchmark_id"], protocol_id=protocol["protocol_id"],
                              domain=protocol["domain"], result_status="protocol_evaluation", benchmark_protocol=public)
    from .reproducibility import collect_reproducibility_snapshot
    report["metadata"]["reproducibility"] = collect_reproducibility_snapshot(run_configs=[])
    report["metadata"]["method_configurations"] = {method_configuration_id(method): method_configuration(method) for method in canonical_methods}
    report["metadata"]["thresholds"] = [{"dataset_name": dataset, "algorithm_name": method, "threshold": value} for (dataset, method), value in calibrated.items()]
    for entry in protocol["datasets"]:
        selected = report["per_dataset"]["dataset_name"] == entry["name"]
        report["per_dataset"].loc[selected, "dataset_fingerprint"] = entry["content"]["sha256"]
    for frame in (report["per_dataset"], report["aggregate"]):
        frame["benchmark_id"] = protocol["benchmark_id"]
        frame["protocol_id"] = protocol["protocol_id"]
    validate_benchmark_result(report)
    artifacts = write_leaderboard_artifacts(report, output_dir, basename=basename) if output_dir is not None else None
    return report, artifacts



def failed_benchmark_result(protocol, algorithms, error="TimeoutError", elapsed_seconds=0, output_dir=None, basename="benchmark_failed"):
    """Build honest null-score rows when a supervised method cannot finish.

    elapsed_seconds is total supervised method wall time, recorded in execution
    failures. Uncompleted dataset scoring has no measured duration and stays zero.
    """
    from .benchmark_cards import leaderboard_cards
    from .leaderboard import _aggregate_leaderboard, _leaderboard_row, _rank_per_dataset, normalize_leaderboard_manifest, write_leaderboard_artifacts
    from .reproducibility import collect_reproducibility_snapshot

    validate_benchmark_protocol(protocol)
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Failed method elapsed time must be finite and nonnegative.")
    config = normalize_leaderboard_manifest({
        "name": protocol["benchmark_id"], "seed": protocol.get("seed"), "strict_coverage": True,
        "pair_metrics": protocol["pair_metrics"], "retrieval_metrics": protocol["retrieval_metrics"],
        "algorithms": algorithms,
        "datasets": [{"name": entry["name"], "domain": entry["domain"], "task": entry["task_family"],
                      "path": entry.get("path", entry["name"]), "similarity_options": entry["similarity_options"]} for entry in protocol["datasets"]],
    })
    _validate_exhaustive_methods(config["algorithms"])
    rows = []
    for entry, dataset in zip(protocol["datasets"], config["datasets"]):
        for method in config["algorithms"]:
            for metric in protocol["pair_metrics" if entry["task_family"] == "pair" else "retrieval_metrics"]:
                row = _leaderboard_row(entry["task_family"], dataset, method, metric, None, 0)
                reason = method_exclusion_reason(method, entry)
                row.update(elapsed_seconds=0.0, dataset_fingerprint=entry["content"]["sha256"])
                if reason:
                    row.update(status="not_applicable", not_applicable_reason=reason)
                else:
                    row.update(error=str(error), status="failed")
                rows.append(row)
    per_dataset = _rank_per_dataset(pd.DataFrame(rows))
    aggregate = _aggregate_leaderboard(per_dataset, config=config)
    for frame in (per_dataset, aggregate):
        frame["benchmark_id"] = protocol["benchmark_id"]
        frame["protocol_id"] = protocol["protocol_id"]
    report = {
        "metadata": {"schema_version": 1, "name": protocol["benchmark_id"], "benchmark_id": protocol["benchmark_id"],
                     "protocol_id": protocol["protocol_id"], "domain": protocol["domain"], "strict_coverage": True,
                     "result_status": "failed", "benchmark_protocol": {**public_benchmark_protocol(protocol), "protocol_id": protocol["protocol_id"]},
                     "method_configurations": {method_configuration_id(method): method_configuration(method) for method in config["algorithms"]},
                     "execution_failures": [{"algorithm_id": method_configuration_id(method), "error": str(error), "elapsed_seconds": elapsed} for method in config["algorithms"]],
                     "reproducibility": collect_reproducibility_snapshot(run_configs=[])},
        "manifest": config, "cards": leaderboard_cards(config), "per_dataset": per_dataset, "aggregate": aggregate,
    }
    validate_benchmark_result(report)
    artifacts = write_leaderboard_artifacts(report, output_dir, basename=basename) if output_dir is not None else None
    return report, artifacts


def merge_benchmark_results(protocol, results, output_dir=None, basename="benchmark"):
    """Combine checkpointed method results from one protocol and recompute ranks."""
    from .leaderboard import _aggregate_leaderboard, _rank_per_dataset, leaderboard_payload, write_leaderboard_artifacts

    validate_benchmark_protocol(protocol)
    payloads = []
    for result in results:
        payload = leaderboard_payload(result) if isinstance(result.get("aggregate"), pd.DataFrame) else result
        validate_benchmark_result(payload, protocol=protocol)
        payloads.append(payload)
    if not payloads:
        raise ValueError("At least one benchmark result is required.")
    methods, configurations, names = [], {}, set()
    for payload in payloads:
        for identity, configuration in payload["metadata"]["method_configurations"].items():
            if identity in configurations:
                raise ValueError("Cannot merge duplicate method configurations.")
            configurations[identity] = configuration
        for method in payload.get("manifest", {}).get("algorithms", []):
            name = str(method["name"]).strip().casefold()
            if name in names:
                raise ValueError("Merged method display names must be unique.")
            names.add(name)
            methods.append(method)
    manifest = {**payloads[0].get("manifest", {}), "algorithms": methods, "strict_coverage": True}
    per_dataset = _rank_per_dataset(pd.DataFrame([row for payload in payloads for row in payload["per_dataset"]]))
    aggregate = _aggregate_leaderboard(per_dataset, config={"strict_coverage": True, "datasets": protocol["datasets"]})
    for frame in (per_dataset, aggregate):
        frame["benchmark_id"] = protocol["benchmark_id"]
        frame["protocol_id"] = protocol["protocol_id"]
    metadata = {**payloads[0]["metadata"], "result_status": "protocol_evaluation", "method_configurations": configurations}
    for field in ("thresholds", "execution_failures", "score_artifacts"):
        metadata[field] = [item for payload in payloads for item in payload["metadata"].get(field, [])]
    metadata["execution_environments"] = [payload["metadata"].get("reproducibility", {}) for payload in payloads]
    metadata["supervised_executions"] = []
    for payload in payloads:
        previous = payload["metadata"].get("supervised_executions")
        if previous is not None:
            metadata["supervised_executions"].extend(previous)
        elif payload["metadata"].get("supervised_execution"):
            metadata["supervised_executions"].extend(
                {"algorithm_id": identity, **payload["metadata"]["supervised_execution"]}
                for identity in payload["metadata"]["method_configurations"]
            )
    metadata.pop("supervised_execution", None)
    report = {
        "metadata": metadata, "manifest": manifest,
        "cards": {"datasets": payloads[0].get("cards", {}).get("datasets", []),
                  "algorithms": [card for payload in payloads for card in payload.get("cards", {}).get("algorithms", [])]},
        "per_dataset": per_dataset, "aggregate": aggregate,
    }
    validate_benchmark_result(report)
    artifacts = write_leaderboard_artifacts(report, output_dir, basename=basename) if output_dir is not None else None
    return report, artifacts

def _optional_result_text(value, field):
    # Nullable string columns may turn JSON null into NaN/NA during validation.
    if value is None or pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Result {field} must be a string or null.")
    return value


def validate_benchmark_result(report_or_payload, protocol=None, *, score_artifact_dir=None):
    """Validate imported protocol results; this verifies consistency, not honest execution."""
    from .leaderboard import _aggregate_leaderboard, leaderboard_payload

    payload = leaderboard_payload(report_or_payload) if isinstance(report_or_payload.get("aggregate"), pd.DataFrame) else report_or_payload
    metadata = payload.get("metadata", {})
    frozen = protocol or metadata.get("benchmark_protocol")
    if not frozen:
        raise ValueError("Result is missing its frozen benchmark protocol.")
    validate_benchmark_protocol(frozen)
    if metadata.get("protocol_id") != frozen["protocol_id"]:
        raise ValueError("Result protocol identity does not match the frozen protocol.")
    rows = pd.DataFrame(payload.get("per_dataset") or [])
    if rows.empty:
        raise ValueError("A benchmark result requires per-dataset scores.")
    for frame in (rows, pd.DataFrame(payload.get("aggregate") or [])):
        if not {"benchmark_id", "protocol_id"}.issubset(frame.columns) or not frame["benchmark_id"].eq(frozen["benchmark_id"]).all() or not frame["protocol_id"].eq(frozen["protocol_id"]).all():
            raise ValueError("Result rows must retain their benchmark and protocol identity.")
    required = {"algorithm_id", "algorithm_name", "dataset_name", "dataset_fingerprint", "domain", "task_family", "metric", "score", "elapsed_seconds", "sample_count", "eligible", "rank", "task_type"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Benchmark result is missing columns: {', '.join(sorted(required - set(rows.columns)))}")
    configurations = metadata.get("method_configurations") or {}
    if set(configurations) != set(rows["algorithm_id"]):
        raise ValueError("Result method identities do not match the recorded full configurations.")
    for configuration in configurations.values():
        _validate_public_metadata(configuration)
    _validate_exhaustive_methods(configurations.values())
    _validate_dataset_exclusion_names(configurations.values(), frozen["datasets"])
    if any(_digest(config) != identity for identity, config in configurations.items()):
        raise ValueError("Result method configuration fingerprint does not match its definition.")
    entries = {entry["name"]: entry for entry in frozen["datasets"]}
    for artifact in metadata.get("score_artifacts", []):
        filename = str(artifact.get("filename", ""))
        if Path(filename).name != filename or PureWindowsPath(filename).name != filename or not filename.endswith(".csv") or not _FINGERPRINT.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError("Invalid score artifact filename or fingerprint.")
        if artifact.get("algorithm_id") not in configurations or artifact.get("dataset_name") not in entries:
            raise ValueError("Score artifact refers to an unknown dataset or method.")
        if score_artifact_dir is not None:
            path = Path(score_artifact_dir) / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Score artifact content fingerprint does not match.")
            frame = pd.read_csv(path)
            if len(frame) != artifact["rows"] or list(frame.columns) != artifact["columns"]:
                raise ValueError("Score artifact shape does not match its index.")
    for row in rows.to_dict("records"):
        entry = entries.get(row["dataset_name"])
        error = row.get("error")
        if error is not None and not isinstance(error, str) and not (isinstance(error, float) and math.isnan(error)):
            raise ValueError("Result errors must be strings or null.")
        has_error = isinstance(error, str) and bool(error)
        if entry is None or row["domain"] != entry["domain"] or row["task_family"] != entry["task_family"] or row["dataset_fingerprint"] != entry["content"]["sha256"]:
            raise ValueError("Result dataset coverage or fingerprint does not match the protocol.")
        metrics = frozen["pair_metrics" if entry["task_family"] == "pair" else "retrieval_metrics"]
        if row["metric"] not in metrics or not _FINGERPRINT.fullmatch(str(row["algorithm_id"])):
            raise ValueError("Result has an invalid metric or method configuration identity.")
        if row["task_type"] != row["domain"]:
            raise ValueError("Result dataset category must match its domain.")
        reason = method_exclusion_reason(configurations[row["algorithm_id"]], entry)
        status = _optional_result_text(row.get("status"), "status")
        reported_reason = _optional_result_text(row.get("not_applicable_reason"), "not_applicable_reason")
        if reason:
            if (status != "not_applicable" or reported_reason != reason
                    or row["score"] is not None and not pd.isna(row["score"])
                    or row["sample_count"] != 0 or row["elapsed_seconds"] != 0 or has_error):
                raise ValueError("Predeclared exclusions require unscored not_applicable rows with the original reason.")
        elif status == "not_applicable" or reported_reason:
            raise ValueError("Not-applicable results require a predeclared method exclusion.")
        elif status not in {None, "completed", "failed"}:
            raise ValueError("Invalid benchmark result status.")
        elif status == "failed" and (not has_error or row["score"] is not None and not pd.isna(row["score"])):
            raise ValueError("Failed rows require an error and a null score.")
        elif status == "completed" and has_error:
            raise ValueError("Completed rows cannot contain a runtime error.")
        count = float(row["sample_count"])
        if isinstance(row["sample_count"], bool) or not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError("Result sample counts must be nonnegative integers.")
        score = row["score"]
        if score is not None and not pd.isna(score):
            if isinstance(score, (str, bool)):
                raise ValueError("Result metric scores must be numeric or null.")
            if math.isfinite(float(score)) and not 0 <= float(score) <= 1:
                raise ValueError("Result metric scores must be between zero and one.")
            if math.isfinite(float(score)) and count != entry["expected_result_count"]:
                raise ValueError("Finite metric scores require complete frozen pair/candidate coverage.")
        if count and entry["task_family"] == "pair" and count != entry["content"]["counts"]["pairs"]:
            raise ValueError("Result pair sample count does not match the frozen dataset.")
        if entry["task_family"] == "retrieval" and count > entry["content"]["counts"]["queries"] * entry["content"]["counts"]["corpus"]:
            raise ValueError("Result retrieval sample count exceeds the frozen query/corpus universe.")
        if not math.isfinite(float(row["elapsed_seconds"])) or float(row["elapsed_seconds"]) < 0:
            raise ValueError("Result runtime must be finite and nonnegative.")
    for method_id in rows["algorithm_id"].unique():
        found = set(zip(rows.loc[rows["algorithm_id"] == method_id, "dataset_name"], rows.loc[rows["algorithm_id"] == method_id, "metric"]))
        required_pairs = {(entry["name"], metric) for entry in frozen["datasets"] for metric in frozen["pair_metrics" if entry["task_family"] == "pair" else "retrieval_metrics"]}
        if found != required_pairs:
            raise ValueError("Result is missing required method/dataset/metric rows; failed runs must retain unranked null-score rows.")
    if rows.duplicated(["algorithm_id", "dataset_name", "metric"]).any():
        raise ValueError("Result contains duplicate method/dataset/metric rows.")
    from .leaderboard import _rank_per_dataset
    ranked = _rank_per_dataset(rows)
    keys = ["algorithm_id", "dataset_name", "metric"]
    per_dataset_check = rows.merge(ranked[keys + ["rank", "eligible"]], on=keys, suffixes=("_actual", "_expected"), validate="one_to_one")
    for row in per_dataset_check.to_dict("records"):
        if not isinstance(row["eligible_actual"], bool) or row["eligible_actual"] != row["eligible_expected"]:
            raise ValueError("Per-dataset eligibility does not match its score.")
        left, right = row["rank_expected"], row["rank_actual"]
        if pd.isna(left) and pd.isna(right):
            continue
        if pd.isna(left) or pd.isna(right) or float(left) != float(right):
            raise ValueError("Per-dataset ranks do not match its scores.")
    # Recompute scores/coverage/ranks so a partial or tampered import cannot claim eligibility.
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
    rows.loc[~rows["score"].map(math.isfinite), "score"] = float("nan")
    expected = _aggregate_leaderboard(rows, config={"strict_coverage": True, "datasets": frozen["datasets"]})
    actual = pd.DataFrame(payload.get("aggregate") or [])
    keys = ["domain", "task_family", "algorithm_id", "metric"]
    if actual.empty or not set(keys + ["eligible", "complete_coverage", "mean_score", "median_score", "rank", "dataset_count", "expected_dataset_count", "coverage", "sample_count", "elapsed_seconds"]).issubset(actual.columns) or actual.duplicated(keys).any():
        raise ValueError("Result aggregate is missing or invalid.")
    checked = expected.merge(actual, on=keys, suffixes=("_expected", "_actual"), how="outer", indicator=True)
    for row in checked.to_dict("records"):
        if row["_merge"] != "both" or not isinstance(row["eligible_actual"], bool) or row["eligible_actual"] != row["eligible_expected"] or not isinstance(row["complete_coverage_actual"], bool) or row["complete_coverage_actual"] != row["complete_coverage_expected"]:
            raise ValueError("Result aggregate eligibility does not match complete dataset coverage.")
        for column in ("mean_score", "median_score", "rank", "dataset_count", "expected_dataset_count", "coverage", "sample_count", "elapsed_seconds"):
            left, right = row[f"{column}_expected"], row[f"{column}_actual"]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) or pd.isna(right) or not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Result aggregate scores/ranks do not match per-dataset results.")
        for column in ("status", "not_applicable_reason"):
            if column + "_expected" in row:
                left = _optional_result_text(row[column + "_expected"], column)
                right = _optional_result_text(row.get(column + "_actual"), column)
                if left != right:
                    raise ValueError("Result aggregate status does not match declared applicability and coverage.")
    return {"valid": True, "protocol_id": frozen["protocol_id"], "benchmark_id": frozen["benchmark_id"],
            "methods": int(rows["algorithm_id"].nunique()), "eligible_rows": int(expected["eligible"].sum())}
