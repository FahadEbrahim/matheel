import hashlib
import html
import json
import math
from time import perf_counter
from pathlib import Path, PureWindowsPath

import pandas as pd

from .algorithms import normalize_algorithm_options
from .benchmark_cards import leaderboard_cards
from .calibration import calibration_report
from .datasets import dataset_task_type, load_pair_datasets, load_retrieval_datasets, normalize_dataset_task_type
from .evaluation import evaluate_pair_dataset, evaluate_retrieval_dataset
from .leaderboard_presets import get_leaderboard_algorithm_preset
from ._path_utils import path_name
from .reproducibility import collect_reproducibility_snapshot, write_reproducibility_snapshot
from .task_eligibility import method_exclusion_reason, normalize_dataset_exclusions, normalize_task_eligibility


PAIR_LEADERBOARD_METRICS = (
    "f1",
    "accuracy",
    "precision",
    "recall",
    "auroc",
    "average_precision",
)
RETRIEVAL_LEADERBOARD_METRICS = (
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
)
_TASK_ALIASES = {
    "pair": "pair",
    "pair_classification": "pair",
    "classification": "pair",
    "retrieval": "retrieval",
    "ranking": "retrieval",
}
_PATH_KEYS = {"identifier", "path", "algorithm_path", "destination", "adapted_destination"}
_ALWAYS_LOCAL_PATH_KEYS = {"path", "algorithm_path", "destination", "adapted_destination"}
_MAYBE_LOCAL_PATH_KEYS = {"identifier"}
_CREDENTIAL_KEYS = {
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "token",
}


def available_leaderboard_metrics(task_family=None):
    if task_family is None:
        return {
            "pair": PAIR_LEADERBOARD_METRICS,
            "retrieval": RETRIEVAL_LEADERBOARD_METRICS,
        }
    task = _normalize_task_family(task_family)
    return PAIR_LEADERBOARD_METRICS if task == "pair" else RETRIEVAL_LEADERBOARD_METRICS


def load_leaderboard_manifest(config_path):
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    credential_paths = _credential_paths(payload)
    if credential_paths:
        names = ", ".join(credential_paths)
        raise ValueError(
            "Leaderboard manifest files must not contain credentials; "
            f"remove credential field(s): {names}. Use provider-managed authentication instead."
        )
    return normalize_leaderboard_manifest(payload, base_dir=path.parent)


def normalize_leaderboard_manifest(manifest, base_dir=None):
    if not isinstance(manifest, dict):
        raise ValueError("Leaderboard manifest must be a JSON object.")
    datasets = manifest.get("datasets")
    algorithms = manifest.get("algorithms")
    if not datasets:
        raise ValueError("Leaderboard manifest must include at least one dataset.")
    if not algorithms:
        raise ValueError("Leaderboard manifest must include at least one algorithm.")
    normalized_datasets = [
        _normalize_leaderboard_dataset(item, index=index, base_dir=base_dir)
        for index, item in enumerate(datasets, start=1)
    ]
    normalized_algorithms = [
        _normalize_leaderboard_algorithm(item, index=index, base_dir=base_dir)
        for index, item in enumerate(algorithms, start=1)
    ]
    _validate_unique_names(normalized_datasets, "dataset")
    _validate_unique_names(normalized_algorithms, "algorithm")
    names = {item["name"] for item in normalized_datasets}
    for method in normalized_algorithms:
        unknown = set(method.get("dataset_exclusions", {})) - names
        if unknown:
            raise ValueError(f"Unknown dataset exclusion(s): {', '.join(sorted(unknown))}")
    return {
        "schema_version": int(manifest.get("schema_version") or 1),
        "name": str(manifest.get("name") or "matheel_leaderboard"),
        "seed": manifest.get("seed"),
        "pair_metrics": _normalize_metric_list(
            manifest.get("pair_metrics") or manifest.get("pair_metric") or ("f1",),
            "pair",
        ),
        "retrieval_metrics": _normalize_metric_list(
            manifest.get("retrieval_metrics") or manifest.get("retrieval_metric") or ("mean_average_precision",),
            "retrieval",
        ),
        "datasets": normalized_datasets,
        "algorithms": normalized_algorithms,
        "strict_coverage": bool(manifest.get("strict_coverage", False)),
        "continue_on_error": bool(manifest.get("continue_on_error", False)),
    }


def run_leaderboard(manifest, output_dir=None, basename="leaderboard", *, score_artifact_dir=None):
    config = normalize_leaderboard_manifest(manifest) if not _is_normalized_manifest(manifest) else manifest
    _validate_unique_names(config["datasets"], "dataset")
    _validate_unique_names(config["algorithms"], "algorithm")
    rows = []
    score_artifacts = []
    for dataset_config in config["datasets"]:
        for algorithm_config in config["algorithms"]:
            reason = method_exclusion_reason(algorithm_config, dataset_config)
            if reason:
                load = load_pair_datasets if dataset_config["task_family"] == "pair" else load_retrieval_datasets
                dataset_config["domain"] = dataset_task_type(load(dataset_config["spec"]))
                metrics = config["pair_metrics" if dataset_config["task_family"] == "pair" else "retrieval_metrics"]
                rows.extend({**_leaderboard_row(dataset_config["task_family"], dataset_config, algorithm_config, metric, None, 0),
                             "status": "not_applicable", "not_applicable_reason": reason, "elapsed_seconds": 0.0}
                            for metric in metrics)
                continue
            started = perf_counter()
            try:
                if score_artifact_dir is None:
                    run_rows = _evaluate_leaderboard_run(config, dataset_config, algorithm_config)
                else:
                    run_rows = _evaluate_leaderboard_run(config, dataset_config, algorithm_config, score_artifact_dir=score_artifact_dir)
            except Exception as exc:
                if not config.get("continue_on_error"):
                    raise
                metrics = config["pair_metrics" if dataset_config["task_family"] == "pair" else "retrieval_metrics"]
                run_rows = [
                    {**_leaderboard_row(dataset_config["task_family"], dataset_config, algorithm_config, metric, None, 0),
                     "error": f"{type(exc).__name__}: {exc}", "status": "failed"}
                    for metric in metrics
                ]
            elapsed = perf_counter() - started
            for row in run_rows:
                row["elapsed_seconds"] = elapsed
                score_artifact = row.pop("_score_artifact", None)
                if score_artifact is not None and score_artifact not in score_artifacts:
                    score_artifacts.append(score_artifact)
            rows.extend(run_rows)
    per_dataset = _rank_per_dataset(pd.DataFrame(rows))
    aggregate = _aggregate_leaderboard(per_dataset, config=config)
    report = {
        "metadata": {
            "schema_version": 1,
            "name": config["name"],
            "seed": config.get("seed"),
            "pair_metrics": list(config["pair_metrics"]),
            "retrieval_metrics": list(config["retrieval_metrics"]),
            "strict_coverage": config.get("strict_coverage", False),
            "score_artifacts": score_artifacts,
        },
        "manifest": config,
        "cards": leaderboard_cards(config),
        "per_dataset": per_dataset,
        "aggregate": aggregate,
    }
    artifacts = None
    if output_dir is not None:
        artifacts = write_leaderboard_artifacts(report, output_dir, basename=basename)
    return report, artifacts


def leaderboard_payload(report):
    return {
        "schema_version": 1,
        "metadata": _json_safe(report.get("metadata", {})),
        "manifest": _json_safe(report.get("manifest", {})),
        "cards": _json_safe(report.get("cards", {})),
        "per_dataset": _frame_records(report["per_dataset"]),
        "aggregate": _frame_records(report["aggregate"]),
    }


def leaderboard_html(report, title="Matheel Leaderboard"):
    payload = leaderboard_payload(report)
    escaped_title = html.escape(str(title))
    aggregate = pd.DataFrame(payload["aggregate"])
    per_dataset = pd.DataFrame(payload["per_dataset"])
    dataset_count = len(payload.get("cards", {}).get("datasets", []))
    algorithm_count = len(payload.get("cards", {}).get("algorithms", []))
    aggregate_html = aggregate.to_html(index=False, escape=True) if not aggregate.empty else "<p>No aggregate rows.</p>"
    per_dataset_html = per_dataset.to_html(index=False, escape=True) if not per_dataset.empty else "<p>No dataset rows.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 16px; }}
    h2 {{ font-size: 1.1rem; margin: 24px 0 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <p>{dataset_count} dataset card(s), {algorithm_count} algorithm card(s)</p>
  <h2>Aggregate</h2>
  {aggregate_html}
  <h2>Per Dataset</h2>
  {per_dataset_html}
</body>
</html>
"""


def write_leaderboard_artifacts(report, output_dir, basename="leaderboard"):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_basename(basename)
    artifacts = {
        "per_dataset_csv": target / f"{stem}_per_dataset.csv",
        "aggregate_csv": target / f"{stem}_aggregate.csv",
        "json": target / f"{stem}.json",
        "html": target / f"{stem}.html",
        "details_html": target / f"{stem}_details.html",
        "reproducibility_json": target / f"{stem}_reproducibility.json",
    }
    payload = leaderboard_payload(report)
    report["per_dataset"].to_csv(artifacts["per_dataset_csv"], index=False)
    report["aggregate"].to_csv(artifacts["aggregate_csv"], index=False)
    artifacts["json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    from .reports import benchmark_detail_report_html, benchmark_report_html

    artifacts["html"].write_text(
        benchmark_report_html(
            report,
            title=str(report["metadata"].get("name") or "Matheel Leaderboard"),
            artifact_links={
                name: path
                for name, path in artifacts.items()
                if name != "html"
            },
        ),
        encoding="utf-8",
    )
    artifacts["details_html"].write_text(
        benchmark_detail_report_html(
            report,
            title=f"{report['metadata'].get('name') or 'Matheel Leaderboard'} Details",
            artifact_links={
                name: path
                for name, path in artifacts.items()
                if name != "details_html"
            },
        ),
        encoding="utf-8",
    )
    snapshot = collect_reproducibility_snapshot(run_configs=[payload.get("manifest", {})])
    write_reproducibility_snapshot(snapshot, artifacts["reproducibility_json"])
    return artifacts


def _evaluate_leaderboard_run(manifest, dataset_config, algorithm_config, *, score_artifact_dir=None):
    if dataset_config["task_family"] == "pair":
        return _evaluate_pair_leaderboard_run(manifest, dataset_config, algorithm_config, score_artifact_dir=score_artifact_dir)
    return _evaluate_retrieval_leaderboard_run(manifest, dataset_config, algorithm_config, score_artifact_dir=score_artifact_dir)


def _evaluate_pair_leaderboard_run(manifest, dataset_config, algorithm_config, *, score_artifact_dir=None):
    dataset = load_pair_datasets(dataset_config["spec"])
    dataset_config["domain"] = dataset_task_type(dataset)
    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=float(_first_not_none(algorithm_config.get("threshold"), dataset_config.get("threshold"), 0.5)),
        similarity_options=_leaderboard_similarity_options(dataset_config, algorithm_config),
        algorithm=algorithm_config.get("algorithm_path"),
        algorithm_options=algorithm_config.get("algorithm_options"),
    )
    metrics = dict(metrics)
    try:
        calibration = calibration_report(scored, score_key="similarity_score", label_key="label")
    except ValueError:
        calibration = None
    if calibration is not None:
        metrics["auroc"] = calibration["summary"]["auroc"]
        metrics["average_precision"] = calibration["summary"]["average_precision"]
    artifact = _write_score_artifact(scored, dataset_config, algorithm_config, score_artifact_dir)
    return [
        {**_leaderboard_row(
            task_family="pair",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            metric=metric,
            value=metrics.get(metric),
            sample_count=len(scored),
        ), "_score_artifact": artifact}
        for metric in manifest["pair_metrics"]
    ]


def _evaluate_retrieval_leaderboard_run(manifest, dataset_config, algorithm_config, *, score_artifact_dir=None):
    dataset = load_retrieval_datasets(dataset_config["spec"])
    dataset_config["domain"] = dataset_task_type(dataset)
    evaluate = evaluate_retrieval_dataset
    retrieval_options = algorithm_config.get("retrieval_options")
    if retrieval_options:
        from .retrieval import evaluate_two_stage_retrieval_dataset
        evaluate = evaluate_two_stage_retrieval_dataset
    scored, metrics = evaluate(
        dataset,
        k=int(_first_not_none(algorithm_config.get("k"), dataset_config.get("k"), 10)),
        exclude_self_matches=bool(dataset_config.get("exclude_self_matches", False)),
        similarity_options=_leaderboard_similarity_options(dataset_config, algorithm_config),
        algorithm=algorithm_config.get("algorithm_path"),
        algorithm_options=algorithm_config.get("algorithm_options"),
        **(retrieval_options or {}),
    )
    artifact = _write_score_artifact(scored, dataset_config, algorithm_config, score_artifact_dir)
    return [
        {**_leaderboard_row(
            task_family="retrieval",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            metric=metric,
            value=metrics.get(metric),
            sample_count=len(scored),
        ), "_score_artifact": artifact,
         **({key: metrics[key] for key in ("retrieval_mode", "candidate_count", "candidate_recall",
                                           "candidate_recall_micro", "first_stage_comparisons", "reranker_comparisons")}
            if retrieval_options else {})}
        for metric in manifest["retrieval_metrics"]
    ]



def _write_score_artifact(scored, dataset_config, algorithm_config, output_dir):
    if output_dir is None:
        return None
    safe_columns = {
        "left_id", "right_id", "label", "similarity_score", "query_id", "document_id",
        "query_file_id", "document_file_id", "relevance", "group", "group_id", "assignment",
        "assignment_id", "family", "family_id", "language", "split",
    }
    frame = scored[[column for column in scored.columns if column in safe_columns]].copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        if any(Path(str(value)).is_absolute() or PureWindowsPath(str(value)).is_absolute() for value in frame[column]):
            raise ValueError("Score artifacts cannot contain absolute paths in identifiers or group metadata.")
    identity = _algorithm_identity(algorithm_config)
    dataset_id = hashlib.sha256(dataset_config["name"].encode()).hexdigest()[:12]
    filename = f"{dataset_id}_{identity[:20]}_{dataset_config['task_family']}.csv"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    frame.to_csv(path, index=False)
    return {"dataset_name": dataset_config["name"], "task_family": dataset_config["task_family"],
            "algorithm_name": algorithm_config["name"], "algorithm_id": identity,
            "filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame), "columns": list(frame.columns)}

def _leaderboard_similarity_options(dataset_config, algorithm_config):
    return {
        **algorithm_config["similarity_options"],
        **dataset_config.get("similarity_options", {}),
    }


def _leaderboard_row(task_family, dataset_config, algorithm_config, metric, value, sample_count):
    score = None if value is None else float(value)
    return {
        "domain": dataset_config.get("domain", "plagiarism"),
        "task_type": dataset_config.get("domain", "plagiarism"),
        "task_family": task_family,
        "dataset_name": dataset_config["name"],
        "algorithm_name": algorithm_config["name"],
        "algorithm_id": _algorithm_identity(algorithm_config),
        "metric": metric,
        "score": score,
        "sample_count": int(sample_count),
        "dataset_source": dataset_config["spec"].get("source", "local"),
        "algorithm_kind": "custom" if algorithm_config.get("algorithm_path") else "builtin",
        "status": "completed",
        "not_applicable_reason": "",
    }


def _algorithm_identity(config):
    # Names are presentation; scores from distinct configurations must not share an identity.
    from .benchmark_protocol import method_configuration_id

    return config.get("_configuration_id") or method_configuration_id(config)


def _rank_per_dataset(frame):
    if frame.empty:
        return pd.DataFrame()
    ranked = frame.copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked.loc[~ranked["score"].map(math.isfinite), "score"] = float("nan")
    if "status" in ranked and (ranked["status"].isin(["not_applicable", "failed"]) & ranked["score"].notna()).any():
        raise ValueError("Failed or not-applicable rows cannot contain a finite score.")
    ranked["eligible"] = ranked["score"].notna()
    keys = [column for column in ("domain", "task_family", "dataset_name", "metric") if column in ranked]
    ranked["rank"] = ranked.groupby(keys)["score"].rank(method="min", ascending=False).astype("Int64")
    return ranked.sort_values(by=keys + ["rank", "algorithm_name"], ignore_index=True, na_position="last")


def _aggregate_leaderboard(per_dataset, config=None):
    if per_dataset.empty:
        return pd.DataFrame()
    keys = [column for column in ("domain", "task_family", "algorithm_name", "algorithm_id", "metric") if column in per_dataset]
    rows = []
    for values, group in per_dataset.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values))
        expected = {
            item["name"] for item in (config or {}).get("datasets", [])
            if item["task_family"] == row["task_family"] and item.get("domain", "plagiarism") == row.get("domain", "plagiarism")
        } or set(group["dataset_name"])
        valid = group[group["score"].notna()]
        covered = set(valid["dataset_name"])
        complete = covered == expected and len(group) == len(expected)
        excluded = group.get("status", pd.Series("", index=group.index)).eq("not_applicable").any()
        eligible = not excluded and (complete if (config or {}).get("strict_coverage") else not valid.empty)
        if "status" in group:
            reasons = sorted(set(group.loc[group["status"].eq("not_applicable"), "not_applicable_reason"].dropna()))
            row.update(status="not_applicable" if excluded else ("completed" if complete else "failed"),
                       not_applicable_reason="; ".join(reasons))
        row.update(
            mean_score=float(valid["score"].mean()) if eligible else None,
            median_score=float(valid["score"].median()) if eligible else None,
            dataset_count=len(covered), expected_dataset_count=len(expected),
            coverage=len(covered) / len(expected), complete_coverage=complete, eligible=eligible,
            sample_count=int(group["sample_count"].sum()),
            elapsed_seconds=float(group.get("elapsed_seconds", pd.Series(dtype=float)).sum()),
        )
        rows.append(row)
    grouped = pd.DataFrame(rows)
    if "domain" in grouped:
        grouped["task_type"] = grouped["domain"]
    grouped["mean_score"] = pd.to_numeric(grouped["mean_score"], errors="coerce")
    rank_keys = [column for column in ("domain", "task_family", "metric") if column in grouped]
    grouped["rank"] = grouped.groupby(rank_keys)["mean_score"].rank(method="min", ascending=False).astype("Int64")
    return grouped.sort_values(by=rank_keys + ["rank", "algorithm_name"], ignore_index=True, na_position="last")


def _normalize_leaderboard_dataset(item, index=1, base_dir=None):
    if not isinstance(item, dict):
        raise ValueError("Leaderboard datasets must be JSON objects.")
    task_family = _normalize_task_family(item.get("task") or item.get("task_family") or item.get("kind"))
    payload = dict(item)
    if "path" in payload and "identifier" not in payload:
        payload["identifier"] = payload.pop("path")
    payload.pop("task", None)
    payload.pop("task_family", None)
    payload.pop("kind", None)
    similarity_options = dict(payload.pop("similarity_options", {}))
    name = str(payload.get("name") or f"{task_family}_dataset_{index}")
    payload["name"] = name
    payload["task_families"] = (task_family,)
    declared_domain = item.get("domain") or item.get("task_type")
    if declared_domain is not None:
        payload["task_type"] = normalize_dataset_task_type(declared_domain)
        if item.get("task_type") is not None and normalize_dataset_task_type(item["task_type"]) != payload["task_type"]:
            raise ValueError("Leaderboard dataset domain and task_type conflict.")
    payload = _resolve_relative_paths(payload, base_dir=base_dir)
    return {
        "name": name,
        "domain": normalize_dataset_task_type(declared_domain),
        "task_family": task_family,
        "spec": payload,
        "threshold": item.get("threshold"),
        "k": item.get("k"),
        "exclude_self_matches": bool(item.get("exclude_self_matches", False)),
        "similarity_options": similarity_options,
    }


def _normalize_leaderboard_algorithm(item, index=1, base_dir=None):
    from .retrieval import normalize_retrieval_options
    if isinstance(item, str):
        item = {"preset": item}
    if not isinstance(item, dict):
        raise ValueError("Leaderboard algorithms must be JSON objects.")
    payload = dict(item)
    preset_name = payload.pop("preset", None) or payload.pop("algorithm_preset", None)
    preset_options = {}
    preset_display_name = None
    if preset_name is not None:
        preset = get_leaderboard_algorithm_preset(preset_name)
        preset_options = dict(preset.get("similarity_options") or {})
        preset_display_name = preset.get("name")
        if payload.get("algorithm_path"):
            raise ValueError("Leaderboard algorithm presets cannot be combined with algorithm_path.")
    options = dict(payload.pop("options", {}))
    options = {**preset_options, **options}
    options.update(payload.pop("similarity_options", {}))
    for key in list(payload):
        if key not in {"name", "algorithm_path", "algorithm_options", "threshold", "k", "model_revision", "task_eligibility", "dataset_exclusions", "retrieval_options"}:
            options[key] = payload.pop(key)
    payload = _resolve_relative_paths(payload, base_dir=base_dir)
    name = str(payload.get("name") or preset_display_name or f"algorithm_{index}")
    return {
        "name": name,
        "algorithm_path": payload.get("algorithm_path"),
        "algorithm_options": normalize_algorithm_options(payload.get("algorithm_options")),
        "similarity_options": options,
        "threshold": payload.get("threshold"),
        "k": payload.get("k"),
        **({"model_revision": payload["model_revision"]} if payload.get("model_revision") else {}),
        **({"task_eligibility": normalize_task_eligibility(payload["task_eligibility"])} if "task_eligibility" in payload else {}),
        **({"dataset_exclusions": normalize_dataset_exclusions(payload["dataset_exclusions"])} if "dataset_exclusions" in payload else {}),
        **({"retrieval_options": normalize_retrieval_options(payload["retrieval_options"])} if "retrieval_options" in payload else {}),
    }


def _normalize_metric_list(metrics, task_family):
    if isinstance(metrics, str):
        raw = [metrics]
    else:
        raw = list(metrics)
    supported = set(available_leaderboard_metrics(task_family))
    normalized = []
    for metric in raw:
        key = str(metric).strip().lower()
        if key not in supported:
            names = ", ".join(sorted(supported))
            raise ValueError(f"Unsupported {task_family} leaderboard metric: {metric}. Supported metrics: {names}.")
        normalized.append(key)
    if not normalized:
        raise ValueError(f"At least one {task_family} metric is required.")
    return tuple(dict.fromkeys(normalized))


def _normalize_task_family(value):
    key = str(value or "").strip().lower()
    try:
        return _TASK_ALIASES[key]
    except KeyError as exc:
        raise ValueError("Leaderboard task must be one of: pair, pair_classification, retrieval.") from exc


def _resolve_relative_paths(payload, base_dir=None):
    if base_dir is None:
        return payload
    resolved = dict(payload)
    source = str(resolved.get("source") or "local").strip().lower()
    for key in ("path", "algorithm_path", "destination", "adapted_destination"):
        if key in resolved:
            resolved[key] = _resolve_relative_path(resolved[key], base_dir)
    if "identifier" in resolved and source == "local":
        resolved["identifier"] = _resolve_relative_path(resolved["identifier"], base_dir, force=True)
    return resolved


def _resolve_relative_path(value, base_dir, force=False):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or Path(text).expanduser().is_absolute() or (not force and not _looks_like_path(text)):
        return value
    return str((Path(base_dir) / text).resolve())


def _looks_like_path(value):
    text = str(value)
    return text.startswith(".") or text.endswith(".py") or "/" in text or "\\" in text


def _is_normalized_manifest(manifest):
    if not isinstance(manifest, dict) or "datasets" not in manifest or "algorithms" not in manifest:
        return False
    datasets = manifest.get("datasets") or []
    algorithms = manifest.get("algorithms") or []
    if not datasets or not algorithms:
        return False
    return "task_family" in datasets[0] and "spec" in datasets[0] and "similarity_options" in algorithms[0]


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _frame_records(frame):
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _json_safe(value, key_name=None):
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _is_credential_key(key) else _json_safe(item, key_name=str(key))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, key_name=key_name) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and _should_sanitize_manifest_path(key_name, value):
        return _path_name(value)
    if pd.isna(value) or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value.item() if hasattr(value, "item") else str(value)


def _is_credential_key(value):
    name = "".join(
        character
        for character in str(value or "").strip().casefold()
        if character.isalnum()
    )
    exact = {"".join(character for character in key if character.isalnum()) for key in _CREDENTIAL_KEYS}
    return (
        name in exact
        or name.endswith("apikey")
        or name.endswith("password")
        or name.endswith("secret")
        or name.endswith("token")
    )


def _credential_paths(value, prefix=""):
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if _is_credential_key(name) and item not in (None, ""):
                paths.append(path)
            else:
                paths.extend(_credential_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_credential_paths(item, prefix=path))
    return paths


def _validate_unique_names(items, kind):
    seen = {}
    duplicates = []
    for item in items:
        name = str(item["name"])
        identity = name.strip().casefold()
        if not identity:
            raise ValueError(f"Leaderboard {kind} names must not be blank.")
        if identity in seen and identity not in duplicates:
            duplicates.append(identity)
        seen.setdefault(identity, name)
    if duplicates:
        names = ", ".join(repr(seen[identity]) for identity in duplicates)
        raise ValueError(f"Leaderboard {kind} names must be unique; duplicate name(s): {names}.")


def _should_sanitize_manifest_path(key_name, value):
    name = str(key_name or "")
    if name in _ALWAYS_LOCAL_PATH_KEYS:
        return True
    if name not in _MAYBE_LOCAL_PATH_KEYS:
        return False
    text = str(value or "").strip()
    if not text:
        return False
    windows_path = PureWindowsPath(text)
    return Path(text).expanduser().is_absolute() or windows_path.is_absolute() or text.startswith(".")


def _path_name(value):
    return path_name(value)


def _safe_basename(value):
    text = str(value or "leaderboard").strip()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    return safe.strip("._") or "leaderboard"
