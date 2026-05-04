import html
import json
from pathlib import Path

import pandas as pd

from .algorithms import normalize_algorithm_options
from .calibration import calibration_report
from .datasets import load_pair_datasets, load_retrieval_datasets
from .evaluation import evaluate_pair_dataset, evaluate_retrieval_dataset
from .reproducibility import collect_reproducibility_snapshot, write_reproducibility_snapshot


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
_PATH_KEYS = {"identifier", "path", "algorithm_path"}


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
        "datasets": [
            _normalize_leaderboard_dataset(item, index=index, base_dir=base_dir)
            for index, item in enumerate(datasets, start=1)
        ],
        "algorithms": [
            _normalize_leaderboard_algorithm(item, index=index, base_dir=base_dir)
            for index, item in enumerate(algorithms, start=1)
        ],
    }


def run_leaderboard(manifest, output_dir=None, basename="leaderboard"):
    config = normalize_leaderboard_manifest(manifest) if not _is_normalized_manifest(manifest) else manifest
    rows = []
    for dataset_config in config["datasets"]:
        for algorithm_config in config["algorithms"]:
            rows.extend(_evaluate_leaderboard_run(config, dataset_config, algorithm_config))
    per_dataset = _rank_per_dataset(pd.DataFrame(rows))
    aggregate = _aggregate_leaderboard(per_dataset)
    report = {
        "metadata": {
            "schema_version": 1,
            "name": config["name"],
            "seed": config.get("seed"),
            "pair_metrics": list(config["pair_metrics"]),
            "retrieval_metrics": list(config["retrieval_metrics"]),
        },
        "manifest": config,
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
        "per_dataset": _frame_records(report["per_dataset"]),
        "aggregate": _frame_records(report["aggregate"]),
    }


def leaderboard_html(report, title="Matheel Leaderboard"):
    payload = leaderboard_payload(report)
    escaped_title = html.escape(str(title))
    aggregate = pd.DataFrame(payload["aggregate"])
    per_dataset = pd.DataFrame(payload["per_dataset"])
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
        "reproducibility_json": target / f"{stem}_reproducibility.json",
    }
    report["per_dataset"].to_csv(artifacts["per_dataset_csv"], index=False)
    report["aggregate"].to_csv(artifacts["aggregate_csv"], index=False)
    artifacts["json"].write_text(json.dumps(leaderboard_payload(report), indent=2, sort_keys=True), encoding="utf-8")
    artifacts["html"].write_text(
        leaderboard_html(report, title=str(report["metadata"].get("name") or "Matheel Leaderboard")),
        encoding="utf-8",
    )
    snapshot = collect_reproducibility_snapshot(run_configs=[report.get("manifest", {})])
    write_reproducibility_snapshot(snapshot, artifacts["reproducibility_json"])
    return artifacts


def _evaluate_leaderboard_run(manifest, dataset_config, algorithm_config):
    if dataset_config["task_family"] == "pair":
        return _evaluate_pair_leaderboard_run(manifest, dataset_config, algorithm_config)
    return _evaluate_retrieval_leaderboard_run(manifest, dataset_config, algorithm_config)


def _evaluate_pair_leaderboard_run(manifest, dataset_config, algorithm_config):
    dataset = load_pair_datasets(dataset_config["spec"])
    scored, metrics = evaluate_pair_dataset(
        dataset,
        threshold=float(_first_not_none(algorithm_config.get("threshold"), dataset_config.get("threshold"), 0.5)),
        similarity_options=algorithm_config["similarity_options"],
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
    return [
        _leaderboard_row(
            task_family="pair",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            metric=metric,
            value=metrics.get(metric),
            sample_count=len(scored),
        )
        for metric in manifest["pair_metrics"]
    ]


def _evaluate_retrieval_leaderboard_run(manifest, dataset_config, algorithm_config):
    dataset = load_retrieval_datasets(dataset_config["spec"])
    scored, metrics = evaluate_retrieval_dataset(
        dataset,
        k=int(_first_not_none(algorithm_config.get("k"), dataset_config.get("k"), 10)),
        similarity_options=algorithm_config["similarity_options"],
        algorithm=algorithm_config.get("algorithm_path"),
        algorithm_options=algorithm_config.get("algorithm_options"),
    )
    return [
        _leaderboard_row(
            task_family="retrieval",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            metric=metric,
            value=metrics.get(metric),
            sample_count=len(scored),
        )
        for metric in manifest["retrieval_metrics"]
    ]


def _leaderboard_row(task_family, dataset_config, algorithm_config, metric, value, sample_count):
    score = None if value is None else float(value)
    return {
        "task_family": task_family,
        "dataset_name": dataset_config["name"],
        "algorithm_name": algorithm_config["name"],
        "metric": metric,
        "score": score,
        "sample_count": int(sample_count),
        "dataset_source": dataset_config["spec"].get("source", "local"),
        "algorithm_kind": "custom" if algorithm_config.get("algorithm_path") else "builtin",
    }


def _rank_per_dataset(frame):
    if frame.empty:
        return pd.DataFrame()
    ranked = frame.copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked["rank"] = (
        ranked.groupby(["task_family", "dataset_name", "metric"])["score"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype(int)
    )
    return ranked.sort_values(
        by=["task_family", "dataset_name", "metric", "rank", "algorithm_name"],
        ignore_index=True,
    )


def _aggregate_leaderboard(per_dataset):
    if per_dataset.empty:
        return pd.DataFrame()
    grouped = (
        per_dataset.groupby(["task_family", "algorithm_name", "metric"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            dataset_count=("dataset_name", "nunique"),
            sample_count=("sample_count", "sum"),
        )
    )
    grouped["rank"] = (
        grouped.groupby(["task_family", "metric"])["mean_score"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype(int)
    )
    return grouped.sort_values(
        by=["task_family", "metric", "rank", "algorithm_name"],
        ignore_index=True,
    )


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
    name = str(payload.get("name") or f"{task_family}_dataset_{index}")
    payload["name"] = name
    payload["task_families"] = (task_family,)
    payload = _resolve_relative_paths(payload, base_dir=base_dir)
    return {
        "name": name,
        "task_family": task_family,
        "spec": payload,
        "threshold": item.get("threshold"),
        "k": item.get("k"),
    }


def _normalize_leaderboard_algorithm(item, index=1, base_dir=None):
    if not isinstance(item, dict):
        raise ValueError("Leaderboard algorithms must be JSON objects.")
    payload = dict(item)
    options = dict(payload.pop("options", {}))
    options.update(payload.pop("similarity_options", {}))
    for key in list(payload):
        if key not in {"name", "algorithm_path", "algorithm_options", "threshold", "k"}:
            options[key] = payload.pop(key)
    payload = _resolve_relative_paths(payload, base_dir=base_dir)
    name = str(payload.get("name") or f"algorithm_{index}")
    return {
        "name": name,
        "algorithm_path": payload.get("algorithm_path"),
        "algorithm_options": normalize_algorithm_options(payload.get("algorithm_options")),
        "similarity_options": options,
        "threshold": payload.get("threshold"),
        "k": payload.get("k"),
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
    for key in _PATH_KEYS:
        if key in resolved:
            resolved[key] = _resolve_relative_path(resolved[key], base_dir)
    return resolved


def _resolve_relative_path(value, base_dir):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or Path(text).expanduser().is_absolute() or not _looks_like_path(text):
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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value.item() if hasattr(value, "item") else str(value)


def _safe_basename(value):
    text = str(value or "leaderboard").strip()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    return safe.strip("._") or "leaderboard"
