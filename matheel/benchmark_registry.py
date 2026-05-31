import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ._path_utils import path_name
from .leaderboard import leaderboard_payload


REGISTRY_SCHEMA_VERSION = 1


def empty_benchmark_registry():
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}


def load_benchmark_registry(registry_path):
    path = Path(registry_path)
    if not path.exists():
        return empty_benchmark_registry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark registry must be a JSON object.")
    runs = payload.get("runs")
    if runs is None:
        runs = []
    if not isinstance(runs, list):
        raise ValueError("Benchmark registry runs must be a list.")
    return {
        "schema_version": int(payload.get("schema_version") or REGISTRY_SCHEMA_VERSION),
        "runs": [_clean_json_object(run) for run in runs],
    }


def write_benchmark_registry(registry_path, registry):
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": int(registry.get("schema_version") or REGISTRY_SCHEMA_VERSION),
        "runs": [_clean_json_object(run) for run in registry.get("runs", [])],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def register_benchmark_run(registry_path, report_or_path, run_name=None, artifact_paths=None):
    registry = load_benchmark_registry(registry_path)
    payload = _coerce_leaderboard_payload(report_or_path)
    name = str(run_name or payload.get("metadata", {}).get("name") or "benchmark_run")
    run_id = _run_id(name, payload)
    entry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "run_id": run_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metadata": _clean_json_object(payload.get("metadata", {})),
        "manifest": _clean_json_object(payload.get("manifest", {})),
        "cards": _clean_json_object(payload.get("cards", {})),
        "aggregate": _clean_json_object(payload.get("aggregate", [])),
        "per_dataset": _clean_json_object(payload.get("per_dataset", [])),
        "artifacts": _sanitize_artifact_paths(artifact_paths),
    }
    registry["runs"] = [run for run in registry.get("runs", []) if run.get("run_id") != run_id]
    registry["runs"].append(entry)
    registry["runs"].sort(key=lambda run: (str(run.get("created_at") or ""), str(run.get("run_id") or "")))
    write_benchmark_registry(registry_path, registry)
    return entry


def list_benchmark_runs(registry_path):
    registry = load_benchmark_registry(registry_path)
    rows = []
    for run in registry.get("runs", []):
        aggregate = pd.DataFrame(run.get("aggregate") or [])
        per_dataset = pd.DataFrame(run.get("per_dataset") or [])
        rows.append(
            {
                "run_id": run.get("run_id"),
                "name": run.get("name"),
                "created_at": run.get("created_at"),
                "aggregate_rows": int(len(aggregate)),
                "per_dataset_rows": int(len(per_dataset)),
                "datasets": int(per_dataset["dataset_name"].nunique()) if "dataset_name" in per_dataset else 0,
                "algorithms": int(aggregate["algorithm_name"].nunique()) if "algorithm_name" in aggregate else 0,
                "metrics": int(aggregate["metric"].nunique()) if "metric" in aggregate else 0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "name",
            "created_at",
            "aggregate_rows",
            "per_dataset_rows",
            "datasets",
            "algorithms",
            "metrics",
        ],
    )


def load_benchmark_run(registry_path, run_id):
    registry = load_benchmark_registry(registry_path)
    for run in registry.get("runs", []):
        if str(run.get("run_id")) == str(run_id):
            return run
    raise KeyError(f"Benchmark run not found: {run_id}")


def compare_benchmark_runs(registry_path, run_ids=None):
    registry = load_benchmark_registry(registry_path)
    selected_ids = [str(run_id) for run_id in run_ids or ()]
    runs = [
        run
        for run in registry.get("runs", [])
        if not selected_ids or str(run.get("run_id")) in selected_ids
    ]
    if selected_ids:
        found = {str(run.get("run_id")) for run in runs}
        missing = sorted(set(selected_ids).difference(found))
        if missing:
            raise KeyError(f"Benchmark run not found: {', '.join(missing)}")
        runs.sort(key=lambda run: selected_ids.index(str(run.get("run_id"))))

    frames = []
    for order, run in enumerate(runs):
        frame = pd.DataFrame(run.get("aggregate") or [])
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "run_order", order)
        frame.insert(1, "run_id", run.get("run_id"))
        frame.insert(2, "run_name", run.get("name"))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["mean_score"] = pd.to_numeric(combined.get("mean_score"), errors="coerce")
    key_columns = ["task_family", "algorithm_name", "metric"]
    baseline = (
        combined[combined["run_order"] == combined["run_order"].min()][key_columns + ["mean_score"]]
        .rename(columns={"mean_score": "baseline_mean_score"})
        .drop_duplicates(subset=key_columns)
    )
    compared = combined.merge(baseline, on=key_columns, how="left")
    compared["delta_mean_score"] = compared["mean_score"] - compared["baseline_mean_score"]
    return compared.sort_values(
        by=["task_family", "metric", "algorithm_name", "run_order"],
        ignore_index=True,
    )


def _coerce_leaderboard_payload(report_or_path):
    if isinstance(report_or_path, (str, Path)):
        payload = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    elif _looks_like_payload(report_or_path):
        payload = dict(report_or_path)
    else:
        payload = leaderboard_payload(report_or_path)
    if not _looks_like_payload(payload):
        raise ValueError("Expected a Matheel leaderboard report or JSON artifact.")
    return _clean_json_object(payload)


def _looks_like_payload(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("metadata"), dict)
        and isinstance(value.get("aggregate"), list)
        and isinstance(value.get("per_dataset"), list)
    )


def _run_id(name, payload):
    stem = _slugify(name)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}"


def _slugify(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "benchmark-run").strip().lower())
    text = text.strip(".-_")
    return text or "benchmark-run"


def _sanitize_artifact_paths(artifact_paths):
    if artifact_paths is None:
        return {}
    if isinstance(artifact_paths, dict):
        raw = artifact_paths.items()
    else:
        raw = ((Path(str(path)).stem, path) for path in artifact_paths)
    return {
        str(name): path_name(path)
        for name, path in raw
        if path not in (None, "")
    }


def _clean_json_object(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _json_default(value):
    if isinstance(value, Path):
        return path_name(value)
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)
