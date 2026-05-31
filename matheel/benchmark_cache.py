import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import pandas as pd

from .algorithms import resolve_pair_algorithm
from ._path_utils import path_name
from .reproducibility import fingerprint_source


BENCHMARK_CACHE_SCHEMA_VERSION = 1
_PACKAGE_NAMES = (
    "matheel",
    "numpy",
    "pandas",
    "rapidfuzz",
    "sentence-transformers",
    "model2vec",
    "pylate",
    "tree-sitter-language-pack",
)
_PATH_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_path",
        "source_path",
        "dataset_path",
        "scores_out",
        "metrics_out",
        "reproducibility_out",
        "summary_out",
        "details_dir",
    }
)


def benchmark_dependency_versions(package_names=None):
    return {
        name: _safe_package_version(name)
        for name in (package_names or _PACKAGE_NAMES)
    }


def benchmark_cache_key(source_fingerprint, run_config, dependency_versions=None, seed=None, extra=None):
    components = {
        "schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "source": _json_safe(source_fingerprint),
        "run_config": _json_safe(run_config),
        "dependency_versions": _json_safe(dependency_versions or benchmark_dependency_versions()),
        "seed": _json_safe(seed),
        "extra": _json_safe(extra or {}),
    }
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "key": sha256(payload.encode("utf-8")).hexdigest(),
        "components": components,
    }


def benchmark_cache_key_for_run(source_path, run_config, seed=None, dependency_versions=None, extra=None):
    options = dict(run_config.get("options", {}))
    algorithm_fingerprint = _algorithm_fingerprint_from_options(options)
    key_extra = dict(extra or {})
    if algorithm_fingerprint:
        key_extra["algorithm_fingerprint"] = algorithm_fingerprint
    return benchmark_cache_key(
        fingerprint_source(source_path),
        run_config,
        dependency_versions=dependency_versions,
        seed=seed,
        extra=key_extra,
    )


def benchmark_cache_paths(cache_dir, cache_key):
    safe_key = _validate_cache_key(cache_key)
    root = Path(cache_dir) / safe_key[:2] / safe_key
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "results": root / "results.csv",
    }


def load_benchmark_cache_result(cache_dir, cache_key):
    paths = benchmark_cache_paths(cache_dir, cache_key)
    if not paths["metadata"].exists() or not paths["results"].exists():
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    results = pd.read_csv(paths["results"])
    results.attrs.update(metadata.get("result_attrs") or {})
    results.attrs["cache_status"] = "hit"
    results.attrs["cache_key"] = metadata.get("cache_key") or _validate_cache_key(cache_key)
    return results, metadata


def write_benchmark_cache_result(cache_dir, cache_key_payload, results, metadata=None):
    cache_key = cache_key_payload["key"] if isinstance(cache_key_payload, dict) else cache_key_payload
    paths = benchmark_cache_paths(cache_dir, cache_key)
    paths["root"].mkdir(parents=True, exist_ok=True)
    frame = results.copy()
    frame.to_csv(paths["results"], index=False)
    payload = {
        "schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": _validate_cache_key(cache_key),
        "key_components": _json_safe(cache_key_payload.get("components", {}) if isinstance(cache_key_payload, dict) else {}),
        "metadata": _json_safe(metadata or {}),
        "result_attrs": _json_safe(getattr(results, "attrs", {})),
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def clear_benchmark_cache(cache_dir, missing_ok=True):
    target = Path(cache_dir)
    if not target.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(target)
    shutil.rmtree(target)
    return True


def _safe_package_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _algorithm_fingerprint_from_options(options):
    algorithm = options.get("algorithm") or options.get("algorithm_path")
    if algorithm is None:
        return None
    try:
        resolved = resolve_pair_algorithm(algorithm)
    except Exception:
        return {"source_type": "unresolved", "name": path_name(algorithm)}
    return {
        "algorithm_name": resolved.name,
        "algorithm_module": resolved.module_name,
        "algorithm_package": resolved.package_name,
        "algorithm_package_version": resolved.package_version,
        "algorithm_source_fingerprint": dict(resolved.source_fingerprint or {}),
    }


def _validate_cache_key(cache_key):
    key = str(cache_key or "").strip().lower()
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise ValueError("cache_key must be a 64-character SHA-256 hex digest.")
    return key


def _json_safe(value, key_name=None):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, key_name=str(key))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return path_name(value)
    if key_name in _PATH_KEYS and isinstance(value, str):
        return path_name(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
