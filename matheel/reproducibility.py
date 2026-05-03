import json
import os
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path


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
_PATH_OPTION_NAMES = frozenset(
    {
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


def _iso_utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_package_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _is_hidden_name(path_name):
    parts = str(path_name).replace("\\", "/").split("/")
    return any(part.startswith(".") for part in parts if part not in ("", "."))


def _hash_file(path):
    target = Path(path)
    digest = sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(path):
    target = Path(path)
    return {
        "source_type": "file",
        "file_name": target.name,
        "total_bytes": int(target.stat().st_size),
        "sha256": _hash_file(target),
    }


def _fingerprint_directory(root):
    root_path = Path(root)
    digest = sha256()
    file_count = 0
    total_size = 0
    for current_root, _, file_names in os.walk(root_path):
        current_root_path = Path(current_root)
        for file_name in sorted(file_names):
            full_path = current_root_path / file_name
            relative = full_path.relative_to(root_path).as_posix()
            if _is_hidden_name(relative):
                continue
            stat = full_path.stat()
            file_hash = _hash_file(full_path)
            file_count += 1
            total_size += int(stat.st_size)
            digest.update(f"{relative}|{int(stat.st_size)}|{file_hash}".encode("utf-8"))
    return {
        "source_type": "directory",
        "file_count": int(file_count),
        "total_bytes": int(total_size),
        "sha256": digest.hexdigest(),
    }


def fingerprint_source(source_path):
    resolved = Path(source_path).expanduser().resolve()
    if resolved.is_file():
        return _fingerprint_file(resolved)
    if resolved.is_dir():
        return _fingerprint_directory(resolved)
    return {"source_type": "unknown", "sha256": None}


def _json_safe(value):
    if isinstance(value, dict):
        payload = {}
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            name = str(key)
            if name in _PATH_OPTION_NAMES and isinstance(item, (str, Path)):
                payload[name] = Path(item).name if item else item
            else:
                payload[name] = _json_safe(item)
        return payload
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def collect_reproducibility_snapshot(source_path=None, run_configs=None, result_attrs=None):
    snapshot = {
        "schema_version": 1,
        "created_utc": _iso_utc_now(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {name: _safe_package_version(name) for name in _PACKAGE_NAMES},
        "run_configs": _json_safe(list(run_configs or ())),
    }
    if source_path is not None:
        snapshot["source"] = fingerprint_source(source_path)
    if result_attrs:
        snapshot["run_metadata"] = _json_safe(dict(result_attrs))
    return snapshot


def write_reproducibility_snapshot(snapshot, output_path):
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return target_path
