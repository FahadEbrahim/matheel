import json
from pathlib import Path, PureWindowsPath

from .algorithms import normalize_algorithm_options, resolve_pair_algorithm
from .datasets import (
    PairDataset,
    RetrievalDataset,
    load_pair_dataset,
    load_pair_datasets,
    load_retrieval_dataset,
    load_retrieval_datasets,
)
from .reproducibility import fingerprint_source


CARD_SCHEMA_VERSION = 1
_PATH_KEYS = {"identifier", "algorithm_path", "path", "destination", "adapted_destination"}
_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "client_secret",
    "password",
    "secret",
    "token",
}


def dataset_card(dataset, name=None, task_family=None, source_spec=None):
    loaded = _load_card_dataset(dataset, task_family=task_family)
    metadata = dict(getattr(loaded, "metadata", {}) or {})
    resolved_task_family = task_family or ("retrieval" if isinstance(loaded, RetrievalDataset) else "pair")
    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "card_type": "dataset",
        "name": str(name or metadata.get("name") or loaded.root.name),
        "task_family": resolved_task_family,
        "dataset_kind": str(metadata.get("dataset_kind") or _dataset_kind_from_task(resolved_task_family)),
        "task_type": str(metadata.get("task_type") or "plagiarism"),
        "license": str(metadata.get("license") or "unknown"),
        "source_url": str(metadata.get("source_url") or metadata.get("url") or ""),
        "counts": _dataset_counts(loaded),
        "fingerprint": fingerprint_source(loaded.root),
        "metadata": _sanitize_mapping(metadata),
        "source": _sanitize_mapping(source_spec or {}),
    }


def algorithm_card(config, name=None):
    payload = dict(config or {})
    options = dict(payload.get("options") or payload.get("similarity_options") or {})
    algorithm_path = payload.get("algorithm_path") or options.pop("algorithm_path", None)
    algorithm_options = normalize_algorithm_options(payload.get("algorithm_options") or options.pop("algorithm_options", None))
    for key, value in payload.items():
        if key not in {"name", "options", "similarity_options", "algorithm_path", "algorithm_options"}:
            options[key] = value
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "card_type": "algorithm",
        "name": str(name or payload.get("name") or _algorithm_name_from_path(algorithm_path) or "matheel"),
        "algorithm_kind": "custom" if algorithm_path else "builtin",
        "algorithm_path_name": _path_name(algorithm_path) if algorithm_path else "",
        "algorithm_options": _sanitize_mapping(algorithm_options),
        "similarity_options": _sanitize_mapping(options),
        "fingerprint": {},
        "package": "",
        "package_version": "",
    }
    if algorithm_path:
        card.update(_custom_algorithm_card_fields(algorithm_path))
    return card


def leaderboard_cards(manifest):
    return {
        "datasets": [
            dataset_card(
                dataset["spec"],
                name=dataset.get("name"),
                task_family=dataset.get("task_family"),
                source_spec=dataset.get("spec"),
            )
            for dataset in manifest.get("datasets", [])
        ],
        "algorithms": [
            algorithm_card(algorithm, name=algorithm.get("name"))
            for algorithm in manifest.get("algorithms", [])
        ],
    }


def card_markdown(card):
    title = f"{card.get('card_type', 'card').title()}: {card.get('name', 'unknown')}"
    lines = [f"# {title}", ""]
    for key, value in card.items():
        if key in {"schema_version", "card_type", "name"}:
            continue
        lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _load_card_dataset(dataset, task_family=None):
    if isinstance(dataset, (PairDataset, RetrievalDataset)):
        return dataset
    task = task_family
    if task is None and isinstance(dataset, dict):
        task_families = dataset.get("task_families") or ()
        task = next(iter(task_families), None)
    if task == "retrieval":
        if isinstance(dataset, dict):
            return load_retrieval_datasets(dataset)
        return load_retrieval_dataset(_dataset_identifier(dataset))
    if isinstance(dataset, dict):
        return load_pair_datasets(dataset)
    return load_pair_dataset(_dataset_identifier(dataset))


def _dataset_identifier(dataset):
    if isinstance(dataset, dict):
        return dataset.get("identifier")
    return dataset


def _dataset_counts(dataset):
    if isinstance(dataset, RetrievalDataset):
        return {
            "files": int(len(dataset.files)),
            "queries": int(len(dataset.queries)),
            "documents": int(len(dataset.corpus)),
            "qrels": int(len(dataset.qrels)),
        }
    return {
        "files": int(len(dataset.files)),
        "pairs": int(len(dataset.pairs)),
        "positive_pairs": int(dataset.pairs["label"].sum()) if "label" in dataset.pairs else 0,
    }


def _dataset_kind_from_task(task_family):
    return "retrieval" if task_family == "retrieval" else "pair_classification"


def _algorithm_name_from_path(algorithm_path):
    if not algorithm_path:
        return None
    return Path(_path_name(algorithm_path)).stem


def _custom_algorithm_card_fields(algorithm_path):
    try:
        resolved = resolve_pair_algorithm(algorithm_path)
    except Exception:
        return {
            "fingerprint": {"source_type": "unresolved", "file_name": _path_name(algorithm_path)},
        }
    return {
        "resolved_algorithm_name": resolved.name,
        "package": resolved.package_name or "",
        "package_version": resolved.package_version or "",
        "fingerprint": dict(resolved.source_fingerprint or {}),
    }


def _sanitize_mapping(values):
    sanitized = {}
    for key, value in dict(values or {}).items():
        name = str(key)
        lowered = name.lower()
        if lowered in _CREDENTIAL_KEYS:
            sanitized[name] = "<redacted>"
        elif lowered in _PATH_KEYS and isinstance(value, (str, Path)):
            sanitized[name] = _path_name(value)
        elif isinstance(value, dict):
            sanitized[name] = _sanitize_mapping(value)
        elif isinstance(value, (list, tuple)):
            sanitized[name] = [
                _sanitize_mapping(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[name] = value
    return sanitized


def _path_name(value):
    text = str(value or "")
    windows_path = PureWindowsPath(text)
    if windows_path.drive or "\\" in text:
        return windows_path.name or text
    return Path(text).name or text
