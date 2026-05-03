import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATASET_TASK_TYPES = ("plagiarism",)
DATASET_KINDS = ("pair_classification", "retrieval")
DATASET_ACCESS_TYPES = ("bundled", "download", "manual", "external")
_DATASET_REGISTRY = {}


@dataclass(frozen=True)
class DatasetRegistryEntry:
    name: str
    task_type: str
    dataset_kind: str
    languages: tuple[str, ...]
    license: str
    source_url: str
    access: str
    citation: str = ""
    notes: str = ""


@dataclass(frozen=True)
class PairDataset:
    root: Path
    files: pd.DataFrame
    pairs: pd.DataFrame
    metadata: dict


def available_dataset_task_types():
    return DATASET_TASK_TYPES


def available_dataset_kinds():
    return DATASET_KINDS


def registered_datasets(task_type=None, dataset_kind=None):
    entries = tuple(_DATASET_REGISTRY[name] for name in sorted(_DATASET_REGISTRY))
    if task_type is not None:
        requested_task = _normalize_choice(task_type, DATASET_TASK_TYPES, "task_type")
        entries = tuple(entry for entry in entries if entry.task_type == requested_task)
    if dataset_kind is not None:
        requested_kind = _normalize_choice(dataset_kind, DATASET_KINDS, "dataset_kind")
        entries = tuple(entry for entry in entries if entry.dataset_kind == requested_kind)
    return entries


def get_dataset_entry(name):
    key = _normalize_name(name)
    try:
        return _DATASET_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown registered dataset: {name}") from exc


def register_dataset_entry(
    name,
    task_type,
    dataset_kind,
    languages=(),
    license="unknown",
    source_url="",
    access="manual",
    citation="",
    notes="",
    overwrite=False,
):
    key = _normalize_name(name)
    if key in _DATASET_REGISTRY and not overwrite:
        raise ValueError(f"Dataset registry entry already exists: {key}")
    entry = DatasetRegistryEntry(
        name=key,
        task_type=_normalize_choice(task_type, DATASET_TASK_TYPES, "task_type"),
        dataset_kind=_normalize_choice(dataset_kind, DATASET_KINDS, "dataset_kind"),
        languages=tuple(str(language).strip().lower() for language in (languages or ()) if str(language).strip()),
        license=str(license or "unknown").strip() or "unknown",
        source_url=str(source_url or "").strip(),
        access=_normalize_choice(access, DATASET_ACCESS_TYPES, "access"),
        citation=str(citation or "").strip(),
        notes=str(notes or "").strip(),
    )
    _DATASET_REGISTRY[key] = entry
    return entry


def load_pair_dataset(dataset_root):
    root = _coerce_path(dataset_root)
    metadata = _read_json(root / "metadata.json")
    files = _load_files_manifest(root)
    pairs_path = root / "pairs.csv"
    if not pairs_path.exists():
        raise ValueError(f"Missing pairs manifest: {pairs_path}")
    pairs = pd.read_csv(pairs_path)
    dataset = PairDataset(root=root, files=files, pairs=pairs, metadata=metadata)
    return validate_pair_dataset(dataset)


def write_pair_dataset(dataset_root, files, pairs, metadata=None):
    root = _coerce_path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    files_manifest = _write_files_manifest(root, files)
    pairs_frame = _coerce_frame(pairs, required_columns=("left_id", "right_id", "label"), frame_name="pairs")
    pairs_frame["left_id"] = pairs_frame["left_id"].map(lambda value: _normalize_id(value, "left_id"))
    pairs_frame["right_id"] = pairs_frame["right_id"].map(lambda value: _normalize_id(value, "right_id"))
    pairs_frame["label"] = pairs_frame["label"].map(_normalize_binary_label)
    pairs_frame.to_csv(root / "pairs.csv", index=False)
    payload = _normalize_metadata(metadata, dataset_kind="pair_classification")
    _write_json(root / "metadata.json", payload)
    return validate_pair_dataset(
        PairDataset(root=root, files=files_manifest, pairs=pairs_frame, metadata=payload)
    )


def validate_pair_dataset(dataset):
    if isinstance(dataset, (str, os.PathLike)):
        dataset = load_pair_dataset(dataset)
    if not isinstance(dataset, PairDataset):
        raise ValueError("validate_pair_dataset expects a PairDataset or dataset path.")

    metadata = _normalize_metadata(dataset.metadata, dataset_kind="pair_classification")
    if metadata["dataset_kind"] != "pair_classification":
        raise ValueError("Pair datasets must use dataset_kind='pair_classification'.")

    files = _coerce_frame(dataset.files, required_columns=("file_id", "file_path"), frame_name="files")
    pairs = _coerce_frame(dataset.pairs, required_columns=("left_id", "right_id", "label"), frame_name="pairs")
    files["file_id"] = files["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    files["file_path"] = files["file_path"].map(_normalize_relative_file_path)
    pairs["left_id"] = pairs["left_id"].map(lambda value: _normalize_id(value, "left_id"))
    pairs["right_id"] = pairs["right_id"].map(lambda value: _normalize_id(value, "right_id"))
    pairs["label"] = pairs["label"].map(_normalize_binary_label)

    duplicate_ids = files["file_id"][files["file_id"].duplicated()].tolist()
    if duplicate_ids:
        raise ValueError(f"files.csv contains duplicate file_id values: {', '.join(sorted(set(duplicate_ids)))}")

    file_ids = set(files["file_id"].tolist())
    missing = sorted(
        {file_id for file_id in pairs["left_id"].tolist() + pairs["right_id"].tolist() if file_id not in file_ids}
    )
    if missing:
        raise ValueError(f"pairs.csv references unknown file ids: {', '.join(missing)}")

    for file_path in files["file_path"].tolist():
        target = dataset.root / file_path
        if not target.exists():
            raise ValueError(f"files.csv references missing file: {file_path}")

    return PairDataset(root=dataset.root, files=files, pairs=pairs, metadata=metadata)


def load_code_texts(dataset):
    if isinstance(dataset, (str, os.PathLike)):
        dataset = load_pair_dataset(dataset)
    if not isinstance(dataset, PairDataset):
        raise ValueError("load_code_texts expects a PairDataset or dataset path.")

    texts = {}
    for row in dataset.files.to_dict(orient="records"):
        file_id = _normalize_id(row["file_id"], "file_id")
        file_path = _normalize_relative_file_path(row["file_path"])
        texts[file_id] = (dataset.root / file_path).read_text(encoding="utf-8", errors="ignore")
    return texts


def _normalize_name(value):
    name = str(value or "").strip().lower()
    if not name:
        raise ValueError("Dataset registry names must be non-empty.")
    if any(separator in name for separator in ("/", "\\")):
        raise ValueError(f"Dataset registry names must not contain path separators. Got: {value}")
    return name


def _normalize_choice(value, choices, label):
    key = str(value or "").strip().lower()
    if key not in choices:
        supported = ", ".join(choices)
        raise ValueError(f"{label} must be one of: {supported}. Got: {value}")
    return key


def _normalize_metadata(metadata, dataset_kind):
    payload = dict(metadata or {})
    payload["task_type"] = _normalize_choice(payload.get("task_type") or "plagiarism", DATASET_TASK_TYPES, "task_type")
    payload["dataset_kind"] = _normalize_choice(
        payload.get("dataset_kind") or dataset_kind,
        DATASET_KINDS,
        "dataset_kind",
    )
    return payload


def _coerce_path(path):
    return Path(path).expanduser().resolve()


def _read_json(path):
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _coerce_frame(records, required_columns, frame_name):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")
    return frame.copy()


def _normalize_id(value, label):
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    if "/" in text or "\\" in text:
        raise ValueError(f"{label} must not contain path separators. Got: {value}")
    return text


def _normalize_relative_file_path(value):
    text = str(value or "").strip()
    if not text:
        raise ValueError("file_path must be non-empty.")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"file_path must be relative to the dataset root. Got: {value}")
    return path.as_posix()


def _normalize_binary_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y", "positive", "plagiarized", "plagiarism", "match"}:
            return 1
        if key in {"0", "false", "no", "n", "negative", "original", "non_plagiarism", "non-match"}:
            return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Binary labels must be 0 or 1. Got: {value}") from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"Binary labels must be 0 or 1. Got: {value}")
    return int(numeric)


def _output_suffix(row):
    for column in ("suffix", "extension"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            suffix = str(value).strip()
            return suffix if suffix.startswith(".") else f".{suffix}"
    source_path = row.get("source_path")
    if pd.notna(source_path) and str(source_path).strip():
        return Path(str(source_path)).suffix or ".txt"
    return ".txt"


def _write_files_manifest(dataset_root, files):
    files_frame = _coerce_frame(files, required_columns=("file_id",), frame_name="files")
    if "text" not in files_frame.columns and "source_path" not in files_frame.columns:
        raise ValueError("files must include either a text column or a source_path column.")

    files_dir = dataset_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_ids = set()
    for row in files_frame.to_dict(orient="records"):
        file_id = _normalize_id(row.get("file_id"), "file_id")
        if file_id in seen_ids:
            raise ValueError(f"Duplicate file_id in files manifest: {file_id}")
        seen_ids.add(file_id)

        relative_path = Path("files") / f"{file_id}{_output_suffix(row)}"
        output_path = dataset_root / relative_path
        text = row.get("text")
        source_path = row.get("source_path")
        if pd.notna(text):
            output_path.write_text(str(text), encoding="utf-8")
        elif pd.notna(source_path):
            shutil.copyfile(os.fspath(source_path), output_path)
        else:
            raise ValueError(f"File entry for {file_id} must include text or source_path.")

        manifest = {
            key: value
            for key, value in row.items()
            if key not in {"text", "source_path", "suffix", "extension"}
        }
        manifest["file_id"] = file_id
        manifest["file_path"] = relative_path.as_posix()
        rows.append(manifest)

    manifest_frame = pd.DataFrame(rows).sort_values(by="file_id", ignore_index=True)
    manifest_frame.to_csv(dataset_root / "files.csv", index=False)
    return manifest_frame


def _load_files_manifest(dataset_root):
    files_path = Path(dataset_root) / "files.csv"
    if not files_path.exists():
        raise ValueError(f"Missing files manifest: {files_path}")
    return pd.read_csv(files_path)
