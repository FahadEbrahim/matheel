import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from hashlib import sha1
from inspect import signature
from pathlib import Path

import pandas as pd


DATASET_TASK_TYPES = ("plagiarism",)
DATASET_KINDS = ("pair_classification", "retrieval")
DATASET_ACCESS_TYPES = ("bundled", "download", "manual", "external")
DATASET_TASK_FAMILIES = ("pair", "retrieval")
_DATASET_REGISTRY = {}
_DATASET_SOURCE_HANDLERS = {}
_DATASET_PRESETS = {}
_DATASET_ADAPTERS = {}


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


@dataclass(frozen=True)
class RetrievalDataset:
    root: Path
    files: pd.DataFrame
    queries: pd.DataFrame
    corpus: pd.DataFrame
    qrels: pd.DataFrame
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


def available_dataset_sources():
    return tuple(sorted(_DATASET_SOURCE_HANDLERS))


def register_dataset_source(name, resolver, overwrite=False):
    key = _normalize_name(name)
    if not callable(resolver):
        raise ValueError("Dataset source resolver must be callable.")
    if key in _DATASET_SOURCE_HANDLERS and not overwrite:
        raise ValueError(f"Dataset source already exists: {key}")
    _DATASET_SOURCE_HANDLERS[key] = resolver
    return resolver


def resolve_dataset_source(
    source,
    identifier,
    destination=None,
    revision="main",
    token=None,
    split=None,
):
    source_key = _normalize_name(source or "local")
    if source_key not in _DATASET_SOURCE_HANDLERS:
        supported = ", ".join(available_dataset_sources())
        raise KeyError(f"Unknown dataset source: {source_key}. Available sources: {supported}")

    resolved_destination = destination
    if resolved_destination is None and source_key != "local":
        resolved_destination = _default_dataset_destination(source_key, identifier)
    if resolved_destination is not None:
        resolved_destination = _coerce_path(resolved_destination)

    resolver = _DATASET_SOURCE_HANDLERS[source_key]
    resolved = _invoke_with_supported_kwargs(
        resolver,
        identifier=identifier,
        destination=resolved_destination,
        revision=revision,
        token=token,
        split=split,
    )
    return _coerce_path(resolved)


def resolve_dataset_path(*args, **kwargs):
    return resolve_dataset_source(*args, **kwargs)


def available_dataset_adapters():
    return tuple(sorted(_DATASET_ADAPTERS))


def register_dataset_adapter(name, adapter, overwrite=False):
    key = _normalize_name(name)
    if not callable(adapter):
        raise ValueError("Dataset adapter must be callable.")
    if key in _DATASET_ADAPTERS and not overwrite:
        raise ValueError(f"Dataset adapter already exists: {key}")
    _DATASET_ADAPTERS[key] = adapter
    return adapter


def adapt_pair_dataset(
    source_root,
    adapter="auto_pair_tabular",
    destination=None,
    dataset_name=None,
    adapter_options=None,
):
    return _adapt_dataset(
        source_root,
        dataset_kind="pair_classification",
        adapter=adapter,
        destination=destination,
        dataset_name=dataset_name,
        adapter_options=adapter_options,
    )


def adapt_retrieval_dataset(
    source_root,
    adapter="auto_retrieval_tabular",
    destination=None,
    dataset_name=None,
    adapter_options=None,
):
    return _adapt_dataset(
        source_root,
        dataset_kind="retrieval",
        adapter=adapter,
        destination=destination,
        dataset_name=dataset_name,
        adapter_options=adapter_options,
    )


def available_dataset_presets():
    return tuple(sorted(_DATASET_PRESETS))


def register_dataset_preset(name, spec, overwrite=False):
    key = _normalize_name(name)
    if not isinstance(spec, dict):
        raise ValueError("Dataset preset spec must be a dict.")
    if "identifier" not in spec:
        raise ValueError("Dataset preset spec must include an identifier.")
    if key in _DATASET_PRESETS and not overwrite:
        raise ValueError(f"Dataset preset already exists: {key}")

    payload = dict(spec)
    payload.setdefault("name", key)
    payload["task_families"] = _normalize_task_families(
        payload.get("task_families") or payload.get("task_family") or ("pair",)
    )
    _DATASET_PRESETS[key] = payload
    return get_dataset_preset(key)


def get_dataset_preset(name):
    key = _normalize_name(name)
    try:
        preset = _DATASET_PRESETS[key]
    except KeyError as exc:
        supported = ", ".join(available_dataset_presets())
        raise KeyError(f"Unknown dataset preset: {key}. Available presets: {supported}") from exc
    payload = dict(preset)
    payload["task_families"] = tuple(payload.get("task_families") or ("pair",))
    return payload


def dataset_preset_task_families(name):
    return get_dataset_preset(name)["task_families"]


def available_dataset_presets_by_task(task_family):
    requested = _normalize_task_family(task_family)
    return tuple(
        preset_name
        for preset_name in available_dataset_presets()
        if requested in dataset_preset_task_families(preset_name)
    )


def load_pair_datasets(dataset_specs):
    specs = _coerce_dataset_specs(dataset_specs)
    loaded = []
    names = []
    for spec in specs:
        _ensure_spec_supports_task(spec, "pair")
        root = _resolve_dataset_spec_root(spec)
        adapter = spec.get("adapter")
        if adapter:
            options = _adapter_options_with_split(spec)
            root = adapt_pair_dataset(
                root,
                adapter=adapter,
                destination=spec.get("adapted_destination"),
                dataset_name=spec.get("name"),
                adapter_options=options,
            )
        loaded.append(load_pair_dataset(root))
        names.append(spec["name"])

    if len(loaded) == 1:
        return loaded[0]
    return merge_pair_datasets(loaded, dataset_names=names)


def load_retrieval_datasets(dataset_specs):
    specs = _coerce_dataset_specs(dataset_specs)
    loaded = []
    names = []
    for spec in specs:
        _ensure_spec_supports_task(spec, "retrieval")
        root = _resolve_dataset_spec_root(spec)
        adapter = spec.get("retrieval_adapter") or spec.get("adapter")
        if adapter:
            options = _adapter_options_with_split(spec)
            root = adapt_retrieval_dataset(
                root,
                adapter=adapter,
                destination=spec.get("adapted_destination"),
                dataset_name=spec.get("name"),
                adapter_options=options,
            )
        loaded.append(load_retrieval_dataset(root))
        names.append(spec["name"])

    if len(loaded) == 1:
        return loaded[0]
    return merge_retrieval_datasets(loaded, dataset_names=names)


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


def load_retrieval_dataset(dataset_root):
    root = _coerce_path(dataset_root)
    metadata = _read_json(root / "metadata.json")
    files = _load_files_manifest(root)

    queries_path = root / "queries.csv"
    corpus_path = root / "corpus.csv"
    qrels_path = root / "qrels.csv"
    for label, path in (
        ("queries manifest", queries_path),
        ("corpus manifest", corpus_path),
        ("qrels manifest", qrels_path),
    ):
        if not path.exists():
            raise ValueError(f"Missing {label}: {path}")

    dataset = RetrievalDataset(
        root=root,
        files=files,
        queries=pd.read_csv(queries_path),
        corpus=pd.read_csv(corpus_path),
        qrels=pd.read_csv(qrels_path),
        metadata=metadata,
    )
    return validate_retrieval_dataset(dataset)


def write_retrieval_dataset(dataset_root, files, queries, corpus, qrels, metadata=None):
    root = _coerce_path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    files_manifest = _write_files_manifest(root, files)

    queries_frame = _coerce_frame(queries, required_columns=("query_id", "file_id"), frame_name="queries")
    corpus_frame = _coerce_frame(corpus, required_columns=("document_id", "file_id"), frame_name="corpus")
    qrels_frame = _coerce_frame(
        qrels,
        required_columns=("query_id", "document_id", "relevance"),
        frame_name="qrels",
    )

    queries_frame["query_id"] = queries_frame["query_id"].map(lambda value: _normalize_id(value, "query_id"))
    queries_frame["file_id"] = queries_frame["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    corpus_frame["document_id"] = corpus_frame["document_id"].map(
        lambda value: _normalize_id(value, "document_id")
    )
    corpus_frame["file_id"] = corpus_frame["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    qrels_frame["query_id"] = qrels_frame["query_id"].map(lambda value: _normalize_id(value, "query_id"))
    qrels_frame["document_id"] = qrels_frame["document_id"].map(
        lambda value: _normalize_id(value, "document_id")
    )
    qrels_frame["relevance"] = qrels_frame["relevance"].map(_normalize_nonnegative_relevance)

    queries_frame.to_csv(root / "queries.csv", index=False)
    corpus_frame.to_csv(root / "corpus.csv", index=False)
    qrels_frame.to_csv(root / "qrels.csv", index=False)

    payload = _normalize_metadata(metadata, dataset_kind="retrieval")
    _write_json(root / "metadata.json", payload)
    return validate_retrieval_dataset(
        RetrievalDataset(
            root=root,
            files=files_manifest,
            queries=queries_frame,
            corpus=corpus_frame,
            qrels=qrels_frame,
            metadata=payload,
        )
    )


def validate_retrieval_dataset(dataset):
    if isinstance(dataset, (str, os.PathLike)):
        dataset = load_retrieval_dataset(dataset)
    if not isinstance(dataset, RetrievalDataset):
        raise ValueError("validate_retrieval_dataset expects a RetrievalDataset or dataset path.")

    metadata = _normalize_metadata(dataset.metadata, dataset_kind="retrieval")
    if metadata["dataset_kind"] != "retrieval":
        raise ValueError("Retrieval datasets must use dataset_kind='retrieval'.")

    files = _coerce_frame(dataset.files, required_columns=("file_id", "file_path"), frame_name="files")
    queries = _coerce_frame(dataset.queries, required_columns=("query_id", "file_id"), frame_name="queries")
    corpus = _coerce_frame(dataset.corpus, required_columns=("document_id", "file_id"), frame_name="corpus")
    qrels = _coerce_frame(
        dataset.qrels,
        required_columns=("query_id", "document_id", "relevance"),
        frame_name="qrels",
    )

    files["file_id"] = files["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    files["file_path"] = files["file_path"].map(_normalize_relative_file_path)
    queries["query_id"] = queries["query_id"].map(lambda value: _normalize_id(value, "query_id"))
    queries["file_id"] = queries["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    corpus["document_id"] = corpus["document_id"].map(lambda value: _normalize_id(value, "document_id"))
    corpus["file_id"] = corpus["file_id"].map(lambda value: _normalize_id(value, "file_id"))
    qrels["query_id"] = qrels["query_id"].map(lambda value: _normalize_id(value, "query_id"))
    qrels["document_id"] = qrels["document_id"].map(lambda value: _normalize_id(value, "document_id"))
    qrels["relevance"] = qrels["relevance"].map(_normalize_nonnegative_relevance)

    duplicate_file_ids = files["file_id"][files["file_id"].duplicated()].tolist()
    if duplicate_file_ids:
        raise ValueError(
            f"files.csv contains duplicate file_id values: {', '.join(sorted(set(duplicate_file_ids)))}"
        )

    duplicate_query_ids = queries["query_id"][queries["query_id"].duplicated()].tolist()
    if duplicate_query_ids:
        raise ValueError(
            f"queries.csv contains duplicate query_id values: {', '.join(sorted(set(duplicate_query_ids)))}"
        )

    duplicate_document_ids = corpus["document_id"][corpus["document_id"].duplicated()].tolist()
    if duplicate_document_ids:
        raise ValueError(
            "corpus.csv contains duplicate document_id values: "
            f"{', '.join(sorted(set(duplicate_document_ids)))}"
        )

    duplicate_qrels = qrels[qrels.duplicated(subset=["query_id", "document_id"])]
    if not duplicate_qrels.empty:
        pairs = sorted(
            f"{row['query_id']}->{row['document_id']}"
            for row in duplicate_qrels.to_dict(orient="records")
        )
        raise ValueError(f"qrels.csv contains duplicate query/document judgments: {', '.join(pairs)}")

    file_ids = set(files["file_id"].tolist())
    missing_query_files = sorted(file_id for file_id in queries["file_id"].tolist() if file_id not in file_ids)
    if missing_query_files:
        raise ValueError(f"queries.csv references unknown file ids: {', '.join(missing_query_files)}")

    missing_corpus_files = sorted(file_id for file_id in corpus["file_id"].tolist() if file_id not in file_ids)
    if missing_corpus_files:
        raise ValueError(f"corpus.csv references unknown file ids: {', '.join(missing_corpus_files)}")

    query_ids = set(queries["query_id"].tolist())
    document_ids = set(corpus["document_id"].tolist())
    missing_queries = sorted(query_id for query_id in qrels["query_id"].tolist() if query_id not in query_ids)
    if missing_queries:
        raise ValueError(f"qrels.csv references unknown query ids: {', '.join(missing_queries)}")

    missing_documents = sorted(
        document_id for document_id in qrels["document_id"].tolist() if document_id not in document_ids
    )
    if missing_documents:
        raise ValueError(f"qrels.csv references unknown document ids: {', '.join(missing_documents)}")

    for file_path in files["file_path"].tolist():
        target = dataset.root / file_path
        if not target.exists():
            raise ValueError(f"files.csv references missing file: {file_path}")

    return RetrievalDataset(
        root=dataset.root,
        files=files,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        metadata=metadata,
    )


def load_code_texts(dataset):
    if isinstance(dataset, (str, os.PathLike)):
        root = _coerce_path(dataset)
        metadata = _read_json(root / "metadata.json")
        dataset_kind = str(metadata.get("dataset_kind") or "").strip().lower()
        if dataset_kind == "retrieval":
            dataset = load_retrieval_dataset(root)
        else:
            dataset = load_pair_dataset(root)
    if not isinstance(dataset, (PairDataset, RetrievalDataset)):
        raise ValueError("load_code_texts expects a PairDataset, RetrievalDataset, or dataset path.")

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


def _normalize_nonnegative_relevance(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Relevance values must be finite and non-negative. Got: {value}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"Relevance values must be finite and non-negative. Got: {value}")
    return numeric


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


def _normalize_task_family(value):
    key = str(value or "").strip().lower()
    if key == "pair_classification":
        key = "pair"
    if key not in DATASET_TASK_FAMILIES:
        supported = ", ".join(DATASET_TASK_FAMILIES)
        raise ValueError(f"task_family must be one of: {supported}. Got: {value}")
    return key


def _normalize_task_families(values):
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = tuple(values or ())
    families = tuple(dict.fromkeys(_normalize_task_family(value) for value in raw_values))
    if not families:
        raise ValueError("Dataset presets must support at least one task family.")
    return families


def _invoke_with_supported_kwargs(func, **kwargs):
    parameters = signature(func).parameters
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
        return func(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return func(**accepted)


def _default_dataset_destination(source, identifier):
    digest = sha1(str(identifier).encode("utf-8")).hexdigest()[:12]
    safe_identifier = _safe_name_component(identifier)[:48]
    return Path(tempfile.gettempdir()) / "matheel_datasets" / f"{source}_{safe_identifier}_{digest}"


def _safe_name_component(value):
    text = str(value or "").strip().lower()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "dataset"


def _resolve_local_dataset_source(identifier, destination=None, revision="main", token=None, split=None):
    _ = (destination, revision, token, split)
    root = _coerce_path(identifier)
    if not root.exists():
        raise ValueError(f"Local dataset source does not exist: {identifier}")
    return root


def _adapt_dataset(source_root, dataset_kind, adapter, destination, dataset_name, adapter_options):
    root = _coerce_path(source_root)
    if adapter is None:
        return root
    if dataset_kind == "pair_classification" and _is_pair_dataset_root(root):
        return root
    if dataset_kind == "retrieval" and _is_retrieval_dataset_root(root):
        return root

    adapter_name = _normalize_name(adapter)
    try:
        adapter_fn = _DATASET_ADAPTERS[adapter_name]
    except KeyError as exc:
        supported = ", ".join(available_dataset_adapters())
        raise KeyError(f"Unknown dataset adapter: {adapter_name}. Available adapters: {supported}") from exc

    target = _coerce_path(destination) if destination is not None else _temporary_dataset_root(adapter_name)
    target.mkdir(parents=True, exist_ok=True)
    options = dict(adapter_options or {})
    adapted = _invoke_with_supported_kwargs(
        adapter_fn,
        source_root=root,
        destination=target,
        dataset_name=dataset_name,
        **options,
    )
    return _coerce_path(adapted)


def _temporary_dataset_root(label):
    safe_label = _safe_name_component(label)
    return Path(tempfile.mkdtemp(prefix=f"matheel_{safe_label}_"))


def _is_pair_dataset_root(path):
    root = _coerce_path(path)
    return (root / "files.csv").exists() and (root / "pairs.csv").exists()


def _is_retrieval_dataset_root(path):
    root = _coerce_path(path)
    return all((root / name).exists() for name in ("files.csv", "queries.csv", "corpus.csv", "qrels.csv"))


def _coerce_dataset_specs(dataset_specs):
    if isinstance(dataset_specs, (str, os.PathLike, dict)):
        raw_specs = [dataset_specs]
    else:
        raw_specs = list(dataset_specs)
    if not raw_specs:
        raise ValueError("At least one dataset spec is required.")
    return [_coerce_dataset_spec(spec) for spec in raw_specs]


def _coerce_dataset_spec(spec):
    if isinstance(spec, (str, os.PathLike)):
        text = os.fspath(spec)
        preset_key = text.strip().lower()
        if preset_key in _DATASET_PRESETS:
            return _preset_to_spec(preset_key, {})
        return _explicit_dataset_spec({"source": "local", "identifier": spec})

    if isinstance(spec, dict):
        payload = dict(spec)
        preset_name = payload.pop("preset", None)
        if preset_name is None and "identifier" not in payload:
            maybe_name = str(payload.get("name") or "").strip().lower()
            if maybe_name in _DATASET_PRESETS:
                preset_name = maybe_name
        if preset_name is not None:
            return _preset_to_spec(preset_name, payload)
        return _explicit_dataset_spec(payload)

    raise ValueError("Dataset specs must be path-like values, preset names, or dicts.")


def _preset_to_spec(name, overrides):
    preset = get_dataset_preset(name)
    payload = dict(preset)
    payload.update(overrides)
    payload["task_families"] = preset["task_families"]
    return _explicit_dataset_spec(payload)


def _explicit_dataset_spec(payload):
    if "identifier" not in payload:
        raise ValueError("Dataset specs must include an identifier.")

    name = str(payload.get("name") or Path(str(payload["identifier"])).name or "dataset").strip()
    task_families = payload.get("task_families") or DATASET_TASK_FAMILIES
    return {
        "source": str(payload.get("source") or "local").strip().lower(),
        "identifier": payload["identifier"],
        "name": _safe_name_component(name),
        "revision": str(payload.get("revision") or "main"),
        "destination": payload.get("destination"),
        "token": payload.get("token"),
        "split": payload.get("split"),
        "path_in_archive": payload.get("path_in_archive"),
        "adapter": payload.get("adapter"),
        "retrieval_adapter": payload.get("retrieval_adapter"),
        "adapter_options": dict(payload.get("adapter_options") or {}),
        "adapted_destination": payload.get("adapted_destination"),
        "task_families": _normalize_task_families(task_families),
    }


def _ensure_spec_supports_task(spec, task_family):
    requested = _normalize_task_family(task_family)
    if requested not in spec["task_families"]:
        name = spec.get("name") or "dataset"
        raise ValueError(f"Dataset spec {name!r} does not support {requested} loading.")


def _resolve_dataset_spec_root(spec):
    root = resolve_dataset_source(
        spec.get("source") or "local",
        spec["identifier"],
        destination=spec.get("destination"),
        revision=spec.get("revision") or "main",
        token=spec.get("token"),
        split=spec.get("split"),
    )
    nested = spec.get("path_in_archive")
    if nested:
        root = root / _normalize_relative_file_path(nested)
    return _coerce_path(root)


def _adapter_options_with_split(spec):
    options = dict(spec.get("adapter_options") or {})
    if spec.get("split") is not None and "split" not in options:
        options["split"] = spec["split"]
    return options


def _prefixed_id(prefix, value):
    raw = _normalize_id(value, "file_id")
    return _normalize_id(f"{_safe_name_component(prefix)}__{raw}", "file_id")


def merge_pair_datasets(datasets, dataset_names=None):
    resolved_datasets = [validate_pair_dataset(dataset) for dataset in list(datasets)]
    if not resolved_datasets:
        raise ValueError("merge_pair_datasets requires at least one dataset.")

    names = list(dataset_names or [f"dataset_{index + 1}" for index in range(len(resolved_datasets))])
    if len(names) != len(resolved_datasets):
        raise ValueError("dataset_names must have the same length as datasets.")

    files = []
    pairs = []
    for dataset_name, dataset in zip(names, resolved_datasets):
        texts = load_code_texts(dataset)
        for file_row in dataset.files.to_dict(orient="records"):
            old_id = str(file_row["file_id"])
            files.append(
                {
                    "file_id": _prefixed_id(dataset_name, old_id),
                    "text": texts[old_id],
                    "suffix": Path(str(file_row["file_path"])).suffix or ".txt",
                    "dataset_name": dataset_name,
                    "original_file_id": old_id,
                }
            )
        for pair_row in dataset.pairs.to_dict(orient="records"):
            merged = dict(pair_row)
            merged["left_id"] = _prefixed_id(dataset_name, pair_row["left_id"])
            merged["right_id"] = _prefixed_id(dataset_name, pair_row["right_id"])
            merged["dataset_name"] = dataset_name
            pairs.append(merged)

    return write_pair_dataset(
        _temporary_dataset_root("merged_pair"),
        files=pd.DataFrame(files),
        pairs=pd.DataFrame(pairs),
        metadata={
            "name": "merged_pair_dataset",
            "task_type": "plagiarism",
            "dataset_kind": "pair_classification",
            "sources": names,
        },
    )


def merge_retrieval_datasets(datasets, dataset_names=None):
    resolved_datasets = [validate_retrieval_dataset(dataset) for dataset in list(datasets)]
    if not resolved_datasets:
        raise ValueError("merge_retrieval_datasets requires at least one dataset.")

    names = list(dataset_names or [f"dataset_{index + 1}" for index in range(len(resolved_datasets))])
    if len(names) != len(resolved_datasets):
        raise ValueError("dataset_names must have the same length as datasets.")

    files = []
    queries = []
    corpus = []
    qrels = []
    for dataset_name, dataset in zip(names, resolved_datasets):
        texts = load_code_texts(dataset)
        for file_row in dataset.files.to_dict(orient="records"):
            old_id = str(file_row["file_id"])
            files.append(
                {
                    "file_id": _prefixed_id(dataset_name, old_id),
                    "text": texts[old_id],
                    "suffix": Path(str(file_row["file_path"])).suffix or ".txt",
                    "dataset_name": dataset_name,
                    "original_file_id": old_id,
                }
            )
        for query_row in dataset.queries.to_dict(orient="records"):
            merged = dict(query_row)
            merged["query_id"] = _prefixed_id(dataset_name, query_row["query_id"])
            merged["file_id"] = _prefixed_id(dataset_name, query_row["file_id"])
            merged["dataset_name"] = dataset_name
            queries.append(merged)
        for corpus_row in dataset.corpus.to_dict(orient="records"):
            merged = dict(corpus_row)
            merged["document_id"] = _prefixed_id(dataset_name, corpus_row["document_id"])
            merged["file_id"] = _prefixed_id(dataset_name, corpus_row["file_id"])
            merged["dataset_name"] = dataset_name
            corpus.append(merged)
        for qrel_row in dataset.qrels.to_dict(orient="records"):
            merged = dict(qrel_row)
            merged["query_id"] = _prefixed_id(dataset_name, qrel_row["query_id"])
            merged["document_id"] = _prefixed_id(dataset_name, qrel_row["document_id"])
            merged["dataset_name"] = dataset_name
            qrels.append(merged)

    return write_retrieval_dataset(
        _temporary_dataset_root("merged_retrieval"),
        files=pd.DataFrame(files),
        queries=pd.DataFrame(queries),
        corpus=pd.DataFrame(corpus),
        qrels=pd.DataFrame(qrels),
        metadata={
            "name": "merged_retrieval_dataset",
            "task_type": "plagiarism",
            "dataset_kind": "retrieval",
            "sources": names,
        },
    )


def _find_tabular_files(root):
    base = _coerce_path(root)
    suffixes = {".csv", ".tsv", ".tab", ".json", ".jsonl"}
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _read_tabular_file(path):
    target = _coerce_path(path)
    suffix = target.suffix.lower()
    if suffix == ".tsv" or suffix == ".tab":
        return pd.read_csv(target, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(target, lines=True)
    if suffix == ".json":
        return pd.read_json(target)
    return pd.read_csv(target)


def _pick_column(columns, aliases):
    lowered = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _column_from_options(frame, options, option_name, aliases, label, required=False):
    configured = options.get(option_name)
    if configured:
        if configured not in frame.columns:
            raise ValueError(f"{label} column does not exist: {configured}")
        return configured
    column = _pick_column(frame.columns, aliases)
    if required and column is None:
        raise ValueError(f"Could not infer {label} column.")
    return column


def _prioritized_tabular_files(root, requested_table=None, split=None):
    base = _coerce_path(root)
    if requested_table:
        table_path = Path(os.fspath(requested_table))
        if not table_path.is_absolute():
            table_path = base / table_path
        return [_coerce_path(table_path)]

    candidates = _find_tabular_files(base)
    if split is None:
        return candidates

    variants = _split_name_variants(split)
    matching = [path for path in candidates if path.stem.lower() in variants or path.name.lower() in variants]
    remaining = [path for path in candidates if path not in matching]
    return matching + remaining


def _split_name_variants(split):
    key = str(split or "").strip().lower()
    if not key:
        return set()
    variants = {key, key.replace("-", "_"), key.replace("_", "-")}
    return variants.union({f"{variant}.csv" for variant in variants})


def _select_pair_table(root, options):
    for path in _prioritized_tabular_files(
        root,
        requested_table=options.get("pair_table") or options.get("table"),
        split=options.get("split"),
    ):
        frame = _read_tabular_file(path)
        try:
            _infer_pair_columns(frame, options)
        except ValueError:
            continue
        return path, frame
    raise ValueError("Could not find a tabular pair dataset with left, right, and label columns.")


def _select_retrieval_table(root, options):
    for path in _prioritized_tabular_files(
        root,
        requested_table=options.get("retrieval_table") or options.get("table"),
        split=options.get("split"),
    ):
        frame = _read_tabular_file(path)
        try:
            _infer_retrieval_columns(frame, options)
        except ValueError:
            continue
        return path, frame
    raise ValueError("Could not find a tabular retrieval dataset with query, document, and relevance columns.")


def _infer_pair_columns(frame, options):
    left_text = _column_from_options(
        frame,
        options,
        "left_text_column",
        ("left_text", "code1", "code_1", "func1", "submission_1_text", "source_1_text"),
        "left text",
    )
    right_text = _column_from_options(
        frame,
        options,
        "right_text_column",
        ("right_text", "code2", "code_2", "func2", "submission_2_text", "source_2_text"),
        "right text",
    )
    left_path = _column_from_options(
        frame,
        options,
        "left_path_column",
        ("left_path", "file1", "file_1", "path1", "submission_1", "source_1_path"),
        "left path",
    )
    right_path = _column_from_options(
        frame,
        options,
        "right_path_column",
        ("right_path", "file2", "file_2", "path2", "submission_2", "source_2_path"),
        "right path",
    )
    if left_text is None and left_path is None:
        raise ValueError("Pair table needs a left text or left path column.")
    if right_text is None and right_path is None:
        raise ValueError("Pair table needs a right text or right path column.")

    label = _column_from_options(
        frame,
        options,
        "label_column",
        ("label", "is_clone", "is_plagiarism", "plagiarism", "verdict", "target"),
        "label",
        required=True,
    )
    return {
        "left_id": _column_from_options(
            frame,
            options,
            "left_id_column",
            ("left_id", "left_file_id", "id1", "file_id_1", "submission_1_id"),
            "left id",
        ),
        "right_id": _column_from_options(
            frame,
            options,
            "right_id_column",
            ("right_id", "right_file_id", "id2", "file_id_2", "submission_2_id"),
            "right id",
        ),
        "left_text": left_text,
        "right_text": right_text,
        "left_path": left_path,
        "right_path": right_path,
        "label": label,
    }


def _infer_retrieval_columns(frame, options):
    query_text = _column_from_options(
        frame,
        options,
        "query_text_column",
        ("query_text", "query_code", "left_text", "code1", "code_1", "func1"),
        "query text",
    )
    document_text = _column_from_options(
        frame,
        options,
        "document_text_column",
        ("document_text", "doc_text", "document_code", "right_text", "code2", "code_2", "func2"),
        "document text",
    )
    query_path = _column_from_options(
        frame,
        options,
        "query_path_column",
        ("query_path", "query_file", "left_path", "file1", "file_1", "path1"),
        "query path",
    )
    document_path = _column_from_options(
        frame,
        options,
        "document_path_column",
        ("document_path", "doc_path", "document_file", "right_path", "file2", "file_2", "path2"),
        "document path",
    )
    if query_text is None and query_path is None:
        raise ValueError("Retrieval table needs a query text or query path column.")
    if document_text is None and document_path is None:
        raise ValueError("Retrieval table needs a document text or document path column.")

    relevance = _column_from_options(
        frame,
        options,
        "relevance_column",
        ("relevance", "label", "is_relevant", "is_clone", "is_plagiarism", "verdict", "target"),
        "relevance",
        required=True,
    )
    return {
        "query_id": _column_from_options(
            frame,
            options,
            "query_id_column",
            ("query_id", "qid", "left_id", "submission_1_id"),
            "query id",
        ),
        "document_id": _column_from_options(
            frame,
            options,
            "document_id_column",
            ("document_id", "doc_id", "did", "right_id", "submission_2_id"),
            "document id",
        ),
        "query_text": query_text,
        "document_text": document_text,
        "query_path": query_path,
        "document_path": document_path,
        "relevance": relevance,
    }


def _has_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _table_value(row, column):
    if column is None:
        return None
    value = row.get(column)
    return value if _has_value(value) else None


def _resolve_tabular_source_path(source_root, value):
    path = Path(os.fspath(value))
    if not path.is_absolute():
        path = _coerce_path(source_root) / path
    if not path.exists():
        raise ValueError(f"Tabular adapter source file does not exist: {value}")
    return path


def _coerce_binary_like(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y", "clone", "positive", "plagiarized", "plagiarism", "match"}:
            return 1
        if key in {
            "0",
            "false",
            "no",
            "n",
            "non-clone",
            "negative",
            "original",
            "nonplagiarized",
            "not plagiarized",
            "non_plagiarism",
            "non-match",
        }:
            return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to coerce binary label value: {value}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Unable to coerce binary label value: {value}")
    return int(numeric > 0.0)


def _coerce_relevance_like(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "yes", "y", "relevant", "positive", "plagiarized", "plagiarism", "match"}:
            return 1.0
        if key in {"false", "no", "n", "irrelevant", "negative", "non-match"}:
            return 0.0
    return _normalize_nonnegative_relevance(value)


def _external_id_or_generated(value, label, prefix):
    if _has_value(value):
        try:
            return _normalize_id(value, label)
        except ValueError:
            return _generated_id(prefix, value)
    return None


def _generated_id(prefix, value):
    digest = sha1(str(value).encode("utf-8")).hexdigest()[:16]
    return _normalize_id(f"{prefix}_{digest}", prefix)


def _add_tabular_file(files, file_ids_by_key, source_root, role, id_value, text_value, path_value, suffix):
    if _has_value(id_value):
        identity_role = "source" if role in {"left", "right"} else role
        key = ("id", identity_role, str(id_value))
        preferred_id = _generated_id(identity_role, id_value)
    elif _has_value(path_value):
        resolved_path = _resolve_tabular_source_path(source_root, path_value)
        key = ("path", resolved_path.as_posix())
        preferred_id = _generated_id(role, resolved_path.as_posix())
    else:
        key = ("text", str(text_value))
        preferred_id = _generated_id(role, text_value)

    if key in file_ids_by_key:
        return file_ids_by_key[key]

    file_id = preferred_id
    file_ids_by_key[key] = file_id
    row = {"file_id": file_id}
    if _has_value(path_value):
        source_path = _resolve_tabular_source_path(source_root, path_value)
        row["source_path"] = source_path
        row["suffix"] = source_path.suffix or suffix
    else:
        row["text"] = str(text_value)
        row["suffix"] = suffix
    files.append(row)
    return file_id


def _adapt_pair_dataset_auto(source_root, destination, dataset_name=None, **options):
    root = _coerce_path(source_root)
    _, frame = _select_pair_table(root, options)
    columns = _infer_pair_columns(frame, options)
    suffix = str(options.get("suffix") or ".txt")

    files = []
    file_ids_by_key = {}
    pairs = []
    for index, row in enumerate(frame.to_dict(orient="records")):
        left_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "left",
            _table_value(row, columns["left_id"]),
            _table_value(row, columns["left_text"]),
            _table_value(row, columns["left_path"]),
            suffix,
        )
        right_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "right",
            _table_value(row, columns["right_id"]),
            _table_value(row, columns["right_text"]),
            _table_value(row, columns["right_path"]),
            suffix,
        )
        pairs.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "label": _coerce_binary_like(row[columns["label"]]),
                "row_index": index,
            }
        )

    metadata = {
        "name": dataset_name or root.name,
        "task_type": "plagiarism",
        "dataset_kind": "pair_classification",
        "adapter": "auto_pair_tabular",
    }
    if options.get("split") is not None:
        metadata["split"] = str(options["split"])
    return write_pair_dataset(destination, files=pd.DataFrame(files), pairs=pd.DataFrame(pairs), metadata=metadata).root


def _adapt_retrieval_dataset_auto(source_root, destination, dataset_name=None, **options):
    root = _coerce_path(source_root)
    _, frame = _select_retrieval_table(root, options)
    columns = _infer_retrieval_columns(frame, options)
    suffix = str(options.get("suffix") or ".txt")

    files = []
    file_ids_by_key = {}
    queries_by_id = {}
    corpus_by_id = {}
    qrels_by_pair = {}
    for index, row in enumerate(frame.to_dict(orient="records")):
        query_identity = _table_value(row, columns["query_id"])
        document_identity = _table_value(row, columns["document_id"])
        query_file_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "query",
            query_identity,
            _table_value(row, columns["query_text"]),
            _table_value(row, columns["query_path"]),
            suffix,
        )
        document_file_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "document",
            document_identity,
            _table_value(row, columns["document_text"]),
            _table_value(row, columns["document_path"]),
            suffix,
        )
        query_id = _external_id_or_generated(query_identity, "query_id", "query")
        if query_id is None:
            query_id = _generated_id("query", query_file_id)
        document_id = _external_id_or_generated(document_identity, "document_id", "document")
        if document_id is None:
            document_id = _generated_id("document", document_file_id)

        queries_by_id.setdefault(query_id, {"query_id": query_id, "file_id": query_file_id})
        corpus_by_id.setdefault(document_id, {"document_id": document_id, "file_id": document_file_id})
        relevance = _coerce_relevance_like(row[columns["relevance"]])
        key = (query_id, document_id)
        qrels_by_pair[key] = max(relevance, qrels_by_pair.get(key, relevance))

    metadata = {
        "name": dataset_name or root.name,
        "task_type": "plagiarism",
        "dataset_kind": "retrieval",
        "adapter": "auto_retrieval_tabular",
    }
    if options.get("split") is not None:
        metadata["split"] = str(options["split"])

    qrels = [
        {"query_id": query_id, "document_id": document_id, "relevance": relevance}
        for (query_id, document_id), relevance in qrels_by_pair.items()
    ]
    return write_retrieval_dataset(
        destination,
        files=pd.DataFrame(files),
        queries=pd.DataFrame(queries_by_id.values()),
        corpus=pd.DataFrame(corpus_by_id.values()),
        qrels=pd.DataFrame(qrels),
        metadata=metadata,
    ).root


def _normalize_soco14_split(split):
    key = str(split or "test").strip().lower()
    if key not in {"train", "test"}:
        raise ValueError("SOCO14 split must be 'train' or 'test'.")
    return key


def _find_soco14_split_root(source_root, split):
    root = _coerce_path(source_root)
    candidates = [
        root / f"fire14-source-code-{split}-dataset",
        root / f"fire14-source-code-{split}-dataset_unzipped" / f"fire14-source-code-{split}-dataset",
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.qrel")):
            return candidate
    for candidate in root.rglob(f"fire14-source-code-{split}-dataset"):
        if candidate.is_dir() and any(candidate.rglob("*.qrel")):
            return candidate
    return root


def _index_source_files(root):
    indexed = {}
    for path in sorted(_coerce_path(root).rglob("*")):
        if not path.is_file() or "qrel" in path.name.lower() or path.name.startswith("."):
            continue
        indexed.setdefault(path.name, path)
        indexed.setdefault(path.stem, path)
    return indexed


def _parse_qrel_line(line):
    parts = line.split()
    if len(parts) >= 4:
        return parts[0], parts[2], _coerce_relevance_like(parts[3])
    if len(parts) >= 2:
        return parts[0], parts[1], 1.0
    return None


def _adapt_soco14_retrieval_dataset(source_root, destination, dataset_name=None, **options):
    split = _normalize_soco14_split(options.get("split"))
    split_root = _find_soco14_split_root(source_root, split)
    source_index = _index_source_files(split_root)
    qrel_files = sorted(split_root.rglob("*.qrel"))
    if not qrel_files:
        raise ValueError("SOCO14 source root does not contain qrel files.")

    files = []
    file_ids_by_path = {}
    queries_by_id = {}
    corpus_by_id = {}
    qrels = []
    skipped = 0

    def file_id_for(path):
        resolved = _coerce_path(path)
        if resolved not in file_ids_by_path:
            file_id = _generated_id("soco14", resolved.as_posix())
            file_ids_by_path[resolved] = file_id
            files.append({"file_id": file_id, "source_path": resolved, "suffix": resolved.suffix or ".txt"})
        return file_ids_by_path[resolved]

    for qrel_file in qrel_files:
        for line in qrel_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = _parse_qrel_line(line.strip())
            if parsed is None:
                continue
            query_token, document_token, relevance = parsed
            query_path = source_index.get(query_token) or source_index.get(Path(query_token).name)
            document_path = source_index.get(document_token) or source_index.get(Path(document_token).name)
            if query_path is None or document_path is None:
                skipped += 1
                continue

            query_id = _external_id_or_generated(query_token, "query_id", "query")
            document_id = _external_id_or_generated(document_token, "document_id", "document")
            queries_by_id.setdefault(
                query_id,
                {"query_id": query_id, "file_id": file_id_for(query_path), "source_id": query_token},
            )
            corpus_by_id.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "file_id": file_id_for(document_path),
                    "source_id": document_token,
                },
            )
            qrels.append({"query_id": query_id, "document_id": document_id, "relevance": relevance})

    if not qrels:
        raise ValueError("SOCO14 adapter found no qrels with matching source files.")

    metadata = {
        "name": dataset_name or "soco14",
        "task_type": "plagiarism",
        "dataset_kind": "retrieval",
        "adapter": "soco14_retrieval",
        "split": split,
        "qrel_rows_skipped_missing_files": skipped,
    }
    return write_retrieval_dataset(
        destination,
        files=pd.DataFrame(files),
        queries=pd.DataFrame(queries_by_id.values()),
        corpus=pd.DataFrame(corpus_by_id.values()),
        qrels=pd.DataFrame(qrels),
        metadata=metadata,
    ).root


def _normalize_irplag_source_identifier(value):
    text = str(value or "").strip().replace("\\", "/")
    cleaned = "/".join(part for part in text.split("/") if part and part != ".")
    return cleaned.lower() or text.lower()


def _find_irplag_dataset_root(source_root):
    root = _coerce_path(source_root)
    direct = root / "IR-Plag-Dataset"
    if direct.exists() and direct.is_dir():
        return direct
    for candidate in root.rglob("IR-Plag-Dataset"):
        if candidate.is_dir():
            return candidate
    for archive in sorted(root.rglob("IR-Plag-Dataset.zip")):
        extracted = root / "irplag_extracted"
        _safe_extract_zip(archive, extracted)
        extracted_root = extracted / "IR-Plag-Dataset"
        if extracted_root.exists():
            return extracted_root
    return None


def _build_irplag_case_tables(irplag_root):
    files = []
    pairs = []
    queries = {}
    corpus = {}
    qrels = []
    file_ids_by_path = {}

    def file_id_for(path):
        resolved = _coerce_path(path)
        if resolved not in file_ids_by_path:
            file_id = _generated_id("irplag", resolved.as_posix())
            file_ids_by_path[resolved] = file_id
            files.append({"file_id": file_id, "source_path": resolved, "suffix": resolved.suffix or ".java"})
        return file_ids_by_path[resolved]

    for case_dir in sorted(path for path in _coerce_path(irplag_root).glob("case-*") if path.is_dir()):
        originals = sorted((case_dir / "original").rglob("*.java"))
        if not originals:
            continue
        original_path = originals[0]
        original_key = f"{case_dir.name}/original/{original_path.name}"
        original_file_id = file_id_for(original_path)
        document_id = _generated_id("document", original_key)
        corpus.setdefault(
            document_id,
            {"document_id": document_id, "file_id": original_file_id, "source_id": original_key},
        )

        for label, folder in ((0, "non-plagiarized"), (1, "plagiarized")):
            for submission_path in sorted((case_dir / folder).rglob("*.java")):
                relative = submission_path.relative_to(case_dir).as_posix()
                submission_key = f"{case_dir.name}/{relative}"
                submission_file_id = file_id_for(submission_path)
                pairs.append(
                    {
                        "left_id": original_file_id,
                        "right_id": submission_file_id,
                        "label": label,
                        "case_id": case_dir.name,
                    }
                )
                query_id = _generated_id("query", submission_key)
                queries.setdefault(
                    query_id,
                    {"query_id": query_id, "file_id": submission_file_id, "source_id": submission_key},
                )
                qrels.append({"query_id": query_id, "document_id": document_id, "relevance": float(label)})

    if not pairs:
        return None
    return files, pairs, list(queries.values()), list(corpus.values()), qrels


def _adapt_irplag_pair_dataset(source_root, destination, dataset_name=None, **options):
    root = _coerce_path(source_root)
    irplag_root = _find_irplag_dataset_root(root)
    if irplag_root is not None:
        tables = _build_irplag_case_tables(irplag_root)
        if tables is not None:
            files, pairs, _, _, _ = tables
            return write_pair_dataset(
                destination,
                files=pd.DataFrame(files),
                pairs=pd.DataFrame(pairs),
                metadata={
                    "name": dataset_name or "irplag",
                    "task_type": "plagiarism",
                    "dataset_kind": "pair_classification",
                    "adapter": "irplag_pair",
                },
            ).root

    adapter_options = dict(options)
    adapter_options.setdefault("label_column", "label")
    adapted_root = _adapt_pair_dataset_auto(
        root,
        destination,
        dataset_name=dataset_name or "irplag",
        **adapter_options,
    )
    metadata = _read_json(Path(adapted_root) / "metadata.json")
    metadata["adapter"] = "irplag_pair"
    _write_json(Path(adapted_root) / "metadata.json", metadata)
    return adapted_root


def _adapt_irplag_retrieval_dataset(source_root, destination, dataset_name=None, **options):
    root = _coerce_path(source_root)
    irplag_root = _find_irplag_dataset_root(root)
    if irplag_root is not None:
        tables = _build_irplag_case_tables(irplag_root)
        if tables is not None:
            files, _, queries, corpus, qrels = tables
            return write_retrieval_dataset(
                destination,
                files=pd.DataFrame(files),
                queries=pd.DataFrame(queries),
                corpus=pd.DataFrame(corpus),
                qrels=pd.DataFrame(qrels),
                metadata={
                    "name": dataset_name or "irplag",
                    "task_type": "plagiarism",
                    "dataset_kind": "retrieval",
                    "adapter": "irplag_retrieval",
                },
            ).root

    table_path, frame = _select_pair_table(root, options)
    _ = table_path
    columns = _infer_pair_columns(frame, options)
    files = []
    file_ids_by_key = {}
    queries_by_id = {}
    corpus_by_id = {}
    qrels = []
    for row in frame.to_dict(orient="records"):
        left_source_id = _normalize_irplag_source_identifier(
            _table_value(row, columns["left_id"]) or _table_value(row, columns["left_path"])
        )
        right_source_id = _normalize_irplag_source_identifier(
            _table_value(row, columns["right_id"]) or _table_value(row, columns["right_path"])
        )
        query_file_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "query",
            left_source_id,
            _table_value(row, columns["left_text"]),
            _table_value(row, columns["left_path"]),
            ".txt",
        )
        document_file_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "document",
            right_source_id,
            _table_value(row, columns["right_text"]),
            _table_value(row, columns["right_path"]),
            ".txt",
        )
        query_id = _generated_id("query", left_source_id)
        document_id = _generated_id("document", right_source_id)
        queries_by_id.setdefault(
            query_id,
            {"query_id": query_id, "file_id": query_file_id, "source_id": left_source_id},
        )
        corpus_by_id.setdefault(
            document_id,
            {"document_id": document_id, "file_id": document_file_id, "source_id": right_source_id},
        )
        qrels.append(
            {
                "query_id": query_id,
                "document_id": document_id,
                "relevance": float(_coerce_binary_like(row[columns["label"]])),
            }
        )

    return write_retrieval_dataset(
        destination,
        files=pd.DataFrame(files),
        queries=pd.DataFrame(queries_by_id.values()),
        corpus=pd.DataFrame(corpus_by_id.values()),
        qrels=pd.DataFrame(qrels),
        metadata={
            "name": dataset_name or "irplag",
            "task_type": "plagiarism",
            "dataset_kind": "retrieval",
            "adapter": "irplag_retrieval",
        },
    ).root


def _find_conplag_versions_root(source_root):
    root = _coerce_path(source_root)
    direct = root / "versions"
    if direct.exists():
        return direct
    for candidate in root.rglob("versions"):
        if candidate.is_dir():
            return candidate
    return None


def _find_conplag_submission_file(versions_root, problem, submission):
    base = _coerce_path(versions_root)
    problem_text = str(problem).strip()
    submission_text = str(submission).strip()
    search_roots = []
    if problem_text:
        search_roots.append(base / f"version_{problem_text}")
    search_roots.append(base)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches = sorted(search_root.rglob(f"{submission_text}.*"))
        if matches:
            return matches[0]
    return None


def _adapt_conplag_pair_dataset(source_root, destination, dataset_name=None, **options):
    root = _coerce_path(source_root)
    versions_root = _find_conplag_versions_root(root)
    label_table = options.get("label_table") or "labels.csv"
    labels_path = None
    if versions_root is not None:
        candidate = versions_root / label_table
        if candidate.exists():
            labels_path = candidate
    if labels_path is None:
        return _adapt_pair_dataset_auto(root, destination, dataset_name=dataset_name or "conplag", **options)

    labels = pd.read_csv(labels_path)
    left_col = _column_from_options(labels, options, "left_id_column", ("sub1", "left_id"), "left id", required=True)
    right_col = _column_from_options(labels, options, "right_id_column", ("sub2", "right_id"), "right id", required=True)
    problem_col = _column_from_options(labels, options, "problem_column", ("problem", "version"), "problem")
    label_col = _column_from_options(labels, options, "label_column", ("verdict", "label"), "label", required=True)

    files = []
    file_ids_by_key = {}
    pairs = []
    for row in labels.to_dict(orient="records"):
        problem = row.get(problem_col) if problem_col else ""
        left_path = _find_conplag_submission_file(versions_root, problem, row[left_col])
        right_path = _find_conplag_submission_file(versions_root, problem, row[right_col])
        if left_path is None or right_path is None:
            continue
        left_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "left",
            f"{problem}:{row[left_col]}",
            None,
            left_path,
            ".java",
        )
        right_id = _add_tabular_file(
            files,
            file_ids_by_key,
            root,
            "right",
            f"{problem}:{row[right_col]}",
            None,
            right_path,
            ".java",
        )
        pairs.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "label": _coerce_binary_like(row[label_col]),
                "problem": problem,
            }
        )

    if not pairs:
        raise ValueError("ConPlag adapter found no labeled pairs with matching source files.")
    return write_pair_dataset(
        destination,
        files=pd.DataFrame(files),
        pairs=pd.DataFrame(pairs),
        metadata={
            "name": dataset_name or "conplag",
            "task_type": "plagiarism",
            "dataset_kind": "pair_classification",
            "adapter": "conplag_pair",
        },
    ).root


def register_default_dataset_adapters(overwrite=False):
    register_dataset_adapter("auto_pair_tabular", _adapt_pair_dataset_auto, overwrite=overwrite)
    register_dataset_adapter("auto_retrieval_tabular", _adapt_retrieval_dataset_auto, overwrite=overwrite)
    register_dataset_adapter("soco14_retrieval", _adapt_soco14_retrieval_dataset, overwrite=overwrite)
    register_dataset_adapter("irplag_pair", _adapt_irplag_pair_dataset, overwrite=overwrite)
    register_dataset_adapter("irplag_retrieval", _adapt_irplag_retrieval_dataset, overwrite=overwrite)
    register_dataset_adapter("conplag_pair", _adapt_conplag_pair_dataset, overwrite=overwrite)


def _is_relative_to(path, parent):
    try:
        _coerce_path(path).relative_to(_coerce_path(parent))
    except ValueError:
        return False
    return True


def _safe_extract_zip(archive_path, output_dir):
    archive = _coerce_path(archive_path)
    destination = _coerce_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as archive_file:
        for member in archive_file.infolist():
            target = destination / member.filename
            if not _is_relative_to(target, destination):
                raise ValueError(f"Archive member would escape output directory: {member.filename}")
        archive_file.extractall(destination)
    return destination


def _safe_extract_tar(archive_path, output_dir):
    archive = _coerce_path(archive_path)
    destination = _coerce_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as archive_file:
        for member in archive_file.getmembers():
            target = destination / member.name
            if not _is_relative_to(target, destination):
                raise ValueError(f"Archive member would escape output directory: {member.name}")
            if member.issym() or member.islnk():
                link_target = target.parent / member.linkname
                if not _is_relative_to(link_target, destination):
                    raise ValueError(f"Archive link would escape output directory: {member.name}")
        archive_file.extractall(destination)
    return destination


def _safe_extract_archive(archive_path, output_dir):
    archive = _coerce_path(archive_path)
    if zipfile.is_zipfile(archive):
        return _safe_extract_zip(archive, output_dir)
    if tarfile.is_tarfile(archive):
        return _safe_extract_tar(archive, output_dir)
    raise ValueError(f"Unsupported archive format: {archive.name}")


def _single_child_directory_or_self(path):
    root = _coerce_path(path)
    children = [child for child in root.iterdir() if child.is_dir()]
    files = [child for child in root.iterdir() if child.is_file()]
    if len(children) == 1 and not files:
        return children[0]
    return root


def _download_url_to_file(url, output_file, headers=None):
    target = _coerce_path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def _read_json_url(url, headers=None):
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _authorization_headers(token):
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _resolve_github_dataset_source(identifier, destination, revision="main", token=None, split=None):
    _ = split
    repo = str(identifier).strip().rstrip("/")
    prefix = "https://github.com/"
    if repo.startswith(prefix):
        repo = repo[len(prefix) :]
    if repo.endswith(".git"):
        repo = repo[:-4]
    parts = [part for part in repo.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub dataset identifiers must use 'owner/repo'.")
    owner_repo = "/".join(parts[:2])

    destination = _coerce_path(destination)
    if destination.exists() and any(destination.iterdir()):
        return _single_child_directory_or_self(destination)
    destination.mkdir(parents=True, exist_ok=True)

    revision_path = urllib.parse.quote(str(revision or "main"), safe="")
    archive_url = f"https://github.com/{owner_repo}/archive/{revision_path}.zip"
    archive_path = destination / "github_archive.zip"
    _download_url_to_file(archive_url, archive_path, headers=_authorization_headers(token))
    _safe_extract_zip(archive_path, destination)
    archive_path.unlink(missing_ok=True)
    return _single_child_directory_or_self(destination)


def _resolve_zenodo_dataset_source(identifier, destination, revision="main", token=None, split=None):
    _ = (revision, split)
    record_id = str(identifier).strip()
    destination = _coerce_path(destination)
    if destination.exists() and any(destination.iterdir()):
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    record = _read_json_url(
        f"https://zenodo.org/api/records/{urllib.parse.quote(record_id)}",
        headers=_authorization_headers(token),
    )
    for file_info in record.get("files", []):
        key = Path(str(file_info.get("key") or file_info.get("filename") or "download")).name
        links = file_info.get("links") or {}
        file_url = links.get("self") or links.get("download")
        if not file_url:
            continue
        output_file = _download_url_to_file(
            file_url,
            destination / key,
            headers=_authorization_headers(token),
        )
        if zipfile.is_zipfile(output_file) or tarfile.is_tarfile(output_file):
            _safe_extract_archive(output_file, destination / output_file.stem)
    return destination


def _resolve_huggingface_dataset_source(identifier, destination, revision="main", token=None, split=None):
    _ = split
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset resolution requires the optional 'huggingface_hub' package."
        ) from exc

    destination = _coerce_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return _invoke_with_supported_kwargs(
        snapshot_download,
        repo_id=str(identifier),
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=destination,
    )


def _resolve_kaggle_dataset_source(identifier, destination, revision="main", token=None, split=None):
    _ = (revision, token, split)
    destination = _coerce_path(destination)
    if destination.exists() and any(destination.iterdir()):
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        kaggle_binary = shutil.which("kaggle")
        if kaggle_binary is None:
            raise ImportError("Kaggle dataset resolution requires the optional Kaggle API or CLI.")
        subprocess.run(
            [
                kaggle_binary,
                "datasets",
                "download",
                "-d",
                str(identifier),
                "-p",
                os.fspath(destination),
                "--unzip",
            ],
            check=True,
        )
        return destination

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(str(identifier), path=os.fspath(destination), unzip=True)
    return destination


def register_default_dataset_sources(overwrite=False):
    register_dataset_source("local", _resolve_local_dataset_source, overwrite=overwrite)
    register_dataset_source("github", _resolve_github_dataset_source, overwrite=overwrite)
    register_dataset_source("zenodo", _resolve_zenodo_dataset_source, overwrite=overwrite)
    register_dataset_source("huggingface", _resolve_huggingface_dataset_source, overwrite=overwrite)
    register_dataset_source("kaggle", _resolve_kaggle_dataset_source, overwrite=overwrite)


def register_default_dataset_presets(overwrite=False):
    register_dataset_preset(
        "soco14",
        {
            "source": "zenodo",
            "identifier": "7433031",
            "retrieval_adapter": "soco14_retrieval",
            "task_families": ("retrieval",),
            "url": "https://zenodo.org/records/7433031",
            "notes": "SOCO14 source-code plagiarism retrieval benchmark.",
        },
        overwrite=overwrite,
    )
    register_dataset_preset(
        "irplag",
        {
            "source": "github",
            "identifier": "oscarkarnalim/sourcecodeplagiarismdataset",
            "revision": "master",
            "adapter": "irplag_pair",
            "retrieval_adapter": "irplag_retrieval",
            "task_families": ("pair", "retrieval"),
            "url": "https://github.com/oscarkarnalim/sourcecodeplagiarismdataset",
            "notes": "IRPlag source-code plagiarism dataset.",
        },
        overwrite=overwrite,
    )
    register_dataset_preset(
        "conplag",
        {
            "source": "zenodo",
            "identifier": "7332790",
            "adapter": "conplag_pair",
            "task_families": ("pair",),
            "url": "https://zenodo.org/records/7332790",
            "notes": "ConPlag source-code plagiarism pair-classification benchmark.",
        },
        overwrite=overwrite,
    )


register_default_dataset_sources(overwrite=True)
register_default_dataset_adapters(overwrite=True)
register_default_dataset_presets(overwrite=True)
