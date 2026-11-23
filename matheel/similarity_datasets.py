"""Local adapters for controlled code similarity and clone benchmark data."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import (
    _normalize_id,
    _resolve_dataset_file,
    normalize_dataset_task_type,
    register_dataset_adapter,
    register_dataset_entry,
    write_pair_dataset,
    write_retrieval_dataset,
)

SIMILCODE_URL = "https://doi.org/10.5281/zenodo.23949390"
POJ104_URL = "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104"
XLCoST_URL = "https://github.com/reddy-lab-code-research/XLCoST"
CODEXGLUE_BCB_URL = "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench"

SIMILCODE_CATEGORIES = ("identical", "functional", "structural", "different")
SIMILCODE_CLONE_TYPES = ("type1", "type2", "type3", "type4", "unrelated")
_CODEXGLUE_TERMS = "Computational Use of Data Agreement (C-UDA) and upstream source terms"
_SPLITS = {"train": "train", "validation": "valid", "test": "test"}
_POJ104_GROUP_COUNTS = {"train": 64, "validation": 16, "test": 24}
_BCB_COUNTS = {"train": 901028, "validation": 415416, "test": 415416}


def register_similarity_datasets(overwrite=False):
    """Expose manual data acquisitions and their adapters through the existing registry."""
    specs = (
        ("similcode", "similcode_pair", "code_similarity", "pair_classification",
         ("java", "csharp"), "CC BY 4.0", SIMILCODE_URL, adapt_similcode),
        ("poj104", "poj104_retrieval", "code_similarity", "retrieval",
         ("c", "cpp"), _CODEXGLUE_TERMS, POJ104_URL, adapt_poj104),
        ("xlcost", "xlcost_retrieval", "code_similarity", "retrieval",
         ("java", "python"), "Review archive and upstream source terms", XLCoST_URL, adapt_xlcost),
        ("codexglue_bcb", "codexglue_bcb_pair", "code_clone", "pair_classification",
         ("java",), _CODEXGLUE_TERMS, CODEXGLUE_BCB_URL, adapt_codexglue_bcb),
    )
    for name, adapter, domain, kind, languages, terms, url, function in specs:
        register_dataset_adapter(adapter, function, overwrite=overwrite)
        register_dataset_entry(
            name, domain, kind, languages=languages, license=terms, source_url=url, access="manual",
            notes=f"Use the {adapter} adapter with local source data.", overwrite=overwrite,
        )


def _dataset_metadata(name, default, domain, task_type, url, license, source_revision):
    if task_type is not None and normalize_dataset_task_type(task_type) != domain:
        raise ValueError(f"{default} requires task_type={domain!r}.")
    return {
        "name": name or default, "task_type": domain, "source_url": url, "license": license,
        "source_revision": str(source_revision or "local-unversioned"),
    }


def _split_name(split):
    split = str(split).lower().strip()
    split = "validation" if split in {"dev", "val", "valid"} else split
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test.")
    return split


def _select_input(source, names, *, required=True):
    source = Path(source).resolve()
    names = tuple(names)
    if source.is_file() and source.name in names:
        return source
    direct = [source / name for name in names if (source / name).is_file()]
    candidates = direct or [path for name in names for path in source.rglob(name) if path.is_file()]
    if len(candidates) == 1:
        return _resolve_dataset_file(source, candidates[0].relative_to(source))
    if not candidates and not required:
        return None
    reason = "Ambiguous inputs" if candidates else "Missing input"
    raise ValueError(f"{reason} for {', '.join(names)}.")


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _csv_records(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing required columns: {', '.join(columns)}.")
        yield from reader


def _json_records(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number} must contain a JSON object.")
                yield row


def adapt_similcode(source_root, destination, dataset_name=None, *, language=None,
                   label_view="related", task_type=None, source_revision="v2.4.0"):
    """Import the 120 original SimilCode cases and separately published clone types.

    The binary label is a documented view of the original category. The default
    distinguishes related examples from the unrelated control. In the optional
    copies-versus-independent view, Type 1–3 are positive and Type 4 is negative.
    """
    if label_view not in {"related", "copies_vs_independent"}:
        raise ValueError("label_view must be related or copies_vs_independent.")
    aliases = {"java": "java", "c#": "csharp", "cs": "csharp", "csharp": "csharp"}
    requested = aliases.get(str(language).lower().strip()) if language is not None else None
    if language is not None and requested is None:
        raise ValueError("SimilCode language must be java or csharp.")
    metadata = _dataset_metadata(dataset_name, "similcode", "code_similarity", task_type,
                                 SIMILCODE_URL, "CC BY 4.0", source_revision)
    metadata_path = _select_input(source_root, ("metadata.csv",))
    types_path = _select_input(metadata_path.parent, ("tipos_clon.csv",))
    types = {}
    for row in _csv_records(types_path, ("case_id", "language", "reference_group", "category", "clone_type")):
        case_id = _normalize_id(row["case_id"], "case_id")
        if case_id in types or row["clone_type"] not in SIMILCODE_CLONE_TYPES:
            raise ValueError(f"Duplicate case or invalid source clone type in tipos_clon.csv: {case_id}.")
        types[case_id] = row
    files, pairs, known_cases = {}, [], set()
    required = ("case_id", "language", "reference_group", "similarity_category", "code_a_id", "code_a_path",
                "code_b_id", "code_b_path", "expected_similarity_min", "expected_similarity_max")
    for row in _csv_records(metadata_path, required):
        case_id = _normalize_id(row["case_id"], "case_id")
        category = row["similarity_category"]
        row_language = aliases.get(row["language"].lower().strip())
        if case_id in known_cases or category not in SIMILCODE_CATEGORIES or row_language is None:
            raise ValueError(f"Duplicate case or invalid SimilCode category/language: {case_id}.")
        known_cases.add(case_id)
        reference = types.get(case_id)
        if reference is None or any(reference[key] != row[key] for key in ("language", "reference_group")) or reference["category"] != category:
            raise ValueError(f"SimilCode source category and clone-type manifests disagree for case {case_id}.")
        clone_type = reference["clone_type"]
        if requested is not None and requested != row_language:
            continue
        if label_view == "copies_vs_independent" and clone_type == "unrelated":
            continue
        low, high = float(row["expected_similarity_min"]), float(row["expected_similarity_max"])
        if not 0 <= low <= high <= 100:
            raise ValueError(f"Invalid expected similarity band for SimilCode case {case_id}.")
        left_id = _normalize_id(row["code_a_id"], "code_a_id")
        right_id = _normalize_id(row["code_b_id"], "code_b_id")
        if left_id == right_id:
            if category != "identical" or row["code_a_path"] != row["code_b_path"]:
                raise ValueError(f"Inconsistent shared submission in SimilCode case {case_id}.")
            right_id += "_COPY"
        for side, file_id in (("a", left_id), ("b", right_id)):
            file_path = row[f"code_{side}_path"]
            source = _resolve_dataset_file(metadata_path.parent, file_path)
            info = {"file_id": file_id, "source_path": source, "source_file": file_path, "language": row_language}
            if files.setdefault(file_id, info) != info:
                raise ValueError(f"A SimilCode submission ID maps to multiple source files: {file_id}.")
        label = (category != "different") if label_view == "related" else (clone_type != "type4")
        pairs.append({
            "case_id": case_id, "left_id": left_id, "right_id": right_id, "label": int(label),
            "language": row_language, "reference_group": f"{row_language}_{row['reference_group']}",
            "similarity_category": category, "clone_type": clone_type,
            "expected_similarity_min": low, "expected_similarity_max": high,
        })
    if known_cases != set(types):
        raise ValueError("SimilCode metadata and tipos_clon.csv must describe the same case IDs.")
    if not pairs:
        raise ValueError("The selected SimilCode view has no cases.")
    metadata.update(
        adapter="similcode_pair", label_view=label_view, languages=sorted({row["language"] for row in pairs}),
        complete_source_case_count=len(known_cases), clone_type_source="tipos_clon.csv (author-derived)",
        source_files_sha256={path.name: _sha256(path) for path in (metadata_path, types_path)},
    )
    return write_pair_dataset(destination, list(files.values()), pairs, metadata).root


def _poj_file(source, split):
    base = _SPLITS[split]
    candidates = tuple(dict.fromkeys((f"{base}.jsonl", f"{split}.jsonl", f"{split}.parquet",
                                      f"{split}-00000-of-00001.parquet")))
    return _select_input(source, candidates)


def _poj_records(path):
    if path.suffix != ".parquet":
        yield from _json_records(path)
        return
    try:
        data = pd.read_parquet(path)
    except ImportError as exc:
        raise ValueError("Parquet input requires pyarrow. POJ-104 JSONL works without optional readers.") from exc
    yield from data.to_dict(orient="records")


def _group_qrels(groups):
    query_chunks, document_chunks = [], []
    for members in groups.values():
        ids = np.asarray(members, dtype=object)
        size = len(ids)
        distances = ~np.eye(size, dtype=bool)
        query_chunks.append(np.repeat(ids, size - 1))
        document_chunks.append(np.broadcast_to(ids, (size, size))[distances])
    queries = np.concatenate(query_chunks)
    documents = np.concatenate(document_chunks)
    return pd.DataFrame({"query_id": queries, "document_id": documents, "relevance": 1})


def adapt_poj104(source_root, destination, dataset_name=None, *, split="test",
                 task_type=None, source_revision=None):
    """Import one official POJ-104 partition from preprocessed JSONL or Parquet."""
    split = _split_name(split)
    path = _poj_file(source_root, split)
    metadata = _dataset_metadata(dataset_name, f"poj104_{split}", "code_similarity", task_type,
                                 POJ104_URL, _CODEXGLUE_TERMS, source_revision)
    files, queries, corpus, groups, seen = [], [], [], defaultdict(list), set()
    for row in _poj_records(path):
        file_id = _normalize_id(row.get("index", row.get("id")), "POJ-104 index")
        group = _normalize_id(row.get("label"), "POJ-104 problem label")
        code = row.get("code")
        if file_id in seen or not isinstance(code, str):
            raise ValueError(f"Duplicate POJ-104 index or invalid source text: {file_id}.")
        seen.add(file_id)
        groups[group].append(file_id)
        files.append({"file_id": file_id, "text": code, "suffix": ".cpp", "problem_id": group})
        queries.append({"query_id": file_id, "file_id": file_id, "problem_id": group})
        corpus.append({"document_id": file_id, "file_id": file_id, "problem_id": group})
    if not files:
        raise ValueError("The selected POJ-104 partition is empty.")
    official_shape = len(groups) == _POJ104_GROUP_COUNTS[split] and all(len(ids) == 500 for ids in groups.values())
    metadata.update(adapter="poj104_retrieval", split=split, input_format=path.suffix.lstrip("."), problem_count=len(groups),
                    source_files_sha256={path.name: _sha256(path)}, official_split_dimensions=official_shape,
                    relevance_definition="same problem excluding the query", ranking_metric="mean_average_precision_at_r",
                    exclude_self_matches=True, languages=["c", "cpp"])
    return write_retrieval_dataset(destination, files, queries, corpus, _group_qrels(groups), metadata).root


def _xlcost_root(source):
    source = Path(source).resolve()
    if source.name == "program_level":
        return source
    candidates = [path for path in source.rglob("program_level")
                  if path.is_dir() and path.parent.name == "code2code_search"
                  and "__MACOSX" not in path.relative_to(source).parts]
    if len(candidates) != 1:
        raise ValueError("Provide one XLCoST code2code_search/program_level directory.")
    return _resolve_dataset_file(source, candidates[0].relative_to(source))


def _xlcost_file(root, language, split):
    variants = ("valid.jsonl", "val.jsonl", "validation.jsonl") if split == "validation" else (f"{split}.jsonl",)
    return _select_input(root / language, variants, required=False)


def _xlcost_alignments(path, query_language, document_language):
    for row in _json_records(path):
        aligned = row.get("idx")
        if not isinstance(aligned, str) or aligned.count("/") != 1:
            raise ValueError("XLCoST program alignment requires its original 'idx' field.")
        query, document = aligned.split("/")
        query_group, _, query_lang = query.rpartition("-")
        document_group, _, document_lang = document.rpartition("-")
        if not query_group or not document_group or query_lang.lower() != query_language:
            raise ValueError(f"Invalid XLCoST program alignment: {aligned}.")
        if document_lang.lower() == document_language:
            yield row, query, document, query_group, document_group


def _xlcost_text(row, key):
    tokens = row.get(key)
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"XLCoST {key} must be a list of strings.")
    return " ".join(tokens)


def adapt_xlcost(source_root, destination, dataset_name=None, *, source_language="java",
                target_language="python", split="test", task_type=None, source_revision=None):
    """Convert one Java–Python program direction into a fixed-pool retrieval task."""
    source_language = str(source_language).lower().strip()
    target_language = str(target_language).lower().strip()
    if {source_language, target_language} != {"java", "python"}:
        raise ValueError("XLCoST supports Java→Python and Python→Java program retrieval.")
    split = _split_name(split)
    root = _xlcost_root(source_root)
    query_language = source_language.capitalize()
    path = _xlcost_file(root, query_language, split)
    if path is None:
        raise ValueError(f"XLCoST program split {split!r} is missing for {query_language}.")
    metadata = _dataset_metadata(dataset_name, f"xlcost_{source_language}_{target_language}_{split}",
                                 "code_similarity", task_type, XLCoST_URL,
                                 "Review archive and upstream source terms", source_revision)
    files, queries, corpus, qrels = {}, {}, {}, {}
    for row, query, document, query_group, document_group in _xlcost_alignments(path, source_language, target_language):
        query = _normalize_id(query, "XLCoST query program ID")
        document = _normalize_id(document, "XLCoST target program ID")
        for file_id, lang, group, key in ((query, source_language, query_group, "docstring_tokens"),
                                          (document, target_language, document_group, "code_tokens")):
            record = {"file_id": file_id, "language": lang, "problem_id": group, "text": _xlcost_text(row, key)}
            if files.setdefault(file_id, record) != record:
                raise ValueError(f"Conflicting source text for XLCoST program ID {file_id}.")
        queries[query] = {"query_id": query, "file_id": query, "language": source_language, "problem_id": query_group}
        corpus[document] = {"document_id": document, "file_id": document, "language": target_language, "problem_id": document_group}
        qrels[(query, document)] = {"query_id": query, "document_id": document, "relevance": 1}
    if not qrels:
        raise ValueError("The selected XLCoST direction has no program alignments.")
    groups = {entry["problem_id"] for entry in files.values()}
    overlap, source_hashes = {}, {split: _sha256(path)}
    for other in sorted(_SPLITS.keys() - {split}):
        other_file = _xlcost_file(root, query_language, other)
        if other_file:
            other_groups = {group for _, _, _, left, right in _xlcost_alignments(other_file, source_language, target_language)
                            for group in (left, right)}
            overlap[other] = len(groups & other_groups)
            source_hashes[other] = _sha256(other_file)
    metadata.update(
        adapter="xlcost_retrieval", split=split, source_language=source_language,
        target_language=target_language, languages=[source_language, target_language],
        direction=f"{source_language}_to_{target_language}", granularity="program",
        text_representation="space_joined_upstream_tokens", ranking_metric="mean_reciprocal_rank",
        problem_overlap_by_partition=overlap, source_files_sha256=source_hashes,
    )
    return write_retrieval_dataset(destination, list(files.values()), list(queries.values()),
                                   list(corpus.values()), list(qrels.values()), metadata).root


def _bcb_pairs(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 3 or values[2] not in {"0", "1"}:
                raise ValueError(f"{path.name}:{line_number} must contain two method IDs and a 0/1 label.")
            yield _normalize_id(values[0], "left method ID"), _normalize_id(values[1], "right method ID"), int(values[2])


def adapt_codexglue_bcb(source_root, destination, dataset_name=None, *, split="test",
                        task_type=None, source_revision=None):
    """Import the CodeXGLUE filtered BigCloneBench binary pair task."""
    split = _split_name(split)
    methods_path = _select_input(source_root, ("data.jsonl",))
    source = methods_path.parent
    pair_path = _select_input(source, (f"{_SPLITS[split]}.txt",))
    metadata = _dataset_metadata(dataset_name, f"codexglue_bcb_{split}", "code_clone", task_type,
                                 CODEXGLUE_BCB_URL, _CODEXGLUE_TERMS, source_revision)
    pair_frame = pd.DataFrame.from_records(_bcb_pairs(pair_path), columns=["left_id", "right_id", "label"])
    if pair_frame.empty:
        raise ValueError("The selected CodeXGLUE BigCloneBench split contains no pairs.")
    method_ids = set(pair_frame["left_id"]) | set(pair_frame["right_id"])
    methods = {}
    for row in _json_records(methods_path):
        method_id = _normalize_id(row.get("idx"), "CodeXGLUE BigCloneBench method ID")
        if method_id in methods or not isinstance(row.get("func"), str):
            raise ValueError(f"Duplicate CodeXGLUE BigCloneBench method ID or invalid source text: {method_id}.")
        methods[method_id] = row["func"]
    missing = method_ids - methods.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"CodeXGLUE BigCloneBench has {len(missing)} unresolved method IDs, including {preview}.")
    overlap, source_hashes = {}, {path.name: _sha256(path) for path in (methods_path, pair_path)}
    for other in sorted(_SPLITS.keys() - {split}):
        other_path = _select_input(source, (f"{_SPLITS[other]}.txt",), required=False)
        if other_path:
            others = {method for left, right, _ in _bcb_pairs(other_path) for method in (left, right)}
            overlap[other] = len(method_ids & others)
            source_hashes[other_path.name] = _sha256(other_path)
    files = [{"file_id": method_id, "text": methods[method_id], "suffix": ".java", "language": "java"}
             for method_id in sorted(method_ids)]
    pairs = pair_frame.assign(split=split)
    metadata.update(
        adapter="codexglue_bcb_pair", split=split, languages=["java"], granularity="method",
        dataset_lineage="Filtered BigCloneBench derivative used in CodeXGLUE", primary_metric="f1",
        shared_methods_by_partition=overlap,
        duplicate_ordered_pairs=int(pairs.duplicated(subset=["left_id", "right_id"]).sum()),
        official_split_dimensions=len(pairs) == _BCB_COUNTS[split],
        source_files_sha256=source_hashes,
    )
    return write_pair_dataset(destination, files, pairs, metadata).root


def similcode_score_slices(scored, score_column="similarity_score"):
    """Summarize SimilCode scores per language/category and language/clone type."""
    columns = {"language", "similarity_category", "clone_type", score_column}
    if not columns.issubset(scored):
        raise ValueError("SimilCode slices require language, category, clone type, and score columns.")
    all_rows = []
    for key, values in (("similarity_category", SIMILCODE_CATEGORIES), ("clone_type", SIMILCODE_CLONE_TYPES)):
        counts = Counter(zip(scored["language"], scored[key]))
        scores = scored.groupby(["language", key])[score_column].agg(["mean", "median", "min", "max"])
        for language in sorted(set(scored["language"])):
            for value in values:
                count = counts[language, value]
                row = scores.loc[(language, value)].to_dict() if count else dict.fromkeys(("mean", "median", "min", "max"))
                all_rows.append({"language": language, "dimension": key, "slice": value,
                                 "pair_count": count, **row})
    return pd.DataFrame(all_rows)
