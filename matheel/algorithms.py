import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from collections.abc import Mapping

import pandas as pd

from ._run_metadata import attach_run_metadata, elapsed_seconds_since, perf_counter
from .preprocessing import preprocess_code
from .similarity import RESULT_COLUMNS, calculate_similarity, extract_and_read_source


@dataclass(frozen=True)
class PairAlgorithm:
    """Resolved pair-scoring algorithm.

    Custom modules should define ``score_pair(code_a, code_b, ...)`` and may also
    define ``prepare_dataset(dataset, ...)``.
    """

    name: str
    score_pair: object
    prepare_dataset: object = None
    module_name: str | None = None
    function_name: str = "score_pair"
    package_name: str | None = None
    package_version: str | None = None
    source_fingerprint: dict | None = None


_PREPARE_RESERVED_KEYS = frozenset({"dataset", "algorithm_options"})
_SCORE_RESERVED_KEYS = frozenset(
    {"code_a", "code_b", "dataset_context", "row", "algorithm_options"}
)


def normalize_algorithm_options(algorithm_options=None):
    if algorithm_options is None:
        return {}
    if not isinstance(algorithm_options, Mapping):
        raise ValueError("algorithm_options must be a mapping of option names to JSON values.")

    normalized = {}
    for key, value in sorted(algorithm_options.items(), key=lambda item: str(item[0])):
        option_name = str(key).strip()
        if not option_name:
            raise ValueError("algorithm_options keys must be non-empty strings.")
        try:
            json.dumps(value, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"algorithm_options[{option_name!r}] must be JSON-serializable."
            ) from exc
        normalized[option_name] = value
    return normalized


def _safe_package_version(name):
    if not name:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_file(path):
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "source_type": "file",
        "file_name": target.name,
        "total_bytes": int(target.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _fingerprint_callable(function):
    try:
        source_path = inspect.getsourcefile(function)
    except TypeError:
        source_path = None
    if source_path and Path(source_path).exists():
        return _hash_file(source_path)
    return {"source_type": "callable", "sha256": None}


def _package_name_from_module(module_name):
    if not module_name:
        return None
    return str(module_name).split(".", 1)[0]


def _load_module_from_path(module_path):
    target = Path(module_path).expanduser().resolve()
    module_name = f"matheel_user_algorithm_{hashlib.sha1(str(target).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load algorithm module from: {target.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_module(algorithm):
    if isinstance(algorithm, ModuleType):
        return algorithm
    target = os.fspath(algorithm)
    if os.path.exists(target):
        return _load_module_from_path(target)
    return importlib.import_module(target)


def _module_source_fingerprint(module):
    module_file = getattr(module, "__file__", None)
    if module_file and Path(module_file).exists():
        return _hash_file(module_file)
    return {"source_type": "module", "sha256": None}


def resolve_pair_algorithm(
    algorithm,
    function_name="score_pair",
    prepare_name="prepare_dataset",
    name=None,
):
    if isinstance(algorithm, PairAlgorithm):
        return algorithm

    if isinstance(algorithm, ModuleType) or isinstance(algorithm, (str, os.PathLike)):
        module = _resolve_module(algorithm)
        score_pair = getattr(module, function_name, None)
        if score_pair is None or not callable(score_pair):
            raise ValueError(f"Algorithm module must define a callable '{function_name}' function.")
        prepare_dataset = getattr(module, prepare_name, None)
        if prepare_dataset is not None and not callable(prepare_dataset):
            raise ValueError(f"'{prepare_name}' must be callable when provided.")
        module_name = getattr(module, "__name__", None)
        package_name = _package_name_from_module(module_name)
        return PairAlgorithm(
            name=str(name or getattr(module, "__name__", "custom_algorithm")),
            score_pair=score_pair,
            prepare_dataset=prepare_dataset,
            module_name=module_name,
            function_name=function_name,
            package_name=package_name,
            package_version=_safe_package_version(package_name),
            source_fingerprint=_module_source_fingerprint(module),
        )

    if callable(algorithm):
        module_name = getattr(algorithm, "__module__", None)
        package_name = _package_name_from_module(module_name)
        return PairAlgorithm(
            name=str(name or getattr(algorithm, "__name__", "custom_algorithm")),
            score_pair=algorithm,
            prepare_dataset=None,
            module_name=module_name,
            function_name=getattr(algorithm, "__name__", function_name),
            package_name=package_name,
            package_version=_safe_package_version(package_name),
            source_fingerprint=_fingerprint_callable(algorithm),
        )

    raise ValueError("algorithm must be a callable, module path, module name, module, or PairAlgorithm.")


def _call_supported(function, values):
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**values)
    filtered = {key: value for key, value in values.items() if key in parameters}
    return function(**filtered)


def _build_call_values(base_values, algorithm_options, reserved_keys):
    options = normalize_algorithm_options(algorithm_options)
    overridden = sorted(set(options).intersection(reserved_keys))
    if overridden:
        names = ", ".join(overridden)
        raise ValueError(f"algorithm_options must not override reserved parameter names: {names}")
    return {
        **base_values,
        "algorithm_options": options,
        **options,
    }


def prepare_algorithm_dataset(algorithm, dataset, algorithm_options=None):
    resolved = resolve_pair_algorithm(algorithm)
    if resolved.prepare_dataset is None:
        return None
    values = _build_call_values({"dataset": dataset}, algorithm_options, _PREPARE_RESERVED_KEYS)
    return _call_supported(resolved.prepare_dataset, values)


def score_pair_with_algorithm(
    code_a,
    code_b,
    algorithm,
    algorithm_options=None,
    dataset_context=None,
    row=None,
):
    resolved = resolve_pair_algorithm(algorithm)
    values = _build_call_values(
        {
            "code_a": code_a,
            "code_b": code_b,
            "dataset_context": dataset_context,
            "row": dict(row or {}),
        },
        algorithm_options,
        _SCORE_RESERVED_KEYS,
    )
    raw_score = _call_supported(resolved.score_pair, values)
    numeric_score = float(raw_score)
    if not math.isfinite(numeric_score):
        raise ValueError(f"Algorithm score must be finite. Got: {raw_score}")
    return numeric_score


def pair_algorithm_metadata(algorithm, algorithm_options=None):
    resolved = resolve_pair_algorithm(algorithm)
    options = normalize_algorithm_options(algorithm_options)
    return {
        "algorithm_name": resolved.name,
        "algorithm_function": resolved.function_name,
        "algorithm_module": resolved.module_name,
        "algorithm_package": resolved.package_name,
        "algorithm_package_version": resolved.package_version,
        "algorithm_options": options,
        "algorithm_source_fingerprint": dict(resolved.source_fingerprint or {}),
    }


def attach_algorithm_metadata(frame, algorithm, algorithm_options=None):
    metadata = pair_algorithm_metadata(algorithm, algorithm_options=algorithm_options)
    frame.attrs["algorithm"] = metadata
    for key, value in metadata.items():
        frame.attrs[key] = value
    return frame


def build_matheel_pair_algorithm(name="matheel", **options):
    def score_pair(code_a, code_b, **override_options):
        merged = dict(options)
        merged.update(dict(override_options or {}))
        merged.pop("dataset_context", None)
        merged.pop("row", None)
        merged.pop("algorithm_options", None)
        return calculate_similarity(code_a, code_b, **merged)

    return PairAlgorithm(
        name=name,
        score_pair=score_pair,
        prepare_dataset=None,
        module_name="matheel.similarity",
        function_name="calculate_similarity",
        package_name="matheel",
        package_version=_safe_package_version("matheel"),
        source_fingerprint=_fingerprint_callable(calculate_similarity),
    )


def score_source_pairs_with_algorithm(
    source_path,
    algorithm,
    preprocess_mode="none",
    code_language=None,
    algorithm_options=None,
    threshold=0.0,
    number_results=10,
    progress=False,
    progress_callback=None,
):
    from itertools import combinations

    start_time = perf_counter()
    resolved_algorithm = resolve_pair_algorithm(algorithm)
    options = normalize_algorithm_options(algorithm_options)
    file_names, raw_codes = extract_and_read_source(source_path)
    codes = [
        preprocess_code(code, mode=preprocess_mode, language=code_language)
        for code in raw_codes
    ]
    source_context = {
        "source_kind": "source_pairs",
        "file_names": tuple(file_names),
        "codes": tuple(codes),
        "code_count": len(codes),
    }
    dataset_context = prepare_algorithm_dataset(
        resolved_algorithm,
        source_context,
        algorithm_options=options,
    )

    rows = []
    for i, j in combinations(range(len(codes)), 2):
        row = {"file_name_1": file_names[i], "file_name_2": file_names[j]}
        score = score_pair_with_algorithm(
            codes[i],
            codes[j],
            resolved_algorithm,
            algorithm_options=options,
            dataset_context=dataset_context,
            row=row,
        )
        if score >= float(threshold):
            rows.append({**row, "similarity_score": round(score, 4)})

    similarity_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not similarity_df.empty:
        similarity_df = similarity_df.sort_values(
            by=["similarity_score", "file_name_1", "file_name_2"],
            ascending=[False, True, True],
            kind="mergesort",
            ignore_index=True,
        )
    result = similarity_df.head(max(1, int(number_results)))
    attach_run_metadata(
        result,
        elapsed_seconds=elapsed_seconds_since(start_time),
        feature_weights={"custom": 1.0},
        vector_backend="inactive",
        code_metric="none",
        chunking_method="none",
    )
    return attach_algorithm_metadata(result, resolved_algorithm, algorithm_options=options)
