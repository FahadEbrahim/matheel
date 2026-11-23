"""Pinned local Copydetect pair scores from a single label-free corpus run."""

import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..datasets import load_code_texts
from .common import (
    DetectorContext, DetectorError, checked_score, complete_context, file_sha256, positive_number,
    private_error_boundary, required_file, run_local, score_pair as score_pair,
    validate_options, validated_portable,
)

VERSION = "0.5.0"
_WORKER = Path(__file__).with_name("_copydetect_worker.py")
_RECORD_DEPENDENCIES = ("numpy", "Pygments", "matplotlib", "Jinja2", "tqdm", "MarkupSafe")
_LIMITS = {
    "max_files": 500, "max_source_bytes": 50_000_000, "max_comparisons": 125_000,
    "timeout_seconds": 300, "max_output_bytes": 2_000_000, "max_result_bytes": 100_000_000,
}
_OPTIONS = {*_LIMITS, "language", "noise_threshold", "guarantee_threshold", "score_mode", "filter_code", "include_spans"}


@dataclass(frozen=True)
class CopydetectContext(DetectorContext):
    """Prepared complete scores with optional native source offsets by pair."""

    native_matches: dict | None

    def matched_slices(self, left_id, right_id):
        """Return matched token count and half-open character offsets in requested orientation."""
        if self.native_matches is None:
            raise DetectorError("Prepare Copydetect with include_spans=True to retain matching slices.")
        left_id, right_id = str(left_id), str(right_id)
        key = tuple(sorted((left_id, right_id)))
        try:
            tokens, left, right = self.native_matches[key]
        except KeyError as exc:
            raise DetectorError("No result covers the requested Copydetect pair.") from exc
        if key != (left_id, right_id):
            left, right = right, left
        return {"matched_tokens": tokens, "left_slices": left, "right_slices": right}


def _configuration(options, *, published=False):
    extra = {"boilerplate_sha256", "python_hash_seed"} if published else {"boilerplate_paths", "expected_installation_sha256"}
    validate_options(options, _OPTIONS | extra)
    language = options.get("language")
    if not isinstance(language, str) or re.fullmatch(r"[a-z][a-z0-9_+#-]*", language) is None:
        raise DetectorError("Specify a lowercase Copydetect/Pygments language alias such as java.")
    score_mode = options.get("score_mode", "average")
    if score_mode not in {"average", "maximum", "query"}:
        raise DetectorError("Copydetect score_mode must be average, maximum, or query.")
    filter_code, include_spans = options.get("filter_code", True), options.get("include_spans", False)
    if not isinstance(filter_code, bool) or not isinstance(include_spans, bool):
        raise DetectorError("Copydetect filter_code and include_spans must be boolean.")
    noise = positive_number(options.get("noise_threshold", 25), "noise_threshold", integer=True)
    guarantee = positive_number(options.get("guarantee_threshold", 30), "guarantee_threshold", integer=True)
    if guarantee < noise:
        raise DetectorError("Copydetect guarantee_threshold must be at least noise_threshold.")
    if guarantee > 100_000:
        raise DetectorError("Copydetect guarantee_threshold must not exceed 100000 characters.")
    configuration = {
        "language": language, "score_mode": score_mode, "filter_code": filter_code,
        "include_spans": include_spans, "noise_threshold": noise, "guarantee_threshold": guarantee,
        "python_hash_seed": 0,
    }
    for key, default in _LIMITS.items():
        configuration[key] = positive_number(options.get(key, default), key, integer=key != "timeout_seconds")
    return configuration


def _boilerplate(options, byte_limit):
    paths = options.get("boilerplate_paths", [])
    if not isinstance(paths, list):
        raise DetectorError("Copydetect boilerplate_paths must be a list of local file paths.")
    output = {}
    for value in paths:
        path = required_file(value, "Copydetect boilerplate file")
        with path.open("rb") as stream:
            content = stream.read(byte_limit + 1)
        if len(content) > byte_limit:
            raise DetectorError("Copydetect boilerplate exceeds max_source_bytes.")
        output[hashlib.sha256(content).hexdigest()] = content.decode("utf-8")
        byte_limit -= len(content)
    return dict(sorted(output.items()))


def _adapter_fingerprint():
    hashes = [file_sha256(path) for path in (Path(__file__), _WORKER, Path(__file__).with_name("common.py"))]
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def _bytes_fingerprint(distribution):
    files = sorted(
        (path for path in (distribution.files or ()) if path.parts[0] == "copydetect" and path.suffix != ".pyc"),
        key=lambda path: path.as_posix(),
    )
    if not {"copydetect/" + name for name in ("__init__.py", "detector.py", "pywinnow.py")} <= {path.as_posix() for path in files}:
        raise DetectorError("Copydetect requires a recorded package installation such as a PyPI wheel.")
    digest = hashlib.sha256()
    for file in files:
        installed = distribution.locate_file(file)
        digest.update(json.dumps([file.as_posix(), file_sha256(installed)]).encode())
    return digest.hexdigest()


def _verified_configuration(options, configuration, boilerplate):
    try:
        distribution = importlib.metadata.distribution("copydetect")
        versions = {name: importlib.metadata.version(name) for name in _RECORD_DEPENDENCIES}
    except importlib.metadata.PackageNotFoundError as exc:
        raise DetectorError(f"Install the optional Copydetect {VERSION} wheel and its dependencies first.") from exc
    if distribution.version != VERSION:
        raise DetectorError(f"This adapter requires Copydetect {VERSION}.")
    from pygments.lexers import find_lexer_class_by_name
    from pygments.util import ClassNotFound

    try:
        find_lexer_class_by_name(configuration["language"])
    except ClassNotFound as exc:
        raise DetectorError("Copydetect language is not a supported Pygments lexer alias.") from exc
    installation = _bytes_fingerprint(distribution)
    expected = options.get("expected_installation_sha256")
    if expected is not None and expected != installation:
        raise DetectorError("Copydetect installation does not match the expected SHA256.")
    return {
        "detector": "copydetect", "version": VERSION, "installation_sha256": installation,
        "runtime": {
            "version": sys.version.split()[0], "executable_sha256": file_sha256(sys.executable),
            "dependency_versions": versions,
        },
        "adapter_sha256": _adapter_fingerprint(), "corpus_policy": "all_prepared_files_single_file_submissions",
        "self_matches": "unsupported", "configuration": {**configuration, "boilerplate_sha256": list(boilerplate)},
    }


@private_error_boundary
def portable_configuration(algorithm_options=None):
    """Return a path-free run identity, or validate an archived identity."""
    options = dict(algorithm_options or {})
    if "detector" in options:
        published = options.get("configuration") or {}
        config = _configuration(published, published=True)
        hashes = published.get("boilerplate_sha256")
        if not isinstance(hashes, list) or any(re.fullmatch(r"[0-9a-f]{64}", str(item)) is None for item in hashes):
            raise DetectorError("Copydetect boilerplate identity must contain only SHA256 fingerprints.")
        config["boilerplate_sha256"] = sorted(set(hashes))
        identity = validated_portable(
            options, detector="copydetect", version=VERSION, configuration=config,
            artifact_fields={"installation_sha256"}, runtime_hashes={"executable_sha256", "dependency_versions"},
        )
        dependencies = identity["runtime"]["dependency_versions"]
        if not isinstance(dependencies, dict) or set(dependencies) != set(_RECORD_DEPENDENCIES) or not all(
            isinstance(value, str) and value.strip() for value in dependencies.values()
        ):
            raise DetectorError("Copydetect identity requires versioned dependencies.")
        return identity
    config = _configuration(options)
    boilerplate = _boilerplate(options, config["max_source_bytes"])
    return _verified_configuration(options, config, boilerplate)


def _score_records(path, aliases, identity):
    config = identity["configuration"]
    include_spans = config["include_spans"]
    native_matches = {} if include_spans else None

    def records():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                result = json.loads(line)
                if len(result) != (7 if include_spans else 5):
                    raise DetectorError("Copydetect returned an invalid result record.")
                left, right, forward, reverse, overlap = result[:5]
                forward, reverse = checked_score(forward), checked_score(reverse)
                if include_spans:
                    left_spans, right_spans = (tuple(tuple(span) for span in spans) for spans in result[5:])
                    first, second = aliases[left], aliases[right]
                    if second < first:
                        left_spans, right_spans = right_spans, left_spans
                    native_matches[tuple(sorted((first, second)))] = overlap, left_spans, right_spans
                mode = config["score_mode"]
                if mode == "average":
                    forward = reverse = (forward + reverse) / 2
                elif mode == "maximum":
                    forward = reverse = max(forward, reverse)
                yield left, right, forward, reverse

    context = complete_context(records(), aliases, identity)
    return CopydetectContext(context.scores, identity, context.file_count, context.comparison_count, native_matches)


@private_error_boundary
def prepare_dataset(dataset, prepared_texts=None, algorithm_options=None, **unused):
    """Fingerprint prepared text once and compare each distinct pair with Copydetect."""
    options = dict(algorithm_options or {})
    config = _configuration(options)
    boilerplate = _boilerplate(options, config["max_source_bytes"])
    identity = _verified_configuration(options, config, boilerplate)
    raw_texts = dict(prepared_texts) if prepared_texts is not None else load_code_texts(dataset)
    texts = {str(key): value for key, value in raw_texts.items()}
    count = len(texts)
    if count != len(raw_texts) or not 2 <= count <= config["max_files"]:
        raise DetectorError("Copydetect requires distinct IDs and a corpus within max_files (minimum two).")
    if count * (count - 1) // 2 > config["max_comparisons"]:
        raise DetectorError("Copydetect corpus exceeds max_comparisons.")
    if any(not isinstance(value, str) for value in texts.values()):
        raise DetectorError("Copydetect expects text inputs.")
    total_bytes = sum(len(value.encode("utf-8")) for value in (*texts.values(), *boilerplate.values()))
    if total_bytes > config["max_source_bytes"]:
        raise DetectorError("Copydetect corpus and boilerplate exceed max_source_bytes.")
    aliases = {f"s{index:06d}": key for index, key in enumerate(sorted(texts))}
    request = {
        "sources": [(alias, texts[key]) for alias, key in aliases.items()],
        "configuration": config, "boilerplate": list(boilerplate.values()),
    }
    with tempfile.TemporaryDirectory(prefix="matheel-copydetect-") as temporary:
        root = Path(temporary)
        request_path, output_path = root / "input.json", root / "scores.jsonl"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        run_local(
            [sys.executable, "-P", _WORKER, request_path, output_path], cwd=root,
            timeout_seconds=config["timeout_seconds"], max_output_bytes=config["max_output_bytes"],
            result_paths=(output_path,), max_result_bytes=config["max_result_bytes"],
            environment_overrides={"PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(root / "mpl")},
        )
        return _score_records(output_path, aliases, identity)
