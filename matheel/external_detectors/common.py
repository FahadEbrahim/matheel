"""Shared safeguards for local, once-per-corpus detector adapters."""

import hashlib
import itertools
import json
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from ..datasets import load_code_texts


class DetectorError(ValueError):
    """A local detector failed, rejected input, or returned incomplete results."""


class DetectorTimeoutError(DetectorError):
    """The bounded local detector exceeded its wall-clock allowance."""


@dataclass(frozen=True)
class DetectorContext:
    scores: dict
    identity: dict
    file_count: int
    comparison_count: int

    def score(self, row):
        for left, right in (("left_id", "right_id"), ("query_file_id", "document_file_id"),
                            ("file_name_1", "file_name_2")):
            if left in row and right in row:
                key = (str(row[left]), str(row[right]))
                break
        else:
            raise DetectorError("Detector scoring requires stable file IDs in the pair row.")
        if key[0] == key[1]:
            raise DetectorError("Local detector adapters exclude self comparisons; use exclude_self_matches=True for retrieval.")
        if key not in self.scores:
            raise DetectorError("Detector result does not cover the requested pair; missing scores are never zero-filled.")
        return self.scores[key]


def score_pair(code_a, code_b, dataset_context=None, row=None, **options):
    """Look up an already computed pair; source text and labels never enter a command."""
    if not isinstance(dataset_context, DetectorContext):
        raise DetectorError("Run prepare_dataset before scoring a detector pair.")
    return dataset_context.score(row or {})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root):
    """Hash installed files by relative name and bytes, excluding build intermediates.

    Native .node libraries are included. npm's build directories also contain
    machine-specific object files and generated Makefiles that never execute;
    these are excluded, while installed JavaScript, metadata and binaries remain.
    """
    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".bin", ".cache", "obj.target", ".deps", "node_gyp_bins"} for part in relative.parts):
            continue
        if "build" in relative.parts and (
            path.suffix in {".o", ".obj", ".d", ".a", ".tlog", ".pdb"}
            or path.name in {"Makefile", "config.gypi", "binding.Makefile", "gyp-mac-tool"}
            or path.name.endswith(".target.mk")
        ):
            continue
        if path.is_symlink():
            # npm's executable links duplicate the target. Other symlinks could
            # escape the declared installation and are deliberately unsupported.
            raise DetectorError("Detector installation must not contain external symbolic links.")
        if path.is_file():
            digest.update(json.dumps([relative.as_posix(), file_sha256(path)], separators=(",", ":")).encode())
            count += 1
    if not count:
        raise DetectorError("Detector installation contains no files.")
    return digest.hexdigest()


def required_file(value, name):
    if not value:
        raise DetectorError(f"Specify {name} explicitly; adapters never install or download tools.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise DetectorError(f"{name} must name an existing local file.")
    return path


def positive_number(value, name, *, integer=False):
    if isinstance(value, bool):
        raise DetectorError(f"{name} must be numeric, not boolean.")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise DetectorError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(number) or number <= 0 or (integer and number != int(number)):
        raise DetectorError(f"{name} must be finite and positive{' integer' if integer else ''}.")
    return int(number) if integer else number


def validate_options(options, allowed):
    unknown = set(options or {}).difference(allowed)
    if unknown:
        raise DetectorError("Unknown detector option(s): " + ", ".join(sorted(unknown)))


def run_local(command, *, cwd, timeout_seconds, max_output_bytes=2_000_000, result_paths=(), max_result_bytes=None):
    """Run without a shell; bound output and wall time and reap a POSIX process group."""
    timeout = positive_number(timeout_seconds, "timeout_seconds")
    limit = positive_number(max_output_bytes, "max_output_bytes", integer=True)
    log_path = Path(cwd) / "process.log"
    env = {key: value for key, value in os.environ.items()
           if key not in {"NODE_OPTIONS", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH"}}
    env.update({"NO_COLOR": "1", "CI": "true"})
    started = time.monotonic()
    result_limit = positive_number(max_result_bytes, "max_result_bytes", integer=True) if max_result_bytes is not None else None
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen([str(part) for part in command], cwd=cwd, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=os.name == "posix")
        except OSError as exc:
            raise DetectorError("Could not start the specified local detector runtime.") from exc
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise DetectorTimeoutError("Local detector exceeded timeout_seconds.")
                if result_limit is not None and sum(Path(path).stat().st_size for path in result_paths if Path(path).is_file()) > result_limit:
                    raise DetectorError("Local detector exceeded max_result_bytes.")
                if log_path.stat().st_size > limit:
                    raise DetectorError("Local detector exceeded max_output_bytes.")
                time.sleep(0.02)
            if log_path.stat().st_size > limit:
                raise DetectorError("Local detector exceeded max_output_bytes.")
            if process.returncode:
                # Tool output can include source text and private paths. Do not
                # copy it into published error rows or exception messages.
                raise DetectorError(f"Local detector exited with status {process.returncode}; no scores were accepted.")
            return log_path.read_text(encoding="utf-8", errors="replace")
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait()


def export_corpus(dataset, prepared_texts, root, extension, *, max_files=2000, max_source_bytes=50_000_000,
                  max_comparisons=2_000_000):
    """Export only prepared source text under opaque IDs, never labels or qrels.

    Corpus membership is all files supplied to prepare_dataset, including files
    that are not individually judged. Corpus-wide fingerprint filtering therefore
    has a stable, explicit meaning.
    """
    texts = dict(prepared_texts) if prepared_texts is not None else load_code_texts(dataset)
    texts = {str(key): value for key, value in texts.items()}
    if len(texts) < 2:
        raise DetectorError("A detector corpus requires at least two files.")
    if len(texts) > positive_number(max_files, "max_files", integer=True):
        raise DetectorError("Detector corpus exceeds max_files.")
    count = len(texts) * (len(texts) - 1) // 2
    if count > positive_number(max_comparisons, "max_comparisons", integer=True):
        raise DetectorError("Detector corpus exceeds max_comparisons.")
    if any(not isinstance(text, str) or not text.strip() for text in texts.values()):
        raise DetectorError("Empty or non-text source files are unsupported; no scores were accepted.")
    if sum(len(text.encode("utf-8")) for text in texts.values()) > positive_number(max_source_bytes, "max_source_bytes", integer=True):
        raise DetectorError("Detector corpus exceeds max_source_bytes.")
    aliases = {}
    root.mkdir()
    for index, (file_id, text) in enumerate(sorted(texts.items())):
        alias = f"s{index:06d}"
        directory = root / alias
        directory.mkdir()
        (directory / ("Main" + extension)).write_text(text, encoding="utf-8")
        aliases[alias] = file_id
    return aliases


def checked_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectorError("Detector returned an invalid numeric score.") from exc
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise DetectorError("Detector similarity must be finite and between zero and one.")
    return score


def complete_context(records, aliases, identity):
    """Require every unordered corpus pair exactly once, then map both directions."""
    expected = {frozenset(pair) for pair in itertools.combinations(aliases, 2)}
    seen, scores = set(), {}
    for left, right, forward, reverse in records:
        pair = frozenset((left, right))
        if left == right or left not in aliases or right not in aliases or pair in seen:
            raise DetectorError("Detector output contains an unknown, self, or duplicate pair.")
        seen.add(pair)
        forward, reverse = checked_score(forward), checked_score(reverse)
        scores[(aliases[left], aliases[right])] = forward
        scores[(aliases[right], aliases[left])] = reverse
    if seen != expected:
        raise DetectorError("Incomplete detector pair coverage; omitted results are never treated as zero.")
    return DetectorContext(scores, identity, len(aliases), len(expected))


def adapter_sha256():
    """Include shared safeguards and both adapters in reproducibility identity."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(json.dumps([path.name, file_sha256(path)], separators=(",", ":")).encode())
    return digest.hexdigest()


def validated_portable(options, *, detector, version, configuration, artifact_fields, runtime_hashes):
    """Validate an already published identity without requiring local runtime paths.

    This is for comparison/import, never execution: prepare_dataset still requires
    and verifies the actual tool installation separately.
    """
    required = {"detector", "version", "runtime", "adapter_sha256", "corpus_policy", "self_matches", "configuration", *artifact_fields}
    if set(options) != required:
        raise DetectorError("Portable detector identity must contain all verified identity fields and no extras.")
    if options.get("corpus_policy") != "all_prepared_files_single_file_submissions" or options.get("self_matches") != "unsupported":
        raise DetectorError("Unexpected portable detector corpus policy.")
    if options.get("detector") != detector or options.get("version") != version:
        raise DetectorError("Unexpected portable detector identity.")
    for key in ("artifact_sha256", "installation_sha256", "lockfile_sha256", "adapter_sha256"):
        if key in options and not re.fullmatch(r"[0-9a-f]{64}", str(options[key])):
            raise DetectorError("Invalid portable detector artifact fingerprint.")
    runtime = options.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {"version", *runtime_hashes} or not runtime.get("version"):
        raise DetectorError("Portable detector identity requires runtime identity.")
    for key, value in runtime.items():
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise DetectorError("Invalid portable detector runtime fingerprint.")
    # Apply the same portable metadata rule as benchmark manifests.
    from ..benchmark_protocol import _validate_public_metadata
    _validate_public_metadata(options)
    if options.get("configuration") != configuration:
        raise DetectorError("Portable detector configuration is not fully normalized.")
    return options


def private_error_boundary(function):
    """Keep unexpected dependency/OS failures useful without publishing their paths."""
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"Local detector failed ({type(exc).__name__}); no scores were accepted."
            ) from exc
    return guarded
