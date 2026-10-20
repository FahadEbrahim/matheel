"""Local JPlag 6.3.0 adapter for the custom PairAlgorithm protocol.

Options require explicit ``jar_path``, ``java_path`` and ``language``. The JAR
must match the upstream release hash. No source is uploaded and version checking
is disabled. Each normalized file is one submission, compared once per corpus.
"""

import json
import tempfile
import zipfile
from pathlib import Path

from .common import (DetectorError, complete_context, export_corpus, file_sha256,
                     positive_number, private_error_boundary, required_file, run_local, score_pair as score_pair, validate_options, validated_portable)

VERSION = "6.3.0"
JAR_SHA256 = "5f2c21e8b88ed77134effcb3a5a3ab13d188f6a3e16d401387f7479e92db9aa2"
LANGUAGES = {"java": ".java", "python3": ".py", "cpp": ".cpp", "c": ".c",
             "javascript": ".js", "typescript": ".ts", "csharp": ".cs", "go": ".go", "rust": ".rs"}
_COMMON = {"timeout_seconds", "max_output_bytes", "max_result_bytes", "max_files", "max_source_bytes", "max_comparisons"}


def _configuration(options):
    options = dict(options or {})
    validate_options(options, _COMMON | {"jar_path", "java_path", "language", "min_tokens", "normalize", "score_mode", "heap_mb"})
    language = options.get("language")
    if language not in LANGUAGES:
        raise DetectorError("JPlag language must be an explicit supported identifier: " + ", ".join(LANGUAGES))
    mode = options.get("score_mode", "average")
    if mode not in {"average", "maximum", "query"}:
        raise DetectorError("JPlag score_mode must be average, maximum, or query.")
    normalize = options.get("normalize", False)
    if not isinstance(normalize, bool) or (normalize and language not in {"java", "cpp"}):
        raise DetectorError("normalize must be boolean and is supported only for java/cpp.")
    config = {"language": language, "min_tokens": positive_number(options.get("min_tokens", 9), "min_tokens", integer=True),
              "normalize": normalize, "score_mode": mode, "heap_mb": positive_number(options.get("heap_mb", 1024), "heap_mb", integer=True)}
    for key, default in {"timeout_seconds": 300, "max_output_bytes": 2_000_000, "max_result_bytes": 250_000_000,
                         "max_files": 500, "max_source_bytes": 50_000_000, "max_comparisons": 125_000}.items():
        config[key] = positive_number(options.get(key, default), key, integer=key != "timeout_seconds")
    return config


@private_error_boundary
def portable_configuration(algorithm_options=None):
    """Verify local bytes and return a path-free, effective method identity."""
    options = dict(algorithm_options or {})
    if "detector" in options:
        config = _configuration(options.get("configuration"))
        if options.get("artifact_sha256") != JAR_SHA256:
            raise DetectorError("JPlag artifact identity does not match the pinned release.")
        return validated_portable(options, detector="jplag", version=VERSION, configuration=config,
                                  artifact_fields={"artifact_sha256"},
                                  runtime_hashes={"executable_sha256", "release_sha256", "modules_sha256"})
    config = _configuration(options)
    jar = required_file(options.get("jar_path"), "jar_path")
    java = required_file(options.get("java_path"), "java_path")
    if file_sha256(jar) != JAR_SHA256:
        raise DetectorError("JPlag JAR SHA256 does not match the pinned upstream 6.3.0 release.")
    home = java.parent.parent
    release = required_file(home / "release", "Java runtime release metadata")
    modules = required_file(home / "lib" / "modules", "Java runtime modules")
    with tempfile.TemporaryDirectory(prefix="matheel-java-version-") as temporary:
        version = run_local([java, "-version"], cwd=temporary, timeout_seconds=15).splitlines()
    if not version or not any('version "25' in line for line in version):
        raise DetectorError("Pinned JPlag 6.3.0 requires a Java 25 runtime.")
    from .common import adapter_sha256
    return {"detector": "jplag", "version": VERSION, "artifact_sha256": JAR_SHA256,
            "runtime": {"version": version[0], "executable_sha256": file_sha256(java),
                        "release_sha256": file_sha256(release), "modules_sha256": file_sha256(modules)},
            "adapter_sha256": adapter_sha256(), "corpus_policy": "all_prepared_files_single_file_submissions",
            "self_matches": "unsupported", "configuration": config}


def _parse_report(report, aliases, identity, max_result_bytes):
    try:
        with zipfile.ZipFile(report) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_result_bytes:
                raise DetectorError("JPlag report exceeds max_result_bytes.")
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise DetectorError("JPlag report contains duplicate archive members.")
            info = json.loads(archive.read("runInformation.json"))
            if info.get("failedSubmissions") or info.get("version") != {"major": 6, "minor": 3, "patch": 0}:
                raise DetectorError("JPlag rejected submissions or returned an unexpected report version.")
            expected = len(aliases) * (len(aliases) - 1) // 2
            if info.get("totalComparisons") != expected:
                raise DetectorError("JPlag did not compare the full declared corpus.")
            files = json.loads(archive.read("submissionFileIndex.json"))["fileIndexes"]
            if set(files) != set(aliases) or any(not items or any(file.get("tokenCount", 0) < 1 for file in items.values()) for items in files.values()):
                raise DetectorError("JPlag did not parse every declared input file.")
            records = []
            for name in archive.namelist():
                if not name.startswith("comparisons/") or not name.endswith(".json"):
                    continue
                entry = json.loads(archive.read(name))
                mode = identity["configuration"]["score_mode"]
                if mode == "query":
                    forward, reverse = entry["firstSimilarity"], entry["secondSimilarity"]
                else:
                    forward = reverse = entry["similarities"]["AVG" if mode == "average" else "MAX"]
                records.append((entry["firstSubmissionId"], entry["secondSubmissionId"], forward, reverse))
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DetectorError("Invalid or incomplete JPlag report; no scores were accepted.") from exc
    return complete_context(records, aliases, identity)


@private_error_boundary
def prepare_dataset(dataset, prepared_texts=None, algorithm_options=None, **unused):
    options = dict(algorithm_options or {})
    identity = portable_configuration(options)  # verify actual artifact before every execution
    config = identity["configuration"]
    jar, java = required_file(options.get("jar_path"), "jar_path"), required_file(options.get("java_path"), "java_path")
    with tempfile.TemporaryDirectory(prefix="matheel-jplag-") as temporary:
        root = Path(temporary)
        aliases = export_corpus(dataset, prepared_texts, root / "inputs", LANGUAGES[config["language"]],
                                **{key: config[key] for key in ("max_files", "max_source_bytes", "max_comparisons")})
        command = [java, f'-Xmx{config["heap_mb"]}m', "--enable-native-access=ALL-UNNAMED", "-jar", jar,
                   "--skip-version-check", "--mode", "RUN", "--language", config["language"], "--min-tokens", str(config["min_tokens"]),
                   "--shown-comparisons", "-1", "--similarity-threshold", "0", "--cluster-skip", "--encoding", "UTF-8",
                   "--result-file", root / "results", root / "inputs"]
        if config["normalize"]:
            command.insert(-1, "--normalize")
        run_local(command, cwd=root, timeout_seconds=config["timeout_seconds"], max_output_bytes=config["max_output_bytes"],
                  result_paths=[root / "results.jplag"], max_result_bytes=config["max_result_bytes"])
        return _parse_report(root / "results.jplag", aliases, identity, config["max_result_bytes"])
