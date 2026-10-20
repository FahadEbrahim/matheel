"""Local Dolos 2.10.0 adapter using its pinned JavaScript API in a bounded Node process."""

import json
import tempfile
from pathlib import Path

from .common import (DetectorError, adapter_sha256, complete_context, export_corpus, file_sha256,
                     positive_number, private_error_boundary, required_file, run_local, score_pair as score_pair, tree_sha256, validate_options, validated_portable)

VERSION = "2.10.0"
NPM_INTEGRITY = "sha512-4Orj8kGOeBnAu9l2qLLXkU3udeLDFpIqUeMxMeMoMwKcmCpZq6qSHyfR+UZsgpUC7r6JlBZ+J/dOzobVYIea7g=="
LANGUAGES = {"java": ".java", "python": ".py", "cpp": ".cpp", "c": ".c", "javascript": ".js", "typescript": ".ts"}
_DRIVER = r'''
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";
const [packageRoot, manifestPath, outputPath] = process.argv.slice(2);
const { Dolos } = await import(pathToFileURL(path.join(packageRoot, "node_modules/@dodona/dolos-lib/dist/index.js")));
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const engine = new Dolos(manifest.options);
const report = await engine.analyzePaths(manifest.files);
if (report.warnings.length) throw new Error("Detector produced input warnings");
const files = report.entries().map(entry => {
  const parsed = engine.tokenizer.parser.parse(entry.file.file.content);
  if (parsed.rootNode.hasError || entry.kgrams.length === 0) {
    throw new Error("Unparseable input or no selected fingerprints");
  }
  return { alias: path.basename(path.dirname(entry.file.file.path)), kgrams: entry.kgrams.length };
});
const pairs = report.allPairs().map(pair => ({
  left: path.basename(path.dirname(pair.leftFile.file.path)),
  right: path.basename(path.dirname(pair.rightFile.file.path)),
  similarity: pair.similarity
}));
writeFileSync(outputPath, JSON.stringify({ schema_version: 1, files, pairs }));
'''


def _configuration(options):
    options = dict(options or {})
    allowed = {"dolos_path", "node_path", "language", "kgram_length", "kgrams_in_window", "max_fingerprint_count",
               "include_comments", "heap_mb", "timeout_seconds", "max_output_bytes", "max_result_bytes",
               "max_files", "max_source_bytes", "max_comparisons", "expected_installation_sha256"}
    validate_options(options, allowed)
    if options.get("language") not in LANGUAGES:
        raise DetectorError("Dolos language must be an explicit supported identifier: " + ", ".join(LANGUAGES))
    include_comments = options.get("include_comments", False)
    if not isinstance(include_comments, bool):
        raise DetectorError("include_comments must be boolean.")
    config = {"language": options["language"], "include_comments": include_comments}
    for key, default in {"kgram_length": 23, "kgrams_in_window": 17, "max_fingerprint_count": 2_147_483_647,
                         "heap_mb": 1024, "timeout_seconds": 300, "max_output_bytes": 2_000_000,
                         "max_result_bytes": 100_000_000, "max_files": 500,
                         "max_source_bytes": 50_000_000, "max_comparisons": 125_000}.items():
        config[key] = positive_number(options.get(key, default), key, integer=key != "timeout_seconds")
    return config


@private_error_boundary
def portable_configuration(algorithm_options=None):
    options = dict(algorithm_options or {})
    if "detector" in options:
        config = _configuration(options.get("configuration"))
        if options.get("npm_integrity") != NPM_INTEGRITY:
            raise DetectorError("Dolos artifact identity does not match the pinned release.")
        if options.get("score_mode") != "symmetric_selected_fingerprint_coverage":
            raise DetectorError("Unexpected portable Dolos score mode.")
        return validated_portable(options, detector="dolos", version=VERSION, configuration=config,
                                  artifact_fields={"npm_integrity", "installation_sha256", "lockfile_sha256", "score_mode"},
                                  runtime_hashes={"executable_sha256"})
    config = _configuration(options)
    if not options.get("dolos_path"):
        raise DetectorError("Specify dolos_path to the local npm installation prefix.")
    installation = Path(options["dolos_path"]).expanduser().resolve()
    package = required_file(installation / "node_modules/@dodona/dolos/package.json", "Dolos package metadata")
    lock = required_file(installation / "package-lock.json", "Dolos npm lockfile")
    node = required_file(options.get("node_path"), "node_path")
    try:
        metadata, locked = json.loads(package.read_text()), json.loads(lock.read_text())
        pinned = locked["packages"]["node_modules/@dodona/dolos"]
        if metadata["version"] != VERSION or metadata["name"] != "@dodona/dolos" or pinned["version"] != VERSION or pinned["integrity"] != NPM_INTEGRITY:
            raise DetectorError("Dolos installation must use the pinned @dodona/dolos 2.10.0 release and lockfile.")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DetectorError("Invalid Dolos package metadata or npm lockfile.") from exc
    with tempfile.TemporaryDirectory(prefix="matheel-node-version-") as temporary:
        version = run_local([node, "--version"], cwd=temporary, timeout_seconds=15).strip()
    try:
        major = int(version.lstrip("v").split(".")[0])
    except ValueError as exc:
        raise DetectorError("Could not verify the Node runtime version.") from exc
    if major < 22:
        raise DetectorError("Pinned Dolos 2.10.0 requires Node 22 or newer.")
    installed_sha = tree_sha256(installation / "node_modules")
    if options.get("expected_installation_sha256") and options["expected_installation_sha256"] != installed_sha:
        raise DetectorError("Dolos installed files do not match expected_installation_sha256.")
    return {"detector": "dolos", "version": VERSION, "npm_integrity": NPM_INTEGRITY,
            "installation_sha256": installed_sha, "lockfile_sha256": file_sha256(lock),
            "runtime": {"version": version, "executable_sha256": file_sha256(node)},
            "adapter_sha256": adapter_sha256(), "corpus_policy": "all_prepared_files_single_file_submissions",
            "self_matches": "unsupported", "score_mode": "symmetric_selected_fingerprint_coverage", "configuration": config}


def _parse_report(path, aliases, identity, max_result_bytes):
    try:
        if path.stat().st_size > max_result_bytes:
            raise DetectorError("Dolos report exceeds max_result_bytes.")
        report = json.loads(path.read_text())
        files = report["files"]
        if report["schema_version"] != 1 or len(files) != len(aliases) or {item["alias"] for item in files} != set(aliases):
            raise DetectorError("Dolos did not analyze every declared input file.")
        if any(not isinstance(item["kgrams"], int) or item["kgrams"] < 1 for item in files):
            raise DetectorError("Dolos returned an input without fingerprints.")
        records = [(entry["left"], entry["right"], entry["similarity"], entry["similarity"]) for entry in report["pairs"]]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DetectorError("Invalid or incomplete Dolos report; no scores were accepted.") from exc
    return complete_context(records, aliases, identity)


@private_error_boundary
def prepare_dataset(dataset, prepared_texts=None, algorithm_options=None, **unused):
    options = dict(algorithm_options or {})
    identity = portable_configuration(options)  # hash actual installed files before every execution
    config = identity["configuration"]
    node = required_file(options.get("node_path"), "node_path")
    installation = Path(options["dolos_path"]).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="matheel-dolos-") as temporary:
        root = Path(temporary)
        aliases = export_corpus(dataset, prepared_texts, root / "inputs", LANGUAGES[config["language"]],
                                **{key: config[key] for key in ("max_files", "max_source_bytes", "max_comparisons")})
        driver = root / "driver.mjs"
        driver.write_text(_DRIVER, encoding="utf-8")
        manifest = {"files": [str(root / "inputs" / alias / ("Main" + LANGUAGES[config["language"]])) for alias in aliases],
                    "options": {"language": config["language"], "kgramLength": config["kgram_length"],
                                "kgramsInWindow": config["kgrams_in_window"], "maxFingerprintCount": config["max_fingerprint_count"],
                                "includeComments": config["include_comments"], "minSimilarity": 0, "limitResults": None}}
        manifest_path = root / "input.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        run_local([node, f'--max-old-space-size={config["heap_mb"]}', driver, installation, manifest_path, root / "results.json"],
                  cwd=root, timeout_seconds=config["timeout_seconds"], max_output_bytes=config["max_output_bytes"],
                  result_paths=[root / "results.json"], max_result_bytes=config["max_result_bytes"])
        return _parse_report(root / "results.json", aliases, identity, config["max_result_bytes"])
