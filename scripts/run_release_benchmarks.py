#!/usr/bin/env python3
"""Run a separately budgeted release edition without changing published v1 data.

Keep the output directory outside Git. Tool paths and input paths are private;
only the validated publisher outputs and the portable plan belong in the repo.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import matheel
from matheel.benchmark_analysis import build_benchmark_analysis
from matheel.benchmark_protocol import (freeze_benchmark_protocol, method_configuration,
                                        method_configuration_id, public_benchmark_protocol,
                                        _validate_public_metadata)

if __package__:
    from scripts.publish_plagiarism_benchmark import publish_benchmark_results
    from scripts.run_plagiarism_benchmark import atomic_json, normalized_methods, run
else:
    from publish_plagiarism_benchmark import publish_benchmark_results
    from run_plagiarism_benchmark import atomic_json, normalized_methods, run


ROOT = Path(__file__).resolve().parents[1]
NATIVE_MAX_FILES = 500
NATIVE_MAX_PAIRS = 125_000


def release_source():
    """Record every Python module, including external detector subpackages."""
    package = Path(matheel.__file__).resolve().parent
    files = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.rglob("*.py"))}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    commit = subprocess.check_output(
        ["git", "-C", str(package.parent), "rev-parse", "HEAD"], text=True).strip()
    return {"version": matheel.__version__, "git_commit": commit,
            "python_files_sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def verify_release_wheel(path, source):
    wheel = Path(path)
    with zipfile.ZipFile(wheel) as archive:
        files = {name.removeprefix("matheel/"): hashlib.sha256(archive.read(name)).hexdigest()
                 for name in archive.namelist() if name.startswith("matheel/") and name.endswith(".py")}
    if files != source["files"]:
        raise ValueError("Release wheel Python modules differ from the scoring checkout.")
    return {"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "all_python_modules_match_scoring_checkout": True}


def runner_source():
    scripts = ("run_release_benchmarks.py", "run_plagiarism_benchmark.py", "publish_plagiarism_benchmark.py")
    return {"git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "files": {f"scripts/{name}": hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                      for name in scripts}}


def edition_definitions(prepared_manifest, *, jar_path, java_path, dolos_path, node_path):
    """Predeclare common scopes, resources and methods before scoring."""
    original = copy.deepcopy(prepared_manifest)
    methods = json.loads((ROOT / "benchmarks/plagiarism-v1/methods.json").read_text())
    full = copy.deepcopy(original)
    full["execution_policy"] = {"max_method_seconds": 180}
    editions = [{"id": "all-methods", "label": "0.7.1 · all 29 methods · 180 s budget",
                 "manifest": full, "methods": methods, "workers": 3, "excluded_views": []}]
    references = [copy.deepcopy(method) for method in methods["algorithms"]
                  if method["name"] in {"Winnowing", "Levenshtein"}]
    counts = json.loads((ROOT / "benchmarks/plagiarism-v1/protocol.json").read_text())
    file_counts = {entry["name"]: entry["content"]["counts"]["files"] for entry in counts["datasets"]}
    for language, jplag_language in (("java", "java"), ("cpp", "cpp"), ("python", "python3")):
        track = copy.deepcopy(original)
        track["datasets"] = []
        excluded = []
        for entry in original["datasets"]:
            if entry.get("similarity_options", {}).get("code_language") != language:
                continue
            files = file_counts[entry["name"]]
            pairs = files * (files - 1) // 2
            if files > NATIVE_MAX_FILES or pairs > NATIVE_MAX_PAIRS:
                excluded.append({"dataset_name": entry["name"], "files": files,
                                 "native_unordered_pairs": pairs,
                                 "reason": "Entire view excluded for every method: predeclared native all-pairs corpus resource limits."})
            else:
                track["datasets"].append(copy.deepcopy(entry))
        track["execution_policy"] = {"max_method_seconds": 600}
        limits = {"timeout_seconds": 120, "heap_mb": 1024,
                  "max_files": NATIVE_MAX_FILES, "max_comparisons": NATIVE_MAX_PAIRS}
        detector_methods = [
            {"name": "JPlag 6.3.0 average", "algorithm_path": "matheel.external_detectors.jplag",
             "algorithm_options": {**limits, "language": jplag_language, "jar_path": str(jar_path),
                                   "java_path": str(java_path), "min_tokens": 9, "score_mode": "average"},
             "similarity_options": {"preprocess_mode": "none"}},
            {"name": "Dolos 2.10.0", "algorithm_path": "matheel.external_detectors.dolos",
             "algorithm_options": {**limits, "language": language, "dolos_path": str(dolos_path),
                                   "node_path": str(node_path)},
             "similarity_options": {"preprocess_mode": "none"}},
        ]
        editions.append({"id": f"detectors-{language}",
                         "label": f"0.7.1 · {language} detector comparison · common bounded scope",
                         "manifest": track, "methods": {"algorithms": references + detector_methods},
                         "workers": 1, "excluded_views": excluded})
    return editions


def main():
    import fcntl  # POSIX measurement lock; scoring/evaluation APIs remain portable.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--jar-path", required=True)
    parser.add_argument("--java-path", required=True)
    parser.add_argument("--dolos-path", required=True)
    parser.add_argument("--node-path", required=True)
    parser.add_argument("--cpu-lock", default="/private/tmp/matheel-071-benchmark-cpu.lock")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--release-wheel", required=True)
    args = parser.parse_args()
    source = release_source()
    if source["version"] != "0.7.1" or source["git_commit"] != args.expected_source_commit:
        raise ValueError("Run against the exact declared 0.7.1 release checkout.")
    wheel = verify_release_wheel(args.release_wheel, source)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Use HF_HUB_OFFLINE=1 with the pinned existing model snapshots.")
    output = Path(args.output_dir).resolve()
    destination = Path(args.destination).resolve()
    original = json.loads(Path(args.prepared_manifest).read_text())
    frozen = freeze_benchmark_protocol(original)
    original_public = {**public_benchmark_protocol(frozen), "protocol_id": frozen["protocol_id"]}
    original_protocol = json.loads((ROOT / "benchmarks/plagiarism-v1/protocol.json").read_text())
    if original_public != original_protocol:
        raise ValueError("Prepared inputs, labels and selection must exactly match the published v1 protocol.")
    editions = edition_definitions(original, jar_path=args.jar_path, java_path=args.java_path,
                                   dolos_path=args.dolos_path, node_path=args.node_path)
    declarations = {}
    for item in editions:
        protocol = freeze_benchmark_protocol(item["manifest"])
        methods = normalized_methods(protocol, item["methods"]["algorithms"])
        declarations[item["id"]] = {
            "protocol_id": protocol["protocol_id"],
            "method_configurations": {method_configuration_id(method): method_configuration(method) for method in methods},
        }
    plan = {"schema_version": 1, "edition": "0.7.1", "source": source, "release_wheel": wheel,
            "runner_source": runner_source(),
            "prior_v1_unchanged": True, "dataset_labels_changed": False,
            "original_protocol_id": original_protocol["protocol_id"],
            "original_method_definitions_sha256": hashlib.sha256(
                (ROOT / "benchmarks/plagiarism-v1/methods.json").read_bytes()).hexdigest(),
            "native_detector_limits": {"timeout_seconds_per_corpus": 120, "heap_mb": 1024,
                                       "max_files": NATIVE_MAX_FILES, "max_unordered_pairs": NATIVE_MAX_PAIRS},
            "uncertainty": {"rounds": 500, "seed": 7, "confidence": 0.95},
            "measurement_policy": {"exclusive_cpu_lock": True,
                                   "other_controlled_benchmark_runs": "serialized",
                                   "cpu_threads_per_worker": 1,
                                   "timings": "observed wall time; concurrency is recorded per edition"},
            "editions": [{"id": item["id"], "label": item["label"],
                          "dataset_views": [entry["name"] for entry in item["manifest"]["datasets"]],
                          "methods": [method["name"] for method in item["methods"]["algorithms"]],
                          "workers": item["workers"], "execution_policy": item["manifest"]["execution_policy"],
                          "excluded_views": item["excluded_views"], **declarations[item["id"]]} for item in editions]}
    _validate_public_metadata(plan)
    atomic_json(output / "plan.json", plan)
    destination.mkdir(parents=True, exist_ok=True)
    with Path(args.cpu_lock).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        for item in editions:
            run_root = output / item["id"]
            manifest_path, methods_path = run_root / "manifest.local.json", run_root / "methods.local.json"
            atomic_json(manifest_path, item["manifest"])
            atomic_json(methods_path, item["methods"])
            report, _ = run(manifest_path, methods_path, run_root, workers=item["workers"])
            if release_source() != source:
                raise RuntimeError("Release source changed during evaluation.")
            # The existing validator verifies full candidate coverage, exact
            # score hashes, method identities and honest unranked failures.
            public = destination / item["id"]
            summary = publish_benchmark_results(
                run_root, score_zip=output / f"{item['id']}-scores.zip", destination=public,
                protocol_path=run_root / "protocol.json", methods_path=methods_path)
            analysis = build_benchmark_analysis(public / "matheel_plagiarism_v1.json", run_root / "scores",
                                                rounds=500, seed=7)
            atomic_json(public / "analysis.json", analysis)
            atomic_json(run_root / "publication-summary.json", summary)
            print(f"Published {item['id']}: {len(report['per_dataset'])} metric rows", flush=True)
        plan["total_wall_seconds"] = time.monotonic() - started
    atomic_json(destination / "edition.json", plan)
    inventory = {path.relative_to(destination).as_posix(): {
        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "edition-manifest.json"}
    atomic_json(destination / "edition-manifest.json", {"schema_version": 1, "files": inventory})
    print(f"Release edition completed in {plan['total_wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
