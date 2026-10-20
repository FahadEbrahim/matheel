#!/usr/bin/env python3
"""Replay a predeclared, bounded comparison of existing retrieval configurations.

The input manifest points at locally prepared datasets. No datasets or model
weights are downloaded. Public artifacts contain identifiers and hashes, never
submission text or source file locators. Run from the repository root.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
from time import perf_counter
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT
DEFAULT_CPU_LOCK = Path("/private/tmp/matheel-071-benchmark-cpu.lock")
SCORE_COLUMNS = ["query_id", "document_id", "query_file_id", "document_file_id",
                 "candidate_score", "candidate_rank", "relevance", "similarity_score"]


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_artifact_manifest(root):
    write_json(root / "artifact-manifest.json", {
        "schema_version": 1,
        "files": {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size,
                                                     "sha256": sha256(path.read_bytes())}
                  for path in sorted(root.rglob("*"))
                  if path.is_file() and path.name != "artifact-manifest.json"},
    })


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_bounded_worker(script, payload, seconds):
    """Isolate the complete worker process tree and clean it up on every exit."""
    with tempfile.TemporaryFile() as error_output:
        process = subprocess.Popen(
            [sys.executable, str(script), "--worker"], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=error_output, start_new_session=True,
        )
        timed_out = False
        try:
            process.communicate(input=json.dumps(payload).encode(), timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if timed_out:
            return "timeout", "Predeclared wall-clock budget exceeded."
        if process.returncode:
            return "failed", f"Worker exit code {process.returncode}"
        return "complete", None


def package_inventory():
    return {str(path.relative_to(SOURCE_ROOT)): sha256(path.read_bytes())
            for path in sorted((SOURCE_ROOT / "matheel").rglob("*.py"))}


def set_source_root(path):
    global SOURCE_ROOT
    SOURCE_ROOT = Path(path).resolve()
    sys.path.insert(0, str(SOURCE_ROOT))


def wheel_proof(path, inventory):
    with zipfile.ZipFile(path) as archive:
        observed = {name: sha256(archive.read(name)) for name in archive.namelist()
                    if name.startswith("matheel/") and name.endswith(".py")}
    if observed != inventory:
        raise ValueError("Published wheel differs from the executed package source.")
    return {"filename": path.name, "sha256": sha256(path.read_bytes()),
            "all_python_modules_match": True}


def runner_inventory():
    return {f"scripts/{name}": sha256((ROOT / "scripts" / name).read_bytes()) for name in
            ("validate_retrieval_presets.py", "compare_existing_configurations.py")}


def environment_snapshot():
    packages = {}
    for name in ("matheel", "pandas", "numpy", "rapidfuzz", "scikit-learn",
                 "sentence-transformers", "transformers", "torch", "chonkie"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "installed_distributions": packages,
            "executed_source_version": re.search(r'(?m)^version = "([^"]+)"',
                                                 (SOURCE_ROOT / "pyproject.toml").read_text()).group(1),
            "thread_limits": {key: os.environ.get(key) for key in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "TOKENIZERS_PARALLELISM")}}


def verify_source(ref):
    """Require the exact release checkout and released package code."""
    observed_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True,
    ).strip()
    if observed_ref != ref:
        raise ValueError("Use the exact source_ref checkout via --source-root.")
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ref, "matheel"], cwd=SOURCE_ROOT, text=True,
    ).splitlines()
    expected = {name: sha256(subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=SOURCE_ROOT))
                for name in names if name.endswith(".py")}
    actual = package_inventory()
    if expected != actual:
        raise ValueError("Working package differs from the declared source ref.")
    return actual


def input_inventory(dataset):
    from matheel.datasets import load_code_texts

    tables = {name: sha256((dataset.root / f"{name}.csv").read_bytes())
              for name in ("files", "queries", "corpus", "qrels")}
    return {"tables_sha256": tables,
            "metadata_sha256": (sha256((dataset.root / "metadata.json").read_bytes())
                                if (dataset.root / "metadata.json").is_file() else None),
            "source_sha256": {key: sha256(value.encode("utf-8"))
                              for key, value in sorted(load_code_texts(dataset).items())},
            "queries": len(dataset.queries), "documents": len(dataset.corpus),
            "qrels": len(dataset.qrels)}


def validate_plan(plan, manifest):
    from matheel.feature_weights import normalize_feature_weights

    ref = str(plan.get("source_ref", ""))
    if plan.get("schema_version") != 1 or len(ref) != 40 or any(c not in "0123456789abcdef" for c in ref):
        raise ValueError("A versioned plan with an exact source_ref is required.")
    datasets = {row["name"]: row for row in manifest["datasets"] if row["task"] == "retrieval"}
    requested = plan["datasets"]
    if len(set(requested)) != len(requested) or set(requested) != set(datasets):
        raise ValueError("The plan must include every retrieval view exactly once.")
    names = set()
    for preset in plan["presets"]:
        name = preset["name"]
        if name in names or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Preset names must be unique lowercase path-safe identifiers.")
        names.add(name)
        if preset["mode"] not in {"exhaustive", "two_stage"}:
            raise ValueError("Unknown retrieval mode.")
        for key in ("similarity_options", "candidate_similarity_options"):
            if key in preset:
                normalize_feature_weights(preset[key]["feature_weights"])
        if preset["mode"] == "two_stage":
            n = preset["candidate_count"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("candidate_count must be a positive integer.")
    if not names or not 0 < plan["seconds_per_view"] <= 300:
        raise ValueError("Declare nonempty presets and a bounded deadline (0, 300].")
    if isinstance(plan["k"], bool) or not isinstance(plan["k"], int) or plan["k"] < 1:
        raise ValueError("k must be a positive integer.")
    return datasets


def worker(payload):
    set_source_root(payload.get("source_root", ROOT))
    from matheel.datasets import load_retrieval_dataset
    from matheel.evaluation import evaluate_retrieval_dataset
    from matheel.retrieval import evaluate_two_stage_retrieval_dataset
    from matheel.result_analysis import analyze_scored_results

    dataset = load_retrieval_dataset(payload["path"])
    if runner_inventory() != payload["runner_inventory"]:
        raise ValueError("Evidence runner differs from its declared inventory.")
    if package_inventory() != payload["package_inventory"]:
        raise ValueError("Package source differs from the declared release inventory.")
    if input_inventory(dataset) != payload["input_inventory"]:
        raise ValueError("Dataset changed after the input inventory was recorded.")
    preset = payload["preset"]
    options = {**payload["similarity_options"], **preset["similarity_options"]}
    started = perf_counter()
    if preset["mode"] == "two_stage":
        scores, metrics = evaluate_two_stage_retrieval_dataset(
            dataset, preset["candidate_count"], k=payload["k"],
            candidate_similarity_options={**payload["similarity_options"],
                                          **preset["candidate_similarity_options"]},
            similarity_options=options, exclude_self_matches=payload["exclude_self_matches"],
        )
        scoring_seconds = perf_counter() - started
        diagnostics = scores.attrs["retrieval"]["candidate_diagnostics"]["per_query"]
    else:
        scores, metrics = evaluate_retrieval_dataset(
            dataset, k=payload["k"], similarity_options=options,
            exclude_self_matches=payload["exclude_self_matches"],
        )
        scoring_seconds = perf_counter() - started
        metrics["exhaustive_comparisons"] = len(scores)
        diagnostics = []
        # Only full-universe rows are supplied to the error-analysis API. A
        # shortlist must never masquerade as exhaustive input or recover missed
        # candidates as synthetic zero-score ties.
        analysis = analyze_scored_results(scores, task="retrieval", k=payload["k"],
                                          rounds=100, seed=7, dataset_root=dataset.root)
        analysis["errors"] = [row for row in analysis["errors"]
                              if row["outcome"] == "relevant_below_k"]
        write_json(payload["output_dir"] / "analysis.json", analysis)
    if (package_inventory() != payload["package_inventory"]
            or runner_inventory() != payload["runner_inventory"]
            or input_inventory(dataset) != payload["input_inventory"]):
        raise ValueError("Package source or inputs changed during evaluation.")
    scores.loc[:, [column for column in SCORE_COLUMNS if column in scores]].to_csv(
        payload["output_dir"] / "scores.csv", index=False,
    )
    elapsed = perf_counter() - started
    write_json(payload["output_dir"] / "result.json", {
        "status": "complete", "scoring_seconds": scoring_seconds,
        "analysis_and_export_seconds": elapsed - scoring_seconds,
        "metrics": metrics, "candidate_queries": diagnostics,
    })


def run(args):
    set_source_root(args.source_root)
    from matheel.datasets import load_retrieval_dataset

    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    datasets = validate_plan(plan, manifest)
    inventory = verify_source(plan["source_ref"])
    wheel = wheel_proof(args.release_wheel, inventory)
    runner = runner_inventory()
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; use a new experiment directory.")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "plan.json", plan)
    provenance = {name: input_inventory(load_retrieval_dataset(datasets[name]["path"]))
                  for name in plan["datasets"]}
    execution = {
        "source_ref": plan["source_ref"], "package_files_sha256": inventory,
        "release_wheel": wheel,
        "plan_sha256": sha256(args.plan.read_bytes()),
        "manifest_sha256": sha256(args.manifest.read_bytes()),
        "runner_files_sha256": runner, "datasets": provenance,
        "runner_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "interpretation": "Single timed observation on existing declared judged views; no dataset relabeling, best-method selection, or official rank change.",
    }
    rows = []
    # Serialize CPU measurements with other local benchmark work. Reproduction
    # can use any writable lock path; it is not part of the scientific protocol.
    with args.cpu_lock.open("a") as lock:
        print("Waiting for the shared CPU measurement slot.", flush=True)
        waiting = perf_counter()
        wait_started = utc_now()
        fcntl.flock(lock, fcntl.LOCK_EX)
        execution["measurement_coordination"] = {
            "exclusive_cpu_lock": True, "lock_name": args.cpu_lock.name,
            "wait_started_utc": wait_started, "lock_acquired_utc": utc_now(),
            "wait_seconds": perf_counter() - waiting,
        }
        execution["environment"] = environment_snapshot()
        write_json(args.output_dir / "provenance.json", execution)
        print("Acquired the shared CPU measurement slot.", flush=True)
        for name in plan["datasets"]:
            dataset = datasets[name]
            for preset in plan["presets"]:
                destination = args.output_dir / name / preset["name"]
                destination.mkdir(parents=True)
                payload = {"path": dataset["path"], "preset": preset, "k": plan["k"],
                           "output_dir": str(destination),
                           "input_inventory": provenance[name],
                           "package_inventory": inventory,
                           "source_root": str(SOURCE_ROOT),
                           "runner_inventory": runner,
                           "similarity_options": dataset.get("similarity_options", {}),
                           "exclude_self_matches": dataset.get("selection", {}).get("candidate_policy") == "exclude_same_file"}
                started = perf_counter()
                status, reason = run_bounded_worker(Path(__file__).resolve(), payload, plan["seconds_per_view"])
                result = (json.loads((destination / "result.json").read_text()) if status == "complete"
                          else {"status": status, "reason": reason, "metrics": None})
                row = {"dataset": name, "preset": preset["name"],
                       "deadline_seconds": plan["seconds_per_view"],
                       "wall_seconds": perf_counter() - started, **result}
                if result["status"] != "complete":
                    # Failed partial files are never presented as completed results.
                    import shutil
                    shutil.rmtree(destination)
                    destination.mkdir()
                    write_json(destination / "result.json", row)
                rows.append(row)
                write_json(args.output_dir / "results.json", {"schema_version": 1, "rows": rows})
                print(f"{name} {preset['name']}: {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
        execution["measurement_coordination"]["measurements_finished_utc"] = utc_now()
        write_json(args.output_dir / "provenance.json", execution)
    import pandas as pd

    pd.DataFrame([{**{key: row[key] for key in ("dataset", "preset", "status", "wall_seconds", "deadline_seconds")},
                   "scoring_seconds": row.get("scoring_seconds"),
                   **(row["metrics"] or {})} for row in rows]).to_csv(args.output_dir / "summary.csv", index=False)
    write_artifact_manifest(args.output_dir)
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit("At least one worker failed; inspect the explicit failure rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Exact released checkout used for package imports.")
    parser.add_argument("--release-wheel", type=Path,
                        help="Published wheel whose recursive Python inventory must match.")
    parser.add_argument("--cpu-lock", type=Path, default=DEFAULT_CPU_LOCK)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.load(sys.stdin)
        payload["output_dir"] = Path(payload["output_dir"])
        worker(payload)
    else:
        if not all((args.manifest, args.plan, args.output_dir, args.release_wheel)):
            parser.error("--manifest, --plan, --output-dir and --release-wheel are required.")
        run(args)


if __name__ == "__main__":
    main()
