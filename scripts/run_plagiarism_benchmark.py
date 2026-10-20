#!/usr/bin/env python3
"""Run frozen methods in supervised processes with validated resumable checkpoints.

Keep the output directory outside Git: it contains local paths, logs, and scored
rows. Only the portable protocol and aggregate artifacts are publication inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from matheel.benchmark_protocol import (
    failed_benchmark_result,
    freeze_benchmark_protocol,
    load_benchmark_protocol,
    merge_benchmark_results,
    method_configuration_id,
    public_benchmark_protocol,
    run_benchmark_protocol,
    validate_benchmark_result,
)
from matheel.leaderboard import leaderboard_payload, normalize_leaderboard_manifest


def normalized_methods(protocol, algorithms):
    manifest = {
        "datasets": [{"name": item["name"], "task": item["task_family"],
                      "path": item["path"]} for item in protocol["datasets"]],
        "algorithms": algorithms,
    }
    methods = normalize_leaderboard_manifest(manifest)["algorithms"]
    identities = [method_configuration_id(method) for method in methods]
    if len(set(identities)) != len(identities):
        raise ValueError("Methods must have distinct effective configurations.")
    return methods


def read_checkpoint(path, protocol, method, score_dir):
    """Reject incompatible or tampered checkpoints rather than silently reuse them."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_benchmark_result(payload, protocol=protocol, score_artifact_dir=score_dir)
    if payload["metadata"].get("supervised_execution", {}).get("matheel_source_sha256") != source_fingerprint():
        raise ValueError("Checkpoint Matheel source fingerprint does not match the current scorer.")
    expected = method_configuration_id(method)
    if set(payload["metadata"]["method_configurations"]) != {expected}:
        raise ValueError("Checkpoint does not match the requested method configuration.")
    if {row["algorithm_name"] for row in payload["per_dataset"]} != {method["name"]}:
        raise ValueError("Checkpoint method display name does not match the methods file.")
    return payload


def atomic_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(target)


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "matheel").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(protocol_path, method_path, checkpoint, score_dir):
    protocol = load_benchmark_protocol(protocol_path)
    method = json.loads(Path(method_path).read_text(encoding="utf-8"))
    source_sha = source_fingerprint()
    started = time.monotonic()
    report, _ = run_benchmark_protocol(protocol, [method], score_artifact_dir=score_dir)
    if source_fingerprint() != source_sha:
        raise RuntimeError("Matheel source changed during evaluation; discard this checkpoint.")
    report["metadata"]["supervised_execution"] = {
        "method_wall_seconds": time.monotonic() - started,
        "matheel_source_sha256": source_sha,
        "cpu_threads_per_worker": 1,
    }
    atomic_json(checkpoint, leaderboard_payload(report))


def run(manifest_path, methods_path, output_dir, workers=3, retry_failed=False):
    if workers < 1:
        raise ValueError("workers must be positive.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    protocol = freeze_benchmark_protocol(manifest, base_dir=manifest_file.parent)
    protocol_path = output / "protocol.local.json"
    atomic_json(protocol_path, protocol)
    atomic_json(output / "protocol.json", {**public_benchmark_protocol(protocol),
                                          "protocol_id": protocol["protocol_id"]})
    algorithms = json.loads(Path(methods_path).read_text(encoding="utf-8"))["algorithms"]
    methods = normalized_methods(protocol, algorithms)
    score_dir = output / "scores"
    score_dir.mkdir(exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    pending, completed, active = [], [], []
    for method in methods:
        identity = method_configuration_id(method)
        checkpoint = checkpoints / f"{identity}.json"
        if checkpoint.exists():
            payload = read_checkpoint(checkpoint, protocol, method, score_dir)
            incomplete = any(row.get("score") is None or row.get("error")
                             for row in payload["per_dataset"])
            if not (retry_failed and incomplete):
                completed.append(payload)
                print(f"Resume: {method['name']}", flush=True)
                continue
        pending.append((method, identity, checkpoint))
    budget = protocol["execution_policy"]["max_method_seconds"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                   "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    try:
        while pending or active:
            while pending and len(active) < workers:
                method, identity, checkpoint = pending.pop(0)
                method_path = checkpoints / f"{identity}.method.json"
                atomic_json(method_path, method)
                log = (checkpoints / f"{identity}.log").open("w", encoding="utf-8")
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker",
                     "--protocol", str(protocol_path), "--method", str(method_path),
                     "--checkpoint", str(checkpoint), "--score-dir", str(score_dir)],
                    stdout=log, stderr=subprocess.STDOUT, env=environment,
                )
                active.append((process, time.monotonic(), method, checkpoint, log))
                print(f"Started: {method['name']} (budget {budget:g}s)", flush=True)
            for item in active[:]:
                process, started, method, checkpoint, log = item
                elapsed = time.monotonic() - started
                expired = elapsed >= budget and process.poll() is None
                if expired:
                    process.kill()
                    process.wait()
                if process.poll() is None:
                    continue
                log.close()
                error = "TimeoutError" if expired else (
                    f"WorkerExit{process.returncode}" if process.returncode else None)
                if error is None:
                    payload = read_checkpoint(checkpoint, protocol, method, score_dir)
                else:
                    report, _ = failed_benchmark_result(protocol, [method], error=error,
                                                        elapsed_seconds=elapsed)
                    report["metadata"]["supervised_execution"] = {
                        "matheel_source_sha256": source_fingerprint(),
                        "cpu_threads_per_worker": 1,
                    }
                    payload = leaderboard_payload(report)
                    atomic_json(checkpoint, payload)
                completed.append(payload)
                active.remove(item)
                print(f"Finished: {method['name']} ({error or 'scored'}, {elapsed:.1f}s)",
                      flush=True)
            if active:
                time.sleep(0.25)
    finally:
        for process, _, _, _, log in active:
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()
    # Stable method order makes resuming independent of process completion order.
    order = {method_configuration_id(method): index for index, method in enumerate(methods)}
    completed.sort(key=lambda payload: order[next(iter(payload["metadata"]["method_configurations"]))])
    report, artifacts = merge_benchmark_results(protocol, completed, output_dir=output,
                                                basename="matheel_plagiarism_v1")
    validate_benchmark_result(report, protocol=protocol, score_artifact_dir=score_dir)
    print(f"Complete: {len(methods)} methods; {artifacts['json']}", flush=True)
    return report, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--methods")
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--score-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol, args.method, args.checkpoint, args.score_dir)
    else:
        if not all((args.manifest, args.methods, args.output_dir)):
            parser.error("--manifest, --methods, and --output-dir are required")
        run(args.manifest, args.methods, args.output_dir, args.workers, args.retry_failed)


if __name__ == "__main__":
    main()
