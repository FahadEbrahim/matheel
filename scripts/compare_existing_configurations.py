#!/usr/bin/env python3
"""Bounded, source-free practical diagnostics using existing scorer APIs.

The fixed plan declares a normalized Python pair view, existing configurations,
and input-selection rules. Neither supplied submissions nor generated variants
are executed. Controlled fixtures use only their existing explicit validators.
"""

import argparse
import ast
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

if __package__:
    from scripts.validate_retrieval_presets import (
        DEFAULT_CPU_LOCK, ROOT, environment_snapshot, package_inventory, run_bounded_worker,
        runner_inventory, set_source_root, sha256, utc_now, verify_source, wheel_proof,
        write_artifact_manifest, write_json,
    )
else:
    from validate_retrieval_presets import (
        DEFAULT_CPU_LOCK, ROOT, environment_snapshot, package_inventory, run_bounded_worker,
        runner_inventory, set_source_root, sha256, utc_now, verify_source, wheel_proof,
        write_artifact_manifest, write_json,
    )


def pair_inventory(dataset):
    from matheel.datasets import load_code_texts

    return {"tables_sha256": {name: sha256((dataset.root / f"{name}.csv").read_bytes())
                              for name in ("files", "pairs")},
            "metadata_sha256": (sha256((dataset.root / "metadata.json").read_bytes())
                                if (dataset.root / "metadata.json").is_file() else None),
            "source_sha256": {key: sha256(value.encode("utf-8"))
                              for key, value in sorted(load_code_texts(dataset).items())},
            "pairs": len(dataset.pairs)}


def select_long_pair(dataset, ceiling):
    from matheel.datasets import load_code_texts

    texts = load_code_texts(dataset)
    eligible = []
    for index, row in enumerate(dataset.pairs.to_dict("records")):
        left, right = texts[row["left_id"]], texts[row["right_id"]]
        size = len(left.encode("utf-8")) + len(right.encode("utf-8"))
        if size > ceiling:
            continue
        try:
            ast.parse(left)
            ast.parse(right)
        except SyntaxError:
            continue
        eligible.append((size, index, row, left, right))
    if not eligible:
        raise ValueError("No parseable pair meets the predeclared byte ceiling.")
    # Choose by size and original row order, never by label or measured score.
    _, index, row, left, right = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
    return index, row, left, right


def public_experiment(report):
    """Keep diagnostics and identities without exporting submission strings."""
    value = {key: report[key] for key in
             ("schema_version", "experiment_type", "experiment_id", "diagnostic_only",
              "runs", "implementation", "scores", "interpretation")}
    if "preflight" in report:
        value["preflight"] = report["preflight"]
        value["inputs"] = {key: {"sha256": item["sha256"]}
                           for key, item in report["inputs"].items()}
    else:
        value["cases"] = [{"case_id": case["case_id"], "transformation": case["transformation"],
                           "provenance": case["provenance"]} for case in report["cases"]]
    # Backend exceptions can quote cache/temp locators or input fragments.
    # Public failures retain status/null scores and a generic reason only.
    value = json.loads(json.dumps(value))
    for score in value["scores"]:
        if score.get("error"):
            score["error"] = "Scoring failed; exception details omitted from public artifacts."
    for diagnostic in value.get("preflight", []):
        if diagnostic.get("error"):
            diagnostic["error"] = "Preflight or scoring failed; exception details omitted from public artifacts."
    return value


def worker(payload):
    set_source_root(payload.get("source_root", ROOT))
    from matheel.datasets import load_pair_dataset
    from matheel.evaluation import score_pair_dataset
    from matheel.long_files import compare_long_files
    from matheel.result_analysis import analyze_scored_results, write_result_analysis
    from matheel.robustness import python_robustness_case, run_robustness_experiment

    dataset = load_pair_dataset(payload["path"])
    if runner_inventory() != payload["runner_inventory"]:
        raise ValueError("Evidence runner differs from its declared inventory.")
    if package_inventory() != payload["package_inventory"]:
        raise ValueError("Package source differs from the declared release inventory.")
    if pair_inventory(dataset) != payload["input_inventory"]:
        raise ValueError("Dataset changed after the input inventory was recorded.")
    plan = payload["plan"]
    destination = Path(payload["output_dir"])
    if payload["kind"] == "pairs":
        reference = None
        reports = []
        for run in plan["lexical_runs"]:
            scores = score_pair_dataset(dataset, similarity_options=run["options"])
            columns = [column for column in ("left_id", "right_id", "label", "row_index", "similarity_score")
                       if column in scores]
            scores.loc[:, columns].to_csv(destination / f"{run['run_name']}-scores.csv", index=False)
            report = analyze_scored_results(
                scores, task="pair", reference_rows=reference, threshold=plan["threshold"],
                rounds=plan["bootstrap_rounds"], seed=plan["seed"], dataset_root=dataset.root,
            )
            write_result_analysis(report, destination / run["run_name"])
            reports.append({"run_name": run["run_name"], "metrics": report["metrics"],
                            "groups": report["group_count"], "slices": report["slices"],
                            "decision_disagreements": sum(row.get("decision_disagreement", False)
                                                          for row in report["errors"])})
            if reference is None:
                reference = scores
        write_json(destination / "result.json", {"status": "complete", "runs": reports})
        if (package_inventory() != payload["package_inventory"]
                or runner_inventory() != payload["runner_inventory"]
                or pair_inventory(dataset) != payload["input_inventory"]):
            raise ValueError("Package source or inputs changed during evaluation.")
        return
    index, row, left, right = select_long_pair(dataset, plan["pair_byte_ceiling"])
    inputs = {"dataset": plan["dataset"], "row_index": index,
              "left_id": row["left_id"], "right_id": row["right_id"],
              "left_sha256": sha256(left.encode("utf-8")), "right_sha256": sha256(right.encode("utf-8")),
              "left_utf8_bytes": len(left.encode("utf-8")), "right_utf8_bytes": len(right.encode("utf-8"))}
    if payload["kind"] == "robustness":
        fixtures, _ = run_robustness_experiment(plan["lexical_runs"], seed=plan["seed"])
        supplied, _ = run_robustness_experiment(
            plan["lexical_runs"], seed=plan["seed"],
            cases=[python_robustness_case(left, name, seed=plan["seed"])
                   for name in ("formatting", "comments")],
        )
        write_json(destination / "result.json", {
            "status": "complete", "source_input": inputs,
            "controlled_fixtures": public_experiment(fixtures), "supplied_source": public_experiment(supplied),
        })
    else:
        from huggingface_hub import snapshot_download

        model = snapshot_download(plan["model_id"], revision=plan["model_revision"], local_files_only=True)
        semantic = {"feature_weights": {"semantic": 1}, "vector_backend": "sentence_transformers",
                    "model_name": model, "device": "cpu", "max_token_length": plan["max_token_length"]}
        chunked = {**semantic, "chunking_method": "chonkie_token", "chunk_size": plan["chunk_size"],
                   "chunk_overlap": plan["chunk_overlap"], "chunker_options": {"tokenizer": "character"}}
        runs = [{"run_name": "raw_minilm", "options": semantic}]
        runs.extend({"run_name": f"chunked_{aggregation}",
                     "options": {**chunked, "chunk_aggregation": aggregation}} for aggregation in ("mean", "max", "first"))
        runs.append({"run_name": "chunked_mean_cap2", "options": {**chunked, "chunk_aggregation": "mean", "max_chunks": 2}})
        runs.append({"run_name": "full_winnowing", "options": {"feature_weights": {"winnowing": 1}}})
        report, _ = compare_long_files(left, right, runs)
        value = public_experiment(report)
        # The portable model identity is pinned; the cache's local path is not
        # a scorer hyperparameter and is never published.
        value = json.loads(json.dumps(value).replace(model, f"{plan['model_id']}@{plan['model_revision']}"))
        value["identity_note"] = "API experiment/run IDs bind the exact execution configuration, including its private model-cache locator. Replay uses the public pinned model identity and file hashes; cache-location changes can alter these opaque IDs."
        model_files = {str(path.relative_to(model)): sha256(path.read_bytes())
                       for path in sorted(Path(model).rglob("*")) if path.is_file()}
        write_json(destination / "result.json", {"status": "complete", "source_input": inputs,
                                                 "model_files_sha256": model_files, **value})
    if (package_inventory() != payload["package_inventory"]
            or runner_inventory() != payload["runner_inventory"]
            or pair_inventory(dataset) != payload["input_inventory"]):
        raise ValueError("Package source or inputs changed during evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--release-wheel", type=Path)
    parser.add_argument("--cpu-lock", type=Path, default=DEFAULT_CPU_LOCK)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(json.load(sys.stdin))
        return
    if not all((args.manifest, args.plan, args.output_dir, args.release_wheel)):
        parser.error("--manifest, --plan, --output-dir and --release-wheel are required.")
    set_source_root(args.source_root)
    plan = json.loads(args.plan.read_text())
    inventory = verify_source(plan["source_ref"])
    wheel = wheel_proof(args.release_wheel, inventory)
    runner = runner_inventory()
    manifest = json.loads(args.manifest.read_text())
    dataset = next(row for row in manifest["datasets"] if row["name"] == plan["dataset"])
    if dataset["task"] != "pair" or dataset.get("similarity_options", {}).get("code_language") != "python":
        raise ValueError("The declared input must be an existing normalized Python pair view.")
    if not 0 < plan["seconds_per_workflow"] <= 300:
        raise ValueError("Declare a positive per-workflow deadline no greater than 300 seconds.")
    if args.output_dir.exists():
        raise ValueError("Use a new output directory.")
    args.output_dir.mkdir(parents=True)
    from matheel.datasets import load_pair_dataset

    inputs = pair_inventory(load_pair_dataset(dataset["path"]))
    write_json(args.output_dir / "plan.json", plan)
    execution = {
        "source_ref": plan["source_ref"], "package_files_sha256": inventory,
        "release_wheel": wheel,
        "plan_sha256": sha256(args.plan.read_bytes()), "manifest_sha256": sha256(args.manifest.read_bytes()),
        "runner_files_sha256": runner,
        "runner_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_inventory": inputs,
        "model_id": plan["model_id"], "model_revision": plan["model_revision"],
        "interpretation": "Existing labels preserved; controlled diagnostics describe these inputs only. No universal preset selection or threshold fitting.",
    }
    rows = []
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
        for kind in ("pairs", "robustness", "long_files"):
            destination = args.output_dir / kind
            destination.mkdir()
            payload = {"kind": kind, "path": dataset["path"], "plan": plan,
                       "input_inventory": inputs, "package_inventory": inventory,
                       "source_root": str(args.source_root.resolve()),
                       "runner_inventory": runner, "output_dir": str(destination)}
            started = perf_counter()
            status, reason = run_bounded_worker(Path(__file__).resolve(), payload, plan["seconds_per_workflow"])
            if status == "complete":
                result = json.loads((destination / "result.json").read_text())
                # The APIs preserve errors as null-score rows. Do not promote a
                # successful process with such rows to a complete workflow.
                reports = ([result["controlled_fixtures"], result["supplied_source"]]
                           if kind == "robustness" else [result] if kind == "long_files" else [])
                if any(row["status"] != "ok" for report in reports for row in report["scores"]):
                    status, reason = "failed", "At least one declared scoring configuration failed."
                    result["status"] = status
                    write_json(destination / "result.json", result)
            else:
                import shutil

                shutil.rmtree(destination)
                destination.mkdir()
                write_json(destination / "result.json", {"status": status, "reason": reason})
            rows.append({"workflow": kind, "status": status, "reason": reason,
                         "wall_seconds": perf_counter() - started,
                         "deadline_seconds": plan["seconds_per_workflow"]})
            write_json(args.output_dir / "results.json", rows)
            print(f"{kind}: {status} ({rows[-1]['wall_seconds']:.2f}s)", flush=True)
        execution["measurement_coordination"]["measurements_finished_utc"] = utc_now()
        write_json(args.output_dir / "provenance.json", execution)
    write_artifact_manifest(args.output_dir)
    if any(row["status"] != "complete" for row in rows):
        raise SystemExit("At least one diagnostic workflow failed or exceeded its declared budget.")


if __name__ == "__main__":
    main()
