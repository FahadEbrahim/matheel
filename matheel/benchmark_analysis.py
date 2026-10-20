"""Companion diagnostics for a published result, without modifying its ranks."""

import hashlib
import json
from pathlib import Path

from .benchmark_protocol import validate_benchmark_result
from .result_analysis import analyze_scored_results


def build_benchmark_analysis(result_path, score_dir, *, rounds=200, seed=7):
    """Audit source-free score CSVs and calculate per-view uncertainty.

    Original results and CSVs are read only. The companion is bound to the exact
    result bytes, protocol and audited score hashes; it carries no new ranks.
    """
    result_path = Path(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_benchmark_result(payload, score_artifact_dir=score_dir)
    metadata = payload["metadata"]
    protocol = metadata["benchmark_protocol"]
    names_to_ids = {row["algorithm_name"]: row["algorithm_id"] for row in payload["per_dataset"]}
    thresholds = {(row["dataset_name"], row.get("algorithm_id") or names_to_ids[row["algorithm_name"]]): row.get("threshold", .5)
                  for row in metadata.get("thresholds", [])}
    datasets = {entry["name"]: entry for entry in protocol["datasets"]}
    records = []
    for artifact in metadata.get("score_artifacts", []):
        dataset_name = artifact["dataset_name"]
        algorithm_id = artifact["algorithm_id"]
        dataset = datasets[dataset_name]
        threshold = thresholds.get((dataset_name, algorithm_id),
                                   protocol.get("threshold_policy", {}).get("value", .5))
        report = analyze_scored_results(
            Path(score_dir) / artifact["filename"], task=artifact["task_family"],
            threshold=threshold, k=dataset.get("k", 10), rounds=rounds, seed=seed,
        )
        for metric in report["metrics"]:
            records.append({"dataset_name": dataset_name, "algorithm_id": algorithm_id,
                            "task_family": artifact["task_family"], "grouping": report["grouping"],
                            "group_count": report["group_count"], "score_sha256": artifact["sha256"],
                            **metric})
    return {
        "schema_version": 1, "benchmark_id": protocol["benchmark_id"],
        "protocol_id": protocol["protocol_id"], "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "rounds": rounds, "seed": seed, "confidence": .95,
        "interpretation": "Per-view conditional grouped bootstrap. Pair groups connect shared source IDs; retrieval groups are queries. Aggregate uncertainty is not inferred from rounded leaderboard scores. Published ranks are unchanged.",
        "metrics": records,
    }


def load_benchmark_analysis(result_path, analysis_path=None):
    """Read a companion only when its exact result identity matches."""
    result_path = Path(result_path)
    if analysis_path is None:
        analysis_path = result_path.with_name("analysis.json")
        if not analysis_path.is_file():
            analysis_path = result_path.parent.with_name(result_path.parent.name + "-analysis.json")
    analysis_path = Path(analysis_path)
    if not analysis_path.is_file():
        return None
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = result["metadata"]
    if (payload.get("schema_version") != 1
            or payload.get("protocol_id") != metadata["protocol_id"]
            or payload.get("result_sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest()):
        raise ValueError("Analysis companion does not match the published result artifact.")
    indexed = {(row["dataset_name"], row["algorithm_id"]): row["sha256"]
               for row in metadata.get("score_artifacts", [])}
    point_scores = {(row["dataset_name"], row["algorithm_id"], row["metric"]): row["score"]
                    for row in result["per_dataset"] if row.get("eligible")}
    seen = set()
    for row in payload.get("metrics", []):
        key = (row["dataset_name"], row["algorithm_id"], row["metric"])
        if key in seen or indexed.get(key[:2]) != row.get("score_sha256"):
            raise ValueError("Analysis companion contains duplicate or unaudited metric records.")
        seen.add(key)
        if key in point_scores and abs(float(point_scores[key]) - float(row["score"])) > 1e-10:
            raise ValueError("Analysis companion point scores differ from the published result.")
        lower, upper = row.get("ci_lower"), row.get("ci_upper")
        if (lower is None) != (upper is None) or (lower is not None and not 0 <= lower <= upper <= 1):
            raise ValueError("Analysis confidence bounds must be ordered finite scores in [0, 1].")
    return payload
