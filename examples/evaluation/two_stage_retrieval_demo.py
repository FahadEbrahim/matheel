"""Run an offline LCS-to-GST retrieval experiment on a tiny synthetic dataset."""

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from matheel import evaluate_two_stage_retrieval_dataset, write_retrieval_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="matheel-two-stage-"))
    output.mkdir(parents=True, exist_ok=True)
    dataset = write_retrieval_dataset(output / "synthetic_retrieval", files=pd.DataFrame([
        {"file_id": "query", "text": "int total = value + 1;"},
        {"file_id": "original", "text": "int total = value + 1;"},
        {"file_id": "edited", "text": "int total = value + 2;"},
        {"file_id": "other", "text": "while (ready) { stop(); }"},
    ]), queries=pd.DataFrame([{"query_id": "q", "file_id": "query"}]),
        corpus=pd.DataFrame([{"document_id": key, "file_id": key} for key in ("original", "edited", "other")]),
        qrels=pd.DataFrame([{"query_id": "q", "document_id": "original", "relevance": 1}]),
        metadata={"name": "synthetic_two_stage_example", "task_type": "plagiarism"})
    scored, metrics = evaluate_two_stage_retrieval_dataset(
        dataset, candidate_count=2, k=2,
        candidate_similarity_options={"feature_weights": {"lcs": 1.0}},
        similarity_options={"feature_weights": {"gst": 1.0}, "gst_min_match_length": 2},
    )
    scored.to_csv(output / "scores.csv", index=False)
    (output / "experiment.json").write_text(json.dumps({"metrics": metrics, "retrieval": scored.attrs["retrieval"]},
                                                       indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
