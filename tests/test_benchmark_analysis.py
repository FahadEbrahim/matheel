import json

import pandas as pd
import pytest

from matheel.benchmark_analysis import build_benchmark_analysis, load_benchmark_analysis
from matheel.benchmark_protocol import freeze_benchmark_protocol, run_benchmark_protocol
from matheel.datasets import write_pair_dataset


def test_companion_is_bound_to_result_and_audited_scores(tmp_path):
    root = tmp_path / "pairs"
    write_pair_dataset(root, files=pd.DataFrame([
        {"file_id": key, "suffix": ".py", "text": text}
        for key, text in (("a", "print(1)"), ("b", "print(1)"), ("c", "different"),
                          ("d", "return value"), ("e", "return value"), ("f", "unrelated"))
    ]), pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1},
        {"left_id": "a", "right_id": "c", "label": 0},
        {"left_id": "d", "right_id": "e", "label": 1},
        {"left_id": "d", "right_id": "f", "label": 0},
    ]), metadata={"task_type": "plagiarism"})
    protocol = freeze_benchmark_protocol({"datasets": [{
        "name": "pairs", "path": str(root), "task": "pair", "split": "test",
        "source_url": "https://example.org/fixture", "source_revision": "a" * 40,
    }]})
    score_dir = tmp_path / "scores"
    _, artifacts = run_benchmark_protocol(protocol, [{"name": "edit", "feature_weights": {"levenshtein": 1}}],
                                          output_dir=tmp_path / "results", score_artifact_dir=score_dir)
    result = artifacts["json"]
    assert load_benchmark_analysis(result) is None
    analysis = build_benchmark_analysis(result, score_dir, rounds=20)
    companion = result.with_name("analysis.json")
    companion.write_text(json.dumps(analysis))
    assert load_benchmark_analysis(result)["protocol_id"] == protocol["protocol_id"]
    assert {row["group_count"] for row in analysis["metrics"]} == {2}
    analysis["result_sha256"] = "0" * 64
    companion.write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="does not match"):
        load_benchmark_analysis(result)
    scores = next(score_dir.glob("*.csv"))
    scores.write_text(scores.read_text().replace("1.0", "0.7"))
    with pytest.raises(ValueError, match="fingerprint"):
        build_benchmark_analysis(result, score_dir, rounds=20)
