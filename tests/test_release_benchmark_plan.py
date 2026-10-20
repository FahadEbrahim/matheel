import copy
import json
from pathlib import Path

from scripts.run_release_benchmarks import edition_definitions


ROOT = Path(__file__).resolve().parents[1]


def test_release_plan_preserves_all_original_views_and_declares_common_detector_scopes():
    protocol = json.loads((ROOT / "benchmarks/plagiarism-v1/protocol.json").read_text())
    manifest = {"datasets": [{**entry, "task": entry["task_family"], "path": entry["name"]}
                             for entry in protocol["datasets"]]}
    original = copy.deepcopy(manifest)
    editions = edition_definitions(manifest, jar_path="jplag.jar", java_path="java",
                                   dolos_path="dolos", node_path="node")
    assert manifest == original
    full = editions[0]
    assert len(full["manifest"]["datasets"]) == 16
    assert len(full["methods"]["algorithms"]) == 29
    assert full["manifest"]["execution_policy"] == {"max_method_seconds": 180}
    assert full["workers"] == 3
    expected_counts = {"java": 8, "cpp": 6, "python": 1}
    for edition in editions[1:]:
        language = edition["id"].removeprefix("detectors-")
        assert len(edition["manifest"]["datasets"]) == expected_counts[language]
        assert all(entry["similarity_options"]["code_language"] == language
                   for entry in edition["manifest"]["datasets"])
        assert edition["manifest"]["execution_policy"] == {"max_method_seconds": 600}
        assert edition["workers"] == 1
        methods = edition["methods"]["algorithms"]
        assert len(methods) == 4
        assert not any(method.get("dataset_exclusions") for method in methods)
        for method in methods[2:]:
            assert method["algorithm_options"]["timeout_seconds"] == 120
            assert method["algorithm_options"]["max_files"] == 500
            assert method["algorithm_options"]["max_comparisons"] == 125_000
            assert method["similarity_options"]["preprocess_mode"] == "none"
        if language == "java":
            assert [entry["dataset_name"] for entry in edition["excluded_views"]] == ["conplag_pair"]
            assert edition["excluded_views"][0]["files"] == 1822
        else:
            assert edition["excluded_views"] == []
