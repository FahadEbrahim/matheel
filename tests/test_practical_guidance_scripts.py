"""Protect the public evidence runner's completeness and source-free contract."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from matheel.datasets import load_pair_dataset, load_retrieval_dataset, write_pair_dataset, write_retrieval_dataset


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_retrieval_presets.py"
SPEC = importlib.util.spec_from_file_location("validate_presets", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_plan_cannot_silently_drop_a_retrieval_view_or_duplicate_it():
    manifest = {"datasets": [{"name": "a", "task": "retrieval"},
                             {"name": "b", "task": "retrieval"},
                             {"name": "pairs", "task": "pair"}]}
    plan = {"schema_version": 1, "source_ref": "a" * 40, "datasets": ["a"],
            "k": 10, "seconds_per_view": 45,
            "presets": [{"name": "winnowing", "mode": "exhaustive"}]}
    with pytest.raises(ValueError, match="every retrieval view"):
        RUNNER.validate_plan(plan, manifest)
    plan["datasets"] = ["a", "b", "b"]
    with pytest.raises(ValueError, match="every retrieval view"):
        RUNNER.validate_plan(plan, manifest)
    plan["datasets"] = ["a", "b"]
    assert set(RUNNER.validate_plan(plan, manifest)) == {"a", "b"}
    plan["presets"][0]["name"] = "../escape"
    with pytest.raises(ValueError, match="path-safe"):
        RUNNER.validate_plan(plan, manifest)


def test_worker_exports_no_source_and_keeps_omitted_positives_in_denominators(tmp_path):
    source = "TOP_SECRET_SOURCE = 12345\nprint(TOP_SECRET_SOURCE)\n"
    root = tmp_path / "dataset"
    write_retrieval_dataset(
        root,
        files=pd.DataFrame([{"file_id": key, "text": text} for key, text in
                            (("q", source), ("a", source), ("b", "unrelated = True\n"))]),
        queries=pd.DataFrame([{"query_id": "query", "file_id": "q"}]),
        corpus=pd.DataFrame([{"document_id": "a", "file_id": "a"},
                             {"document_id": "b", "file_id": "b"}]),
        qrels=pd.DataFrame([{"query_id": "query", "document_id": "b", "relevance": 1}]),
    )
    destination = tmp_path / "results"
    destination.mkdir()
    payload = {"path": str(root), "output_dir": str(destination), "k": 10,
               "exclude_self_matches": False, "similarity_options": {},
               "input_inventory": RUNNER.input_inventory(load_retrieval_dataset(root)),
               "package_inventory": RUNNER.package_inventory(),
               "runner_inventory": RUNNER.runner_inventory(),
               "preset": {"name": "lcs_top1", "mode": "two_stage", "candidate_count": 1,
                          "candidate_similarity_options": {"feature_weights": {"lcs": 1}},
                          "similarity_options": {"feature_weights": {"levenshtein": 1}}}}
    completed = subprocess.run([sys.executable, str(SCRIPT), "--worker"], input=json.dumps(payload),
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((destination / "result.json").read_text())
    assert report["metrics"]["candidate_recall"] == 0
    assert report["metrics"]["mean_average_precision"] == 0
    assert report["metrics"]["relevant_count"] == 1
    assert report["candidate_queries"][0]["positive_judgments"] == 1
    scores = pd.read_csv(destination / "scores.csv")
    assert scores["document_id"].tolist() == ["a"]
    for path in destination.rglob("*"):
        if path.is_file():
            exported = path.read_text()
            assert "TOP_SECRET_SOURCE" not in exported
            assert str(root) not in exported
    # A changed source can no longer be presented as the inventoried input.
    file_path = root / pd.read_csv(root / "files.csv").iloc[0]["file_path"]
    file_path.write_text("changed source\n")
    changed = subprocess.run([sys.executable, str(SCRIPT), "--worker"], input=json.dumps(payload),
                             capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "Dataset changed" in changed.stderr


def test_diagnostic_selection_is_label_blind_and_public_export_omits_submission(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import compare_existing_configurations as diagnostics
    from matheel.robustness import python_robustness_case, run_robustness_experiment

    secret = 'raise AssertionError("PRIVATE_SUBMISSION_MUST_NOT_EXECUTE")\n'
    root = tmp_path / "pairs"
    write_pair_dataset(
        root,
        files=pd.DataFrame([{"file_id": key, "text": text} for key, text in
                            (("a", "a = 1\n"), ("b", "b = 2\n"), ("c", secret))]),
        pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 1},
                            {"left_id": "a", "right_id": "c", "label": 0}]),
    )
    dataset = load_pair_dataset(root)
    selected = diagnostics.select_long_pair(dataset, 20000)
    assert selected[0] == 1
    dataset.pairs["label"] = 1 - dataset.pairs["label"]
    changed = diagnostics.select_long_pair(dataset, 20000)
    assert (changed[0], changed[2:]) == (selected[0], selected[2:])
    assert changed[1]["left_id"] == selected[1]["left_id"]
    assert changed[1]["right_id"] == selected[1]["right_id"]
    report, _ = run_robustness_experiment(
        [{"run_name": "lcs", "feature_weights": {"lcs": 1}}],
        cases=[python_robustness_case(secret, "comments", seed=17)],
    )
    exported = diagnostics.public_experiment(report)
    assert "PRIVATE_SUBMISSION_MUST_NOT_EXECUTE" not in json.dumps(exported)
    assert exported["cases"][0]["provenance"]["source_sha256"]["reference"]
    report["scores"][0]["status"] = "error"
    report["scores"][0]["error"] = "RuntimeError: /private/tmp/private-input.py PRIVATE_ERROR_SOURCE"
    failed = diagnostics.public_experiment(report)
    assert failed["scores"][0]["status"] == "error"
    assert "PRIVATE_ERROR_SOURCE" not in json.dumps(failed)
    assert "/private/tmp" not in json.dumps(failed)
