import json

import pandas as pd
import pytest

from matheel.benchmark_protocol import freeze_benchmark_protocol
from matheel.datasets import write_pair_dataset
from scripts.run_plagiarism_benchmark import normalized_methods, read_checkpoint, run


def _inputs(tmp_path):
    data = tmp_path / "data"
    write_pair_dataset(data, files=pd.DataFrame([
        {"file_id": "a", "text": "print(1)"},
        {"file_id": "b", "text": "print(1)"},
        {"file_id": "c", "text": "unrelated program"},
    ]), pairs=pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1},
        {"left_id": "a", "right_id": "c", "label": 0},
    ]), metadata={"task_type": "plagiarism"})
    manifest = {"pair_metrics": ["auroc", "f1"], "datasets": [
        {"name": "pairs", "task": "pair", "path": str(data), "split": "test",
         "source_revision": "version:1", "source_url": "https://example.org/data"},
    ]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    methods_path = tmp_path / "methods.json"
    methods_path.write_text(json.dumps({"algorithms": ["Levenshtein"]}))
    return manifest, manifest_path, methods_path


def test_supervised_runner_resumes_verified_scores_and_rejects_tampering(tmp_path, monkeypatch):
    manifest, manifest_path, methods_path = _inputs(tmp_path)
    output = tmp_path / "run"
    report, _ = run(manifest_path, methods_path, output, workers=1)
    assert report["aggregate"]["eligible"].all()
    checkpoints = list((output / "checkpoints").glob("*.json"))
    checkpoint = next(path for path in checkpoints if not path.name.endswith(".method.json"))
    before = checkpoint.read_bytes()
    monkeypatch.setattr("scripts.run_plagiarism_benchmark.subprocess.Popen",
                        lambda *a, **kw: pytest.fail("Valid checkpoints must avoid rescoring"))
    run(manifest_path, methods_path, output, workers=1)
    assert checkpoint.read_bytes() == before
    protocol = freeze_benchmark_protocol(manifest)
    method = normalized_methods(protocol, ["Levenshtein"])[0]
    payload = read_checkpoint(checkpoint, protocol, method, output / "scores")
    with monkeypatch.context() as stale:
        stale.setattr("scripts.run_plagiarism_benchmark.source_fingerprint", lambda: "0" * 64)
        with pytest.raises(ValueError, match="source fingerprint"):
            read_checkpoint(checkpoint, protocol, method, output / "scores")
    score_file = output / "scores" / payload["metadata"]["score_artifacts"][0]["filename"]
    score_file.write_text(score_file.read_text().replace("1.0", "0.0"))
    with pytest.raises(ValueError, match="content fingerprint"):
        run(manifest_path, methods_path, output, workers=1)


def test_supervised_timeout_produces_unranked_failure(tmp_path):
    manifest, manifest_path, methods_path = _inputs(tmp_path)
    manifest["execution_policy"] = {"max_method_seconds": 0.001}
    manifest_path.write_text(json.dumps(manifest))
    report, _ = run(manifest_path, methods_path, tmp_path / "run", workers=1)
    assert not report["aggregate"]["eligible"].any()
    assert report["aggregate"]["rank"].isna().all()
    assert report["per_dataset"]["score"].isna().all()
    assert report["metadata"]["execution_failures"][0]["error"] == "TimeoutError"


def test_method_aliases_cannot_duplicate_benchmark_entries(tmp_path):
    manifest, _, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="distinct effective"):
        normalized_methods(freeze_benchmark_protocol(manifest), [
            {"name": "one", "feature_weights": {"lcs": 1}},
            {"name": "two", "feature_weights": {"lcs": 1}},
        ])


def test_worker_process_group_options_are_explicit_on_posix_and_windows():
    from scripts.run_plagiarism_benchmark import worker_process_options
    assert worker_process_options("posix") == {"start_new_session": True}
    assert worker_process_options("nt")["creationflags"] != 0
    assert worker_process_options("unknown") == {}


def test_windows_tree_cleanup_uses_taskkill_and_falls_back_to_process_kill(monkeypatch):
    from types import SimpleNamespace
    from scripts.run_plagiarism_benchmark import terminate_worker_tree
    calls = []
    class Process:
        pid = 123
        returncode = None
        def poll(self):
            return self.returncode
        def kill(self):
            calls.append("kill")
            self.returncode = -9
        def wait(self):
            calls.append("wait")
            return self.returncode
    def taskkill(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr("scripts.run_plagiarism_benchmark.subprocess.run", taskkill)
    terminate_worker_tree(Process(), platform_name="nt")
    assert calls[0] == ["taskkill", "/PID", "123", "/T", "/F"]
    assert calls[1:] == ["kill", "wait"]


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX process groups")
def test_timeout_cleanup_terminates_worker_descendants(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from scripts.run_plagiarism_benchmark import terminate_worker_tree, worker_process_options
    child_pid_file = tmp_path / "child_pid"
    heartbeat = tmp_path / "heartbeat"
    child_program = "import sys,time; from pathlib import Path\np=Path(sys.argv[1])\nwhile True:\n p.write_text(str(time.monotonic_ns())); time.sleep(0.01)"
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_program!r},sys.argv[2]]); "
        "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", program, str(child_pid_file), str(heartbeat)], **worker_process_options())
    try:
        deadline = time.monotonic() + 5
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.exists()
        child_pid = int(child_pid_file.read_text())
        os.kill(child_pid, 0)
        terminate_worker_tree(process)
        time.sleep(0.05)
        stopped = heartbeat.read_text()
        time.sleep(0.05)
        assert heartbeat.read_text() == stopped

    finally:
        terminate_worker_tree(process)
