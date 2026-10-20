import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.check_pinned_matheel_release import (
    check_pinned_release,
    parse_args,
    pinned_matheel_version,
    release_matches_source,
    wheel_package_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def write_sample_repo(root, version="1.2.3"):
    (root / "gradio_app").mkdir()
    (root / "gradio_app" / "requirements.txt").write_text(
        f"matheel[all]=={version}\n",
        encoding="utf-8",
    )


def read_outputs(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_pinned_matheel_version_parses_extra_requirement():
    assert pinned_matheel_version("numpy\nmatheel[all]==0.5.0\n") == "0.5.0"


def test_pinned_matheel_version_rejects_missing_pin():
    with pytest.raises(ValueError, match="No pinned matheel requirement"):
        pinned_matheel_version("matheel>=0.5\n")


def test_push_event_skips_successfully_when_pin_is_not_published(tmp_path):
    write_sample_repo(tmp_path, version="9.9.9")
    output_path = tmp_path / "github_output.txt"

    result = check_pinned_release(
        tmp_path,
        event_name="push",
        availability_check=lambda version: False,
        github_output=output_path,
    )

    assert result == 0
    assert read_outputs(output_path) == {
        "should_sync_space": "false",
        "matheel_version": "9.9.9",
    }


def test_push_event_syncs_when_pin_is_already_published(tmp_path):
    write_sample_repo(tmp_path, version="0.5.0")
    output_path = tmp_path / "github_output.txt"

    result = check_pinned_release(
        tmp_path,
        event_name="push",
        availability_check=lambda version: True,
        github_output=output_path,
    )

    assert result == 0
    assert read_outputs(output_path) == {
        "should_sync_space": "true",
        "matheel_version": "0.5.0",
    }


def test_publish_event_waits_for_pin_to_be_available(tmp_path):
    write_sample_repo(tmp_path, version="0.5.0")
    output_path = tmp_path / "github_output.txt"
    attempts = iter([False, True])
    sleeps = []

    result = check_pinned_release(
        tmp_path,
        event_name="workflow_run",
        timeout_seconds=30,
        delay_seconds=5,
        availability_check=lambda version: next(attempts),
        sleep=sleeps.append,
        github_output=output_path,
    )

    assert result == 0
    assert sleeps == [5.0]
    assert read_outputs(output_path)["should_sync_space"] == "true"


def test_publish_event_fails_when_pin_never_appears(tmp_path):
    write_sample_repo(tmp_path, version="0.5.0")
    output_path = tmp_path / "github_output.txt"

    result = check_pinned_release(
        tmp_path,
        event_name="workflow_run",
        timeout_seconds=0,
        delay_seconds=0,
        availability_check=lambda version: False,
        sleep=lambda seconds: None,
        github_output=output_path,
    )

    assert result == 1
    assert read_outputs(output_path)["should_sync_space"] == "false"


def test_space_sync_updates_the_existing_repository_without_create_api():
    workflow = (ROOT / ".github" / "workflows" / "sync-hf-space.yml").read_text(
        encoding="utf-8"
    )

    assert "hf repo create" not in workflow
    assert "hf upload" not in workflow
    assert "git clone --depth 1" in workflow
    assert "git archive HEAD:gradio_app" in workflow
    assert "rsync --archive --checksum --delete" in workflow
    assert '--exclude=".gitattributes"' in workflow
    assert "HEAD:main" in workflow
    assert "check_pinned_matheel_release.py --check-source" in workflow


def make_wheel(sources):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        for name, data in sources.items():
            wheel.writestr(name, data)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "published",
    [
        {"matheel/__init__.py": b"", "matheel/lexical.py": b"value = 1\n"},
        {"matheel/__init__.py": b"", "matheel/lexical.py": b"value = 2\n"},
        {"matheel/__init__.py": b""},
        {"matheel/__init__.py": b"", "matheel/lexical.py": b"value = 1\n", "matheel/old.py": b""},
    ],
    ids=["identical", "changed-module", "new-unpublished-module", "missing-local-module"],
)
def test_published_source_comparison_checks_bytes_and_module_names_offline(tmp_path, published):
    current = {"matheel/__init__.py": b"", "matheel/lexical.py": b"value = 1\n"}
    for name, data in current.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    wheel = make_wheel({**published, "matheel-1.2.3.dist-info/METADATA": b"ignored"})
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        if url.endswith("/json"):
            return io.BytesIO(json.dumps({"urls": [{
                "packagetype": "bdist_wheel", "filename": "matheel-1.2.3-py3-none-any.whl",
                "url": "https://files.pythonhosted.org/example.whl",
                "digests": {"sha256": hashlib.sha256(wheel).hexdigest()},
            }]}).encode())
        return io.BytesIO(wheel)

    assert release_matches_source(tmp_path, "1.2.3", urlopen=fake_urlopen) is (current == published)
    assert len(calls) == 2


def test_wheel_inspection_reads_only_package_python_files_without_execution():
    sources = {
        "matheel/__init__.py": b"raise RuntimeError('do not execute')\n",
        "matheel/nested/module.py": b"nested = True\n",
    }
    wheel = make_wheel({
        **sources, "matheel/assets/data.json": b"{}", "other/__init__.py": b"",
    })

    assert wheel_package_sources(wheel) == sources


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_published_wheel_requires_matching_sha256_before_source_comparison(tmp_path, digest):
    wheel = make_wheel({"matheel/__init__.py": b""})

    def fake_urlopen(url, timeout):
        if url.endswith("/json"):
            return io.BytesIO(json.dumps({"urls": [{
                "packagetype": "bdist_wheel", "filename": "matheel.whl",
                "url": "https://files.pythonhosted.org/example.whl", "digests": {"sha256": digest},
            }]}).encode())
        return io.BytesIO(wheel)

    with pytest.raises(ValueError, match="SHA256"):
        release_matches_source(tmp_path, "1.2.3", urlopen=fake_urlopen)


@pytest.mark.parametrize("event,expected", [("push", 0), ("workflow_run", 1), ("workflow_dispatch", 1)])
def test_mismatched_source_skips_push_and_fails_release_or_manual_sync_immediately(tmp_path, event, expected):
    write_sample_repo(tmp_path)
    output = tmp_path / "output.txt"
    checks = []
    sleeps = []

    def source_check(root, version):
        checks.append((root, version))
        return False

    result = check_pinned_release(
        tmp_path, event_name=event, availability_check=lambda version: True,
        source_check=source_check, sleep=sleeps.append, github_output=output,
    )

    assert result == expected
    assert checks == [(tmp_path, "1.2.3")]
    assert sleeps == []
    assert read_outputs(output)["should_sync_space"] == "false"


def test_matching_published_source_permits_space_sync(tmp_path):
    write_sample_repo(tmp_path)
    output = tmp_path / "output.txt"

    result = check_pinned_release(
        tmp_path, event_name="workflow_run", availability_check=lambda version: True,
        source_check=lambda root, version: True, github_output=output,
    )

    assert result == 0
    assert read_outputs(output)["should_sync_space"] == "true"


def test_source_check_failure_does_not_allow_space_sync(tmp_path, capsys):
    write_sample_repo(tmp_path)
    output = tmp_path / "output.txt"

    def failing_source_check(root, version):
        raise ValueError("Invalid published wheel")

    assert check_pinned_release(
        tmp_path, event_name="push", availability_check=lambda version: True,
        source_check=failing_source_check, github_output=output,
    ) == 0
    assert read_outputs(output)["should_sync_space"] == "false"
    assert "Could not verify" in capsys.readouterr().err


def test_source_check_is_an_opt_in_cli_flag():
    assert parse_args([]).check_source is False
    assert parse_args(["--check-source"]).check_source is True
