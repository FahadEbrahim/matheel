from pathlib import Path

import pytest

from scripts.check_pinned_matheel_release import check_pinned_release, pinned_matheel_version


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
    assert "rsync --archive --delete" in workflow
    assert '--exclude=".gitattributes"' in workflow
    assert "HEAD:main" in workflow
