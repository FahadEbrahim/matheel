import json

from scripts.sync_gradio_app_version import RELEASE_NOTEBOOKS, sync_gradio_app_version


def write_sample_repo(root, version="1.2.3", synced_version="0.1.0"):
    (root / "gradio_app").mkdir()
    (root / "examples" / "notebooks").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "matheel"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "[![PyPI](https://img.shields.io/pypi/v/matheel.svg"
        f"?cacheSeconds=3600&release=v{synced_version})](https://pypi.org/project/matheel/)\n",
        encoding="utf-8",
    )
    (root / "gradio_app" / "README.md").write_text(
        f"This Space is aligned with `matheel=={synced_version}` and includes:\n",
        encoding="utf-8",
    )
    (root / "gradio_app" / "requirements.txt").write_text(
        f"matheel[all]=={synced_version}\n",
        encoding="utf-8",
    )
    for index, notebook_name in enumerate(RELEASE_NOTEBOOKS):
        assignment = (
            f'MATHEEL_REF="v{synced_version}"\n'
            if index == 3
            else f'MATHEEL_REF = "v{synced_version}"\n'
        )
        (root / "examples" / "notebooks" / notebook_name).write_text(
            json.dumps({"cells": [{"source": [assignment]}]}) + "\n",
            encoding="utf-8",
        )


def test_sync_gradio_app_version_updates_stale_files(tmp_path):
    write_sample_repo(tmp_path)

    assert sync_gradio_app_version(tmp_path) == 0

    assert "cacheSeconds=3600&release=v1.2.3" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )
    assert "`matheel==1.2.3`" in (tmp_path / "gradio_app" / "README.md").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "gradio_app" / "requirements.txt").read_text(
        encoding="utf-8"
    ) == "matheel[all]==1.2.3\n"
    for notebook_name in RELEASE_NOTEBOOKS:
        notebook = json.loads(
            (tmp_path / "examples" / "notebooks" / notebook_name).read_text(
                encoding="utf-8"
            )
        )
        assert notebook["cells"][0]["source"][0].endswith('"v1.2.3"\n')
    assert sync_gradio_app_version(tmp_path, check=True) == 0


def test_sync_gradio_app_version_check_reports_stale_files(tmp_path):
    write_sample_repo(tmp_path)

    assert sync_gradio_app_version(tmp_path, check=True) == 1

    assert "cacheSeconds=3600&release=v0.1.0" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )
    assert "`matheel==0.1.0`" in (tmp_path / "gradio_app" / "README.md").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "gradio_app" / "requirements.txt").read_text(
        encoding="utf-8"
    ) == "matheel[all]==0.1.0\n"
    for notebook_name in RELEASE_NOTEBOOKS:
        notebook_text = (tmp_path / "examples" / "notebooks" / notebook_name).read_text(
            encoding="utf-8"
        )
        assert "v0.1.0" in notebook_text
