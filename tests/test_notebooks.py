import json
from pathlib import Path

import pytest

from scripts.sync_gradio_app_version import RELEASE_NOTEBOOKS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("notebook_name", RELEASE_NOTEBOOKS)
def test_release_notebook_is_valid_nbformat_json(notebook_name):
    notebook_path = REPOSITORY_ROOT / "examples" / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]
    assert all(cell.get("cell_type") in {"code", "markdown", "raw"} for cell in notebook["cells"])
