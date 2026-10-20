import json
from pathlib import Path

import pytest

from scripts.sync_gradio_app_version import RELEASE_NOTEBOOKS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _python_cells(notebook_name):
    notebook_path = REPOSITORY_ROOT / "examples" / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%%bash"):
            continue
        # Colab's directory magic is not Python. Preserve line numbers so an
        # invalid cell points to the line users need to fix in the notebook.
        source = "\n".join(
            "" if line.lstrip().startswith("%cd ") else line
            for line in source.splitlines()
        )
        yield index, source


@pytest.mark.parametrize("notebook_name", RELEASE_NOTEBOOKS)
def test_release_notebook_is_valid_nbformat_json(notebook_name):
    notebook_path = REPOSITORY_ROOT / "examples" / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]
    assert all(cell.get("cell_type") in {"code", "markdown", "raw"} for cell in notebook["cells"])


@pytest.mark.parametrize("notebook_name", RELEASE_NOTEBOOKS)
def test_release_notebook_python_cells_compile(notebook_name):
    for index, source in _python_cells(notebook_name):
        compile(source, f"{notebook_name}:cell_{index}", "exec")


@pytest.mark.parametrize(
    "notebook_name",
    ("02_datasets_and_reproducibility.ipynb", "03_custom_algorithms.ipynb"),
)
def test_local_notebook_workflows_run(notebook_name, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    namespace = {"display": lambda *args: None}
    for index, source in _python_cells(notebook_name):
        if "MATHEEL_REF" in source:
            continue  # The install cell requires Colab and network access.
        source = source.replace("/content/", f"{tmp_path}/")
        exec(compile(source, f"{notebook_name}:cell_{index}", "exec"), namespace)
