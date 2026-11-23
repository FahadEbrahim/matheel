import os
import subprocess
import sys

import pytest

from matheel import grammars


def test_clean_cache_offline_startup_and_missing_grammar_errors(tmp_path):
    env = os.environ.copy()
    env.pop("TREE_SITTER_LANGUAGE_PACK_LIBS_DIR", None)
    env["TREE_SITTER_LANGUAGE_PACK_CACHE_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", """
from importlib.util import find_spec

from matheel.chunking import chunk_text
from matheel.code_metrics import (
    codebleu_components, parser_tokenize_for_code_metrics, score_code_metric_pair,
)
from matheel.grammars import GrammarUnavailableError

assert chunk_text("print(1)") == ["print(1)"]
assert score_code_metric_pair("print(1)", "print(1)", metric_name="ruby", language="python") == 1.0
operations = [
    lambda: parser_tokenize_for_code_metrics("print(1)", "python"),
    lambda: codebleu_components("print(1)", "print(1)", language="python"),
    lambda: score_code_metric_pair("print(1)", "print(1)", metric_name="ruby", language="python", ruby_mode="tree"),
]
if find_spec("chonkie") is not None:
    operations.append(lambda: chunk_text("print(1)", method="code", chunk_language="python"))
for operation in operations:
    try:
        operation()
    except GrammarUnavailableError as exc:
        assert "matheel grammars python" in str(exc), str(exc)
    else:
        raise AssertionError("Expected a local grammar provisioning error")
"""],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


def test_offline_parser_reports_syntax_errors_and_rejects_unknown_languages():
    tree = grammars.get_parser("python").parse(b"def incomplete(")
    assert tree.root_node.has_error
    with pytest.raises(ValueError, match="Unsupported Tree-sitter language"):
        grammars.get_parser("not-a-grammar")
