import pytest

from matheel.preprocessing import available_preprocess_modes, preprocess_code


def test_basic_preprocess_removes_comments_and_collapses_whitespace():
    code = """
    int value = 1; // trailing note
    /* remove this block */
    # comment
    int second = 2;
    """

    assert preprocess_code(code, mode="basic") == "int value = 1; int second = 2;"


def test_basic_preprocess_keeps_cpp_directives():
    code = """
    #include <stdio.h>
    int value = 1; # inline comment
    """

    assert preprocess_code(code, mode="basic") == "#include <stdio.h> int value = 1;"


def test_available_preprocess_modes_expose_advanced_only():
    modes = available_preprocess_modes()

    assert "advanced" in modes
    assert "synsem_basic" not in modes


def test_normalize_preprocess_drops_blank_lines_only():
    code = "line_a  \n\n\nline_b\n"

    assert preprocess_code(code, mode="normalize") == "line_a\nline_b"


def test_advanced_preprocess_strips_imports_literals_and_identifiers():
    code = """
    import os
    from math import sqrt
    #include <stdio.h>
    using namespace std;
    def AddValue(total_count, item2):
        text = "hello"
        n = 42
        return total_count + item2 + n
    """

    processed = preprocess_code(code, mode="advanced")

    assert "import os" not in processed
    assert "from math import sqrt" not in processed
    assert "#include" not in processed
    assert "using namespace" not in processed
    assert "<STR>" in processed
    assert "<NUM>" in processed
    assert "AddValue" not in processed
    assert "total_count" not in processed
    assert "item2" not in processed
    assert "id1" in processed


def test_preprocess_code_rejects_unknown_mode():
    with pytest.raises(ValueError):
        preprocess_code("x = 1", mode="unsupported-mode")
