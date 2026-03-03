from matheel.preprocessing import preprocess_code


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
