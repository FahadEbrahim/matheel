import pytest

from matheel import chunking
from matheel.chunking import chunk_text, chunker_parameter_names, parse_chunker_options


def test_line_chunking_supports_overlap():
    text = "a\nb\nc\nd"

    assert chunk_text(text, method="lines", chunk_size=2, chunk_overlap=1) == [
        "a\nb",
        "b\nc",
        "c\nd",
        "d",
    ]


def test_character_chunking_supports_overlap():
    assert chunk_text("abcdef", method="characters", chunk_size=3, chunk_overlap=1) == [
        "abc",
        "cde",
        "ef",
    ]


def test_code_chunker_falls_back_cleanly_without_chonkie(monkeypatch):
    monkeypatch.setattr(chunking, "_load_chonkie_class", lambda class_name: None)

    chunks = chunk_text(
        "line1\nline2\nline3",
        method="code",
        chunk_size=2,
        chunk_overlap=1,
        chunk_language="python",
        chunker_options=("include_line_numbers=true",),
    )

    assert chunks == ["line1\nline2", "line2\nline3", "line3"]


def test_parse_chunker_options_supports_strings():
    assert parse_chunker_options("flag=true,count=2,name=demo") == {
        "flag": True,
        "count": 2,
        "name": "demo",
    }


def test_chunker_parameter_names_exposes_native_chonkie_methods():
    assert chunker_parameter_names("code") == ("chunk_size", "language", "include_nodes", "tokenizer")
    assert chunker_parameter_names("chonkie_token") == ("chunk_size", "chunk_overlap", "tokenizer")
    assert chunker_parameter_names("chonkie_fast") == (
        "chunk_size",
        "delimiters",
        "pattern",
        "prefix",
        "consecutive",
        "forward_fallback",
    )


def test_native_chonkie_chunkers_work_when_installed():
    pytest.importorskip("chonkie")

    text = "def add(a, b):\n    total = a + b\n    return total\n\nclass Value:\n    pass\n"

    code_chunks = chunk_text(text, method="code", chunk_size=64, chunk_language="python")
    fast_chunks = chunk_text(text, method="chonkie_fast", chunk_size=64)

    assert len(code_chunks) >= 1
    assert len(fast_chunks) >= 1
    assert "def add" in code_chunks[0]
