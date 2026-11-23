import pytest

from matheel import chunking
from matheel import grammars
from matheel.chunking import available_chunking_methods, chunk_text, chunker_parameter_names, parse_chunker_options


def test_public_chunking_methods_are_chonkie_first():
    assert available_chunking_methods() == (
        "none",
        "code",
        "chonkie_token",
        "chonkie_sentence",
        "chonkie_recursive",
        "chonkie_fast",
    )


def test_legacy_line_chunking_is_removed():
    text = "a\nb\nc\nd"

    with pytest.raises(ValueError):
        chunk_text(text, method="lines", chunk_size=2, chunk_overlap=1)


def test_legacy_character_chunking_is_removed():
    with pytest.raises(ValueError):
        chunk_text("abcdef", method="characters", chunk_size=3, chunk_overlap=1)


def test_code_chunker_requires_chonkie_when_unavailable(monkeypatch):
    monkeypatch.setattr(chunking, "_load_chonkie_class", lambda class_name: None)

    with pytest.raises(ImportError):
        chunk_text(
            "line1\nline2\nline3",
            method="code",
            chunk_size=2,
            chunk_overlap=1,
            chunk_language="python",
            chunker_options=("include_line_numbers=true",),
        )


def test_chonkie_constructor_errors_are_reported_as_configuration_errors(monkeypatch):
    class BrokenChunker:
        def __init__(self, chunk_size=0, chunk_overlap=0, tokenizer=None):
            raise TypeError("bad chunker option")

    monkeypatch.setattr(chunking, "_load_chonkie_class", lambda class_name: BrokenChunker)

    with pytest.raises(ValueError, match="Failed to initialize TokenChunker") as exc_info:
        chunk_text("line1\nline2", method="chonkie_token", chunk_size=2)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_parse_chunker_options_supports_strings():
    assert parse_chunker_options("flag=true,count=2,name=demo") == {
        "flag": True,
        "count": 2,
        "name": "demo",
    }


def test_chunker_parameter_names_exposes_native_chonkie_methods():
    assert chunker_parameter_names("code") == ("chunk_size", "language", "include_nodes", "tokenizer")
    assert chunker_parameter_names("codechunker") == ()
    assert chunker_parameter_names("chonkie_code") == ()
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


def test_offline_code_chunks_use_utf8_byte_spans_and_token_budget():
    AutoTokenizer = pytest.importorskip("chonkie").AutoTokenizer
    text = "def café(x):\n    z = x + 1\n    return z\n\nprint(café(1))\n"
    data = text.encode("utf-8")
    tokenizer = AutoTokenizer("character")
    budget = int(16 * len(data) / tokenizer.count_tokens(text))
    chunks = grammars.process_code(text, "python", chunk_max_size=budget).chunks

    assert len(chunks) > 1
    assert chunks[-1].end_byte == len(data) > len(text)
    assert [data[c.start_byte:c.end_byte].decode() for c in chunks] == [c.content for c in chunks]
    assert "".join(c.content for c in chunks) == text
    assert sum(tokenizer.count_tokens(c.content) for c in chunks) == tokenizer.count_tokens(text)
    expected = [c.content.strip() for c in chunks if c.content.strip()]
    assert expected == ["def café(x)", ":", "z = x + 1", "return z", "print(café(1))"]
    assert chunk_text(text, method="code", chunk_size=16, chunk_language="python") == expected
    assert chunk_text(text, method="code", chunk_size=16, chunk_language="python", max_chunks=1) == expected[:1]


def test_code_chunking_auto_detection_respects_shebang():
    pytest.importorskip("chonkie")
    text = "#!/usr/bin/env python3\ndef café():\n    return 1\n"

    assert grammars.detect_code_language(text) == "python"
    assert chunk_text(text, method="code", chunk_size=12) == chunk_text(
        text, method="code", chunk_size=12, chunk_language="python"
    )
