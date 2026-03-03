from matheel.chunking import chunk_text


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
