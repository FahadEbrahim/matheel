from gradio_app.html_utils import escape_html


def test_escape_html_escapes_markup_and_quotes():
    assert escape_html('<img src=x onerror="alert(1)">') == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"
    )


def test_escape_html_handles_none_as_empty_text():
    assert escape_html(None) == ""
