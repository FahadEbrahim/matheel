import html


def escape_html(value):
    return html.escape("" if value is None else str(value), quote=True)
