FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    TREE_SITTER_LANGUAGE_PACK_CACHE_DIR=/opt/matheel-grammars

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 app \
    && mkdir -p "$TREE_SITTER_LANGUAGE_PACK_CACHE_DIR" \
    && chown app:app /app "$TREE_SITTER_LANGUAGE_PACK_CACHE_DIR"

USER app
# The grammar pack creates private directories owned by the provisioning user.
RUN matheel grammars \
    && python -c "from matheel.grammars import get_parser; assert not get_parser('python').parse(b'print(1)').root_node.has_error"

COPY --chown=app:app . .
EXPOSE 7860
CMD ["python", "app.py"]
