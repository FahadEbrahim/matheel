FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    TREE_SITTER_LANGUAGE_PACK_CACHE_DIR=/opt/matheel-grammars

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && matheel grammars \
    && useradd --create-home --uid 1000 app \
    && chown app:app /app

COPY --chown=app:app . .
USER app
EXPOSE 7860
CMD ["python", "app.py"]
