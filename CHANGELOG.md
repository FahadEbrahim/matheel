# Changelog

## 0.2.1

- Refactored similarity scoring to use normalized feature-weight maps, so any feature can be weighted without hard-coding the final score blend.
- Added Hugging Face model routing helpers that can infer `sentence-transformers`, `model2vec`, or `PyLate` from model metadata when `vector_backend=auto`.
- Added optional `Chonkie` chunker adapters, including `CodeChunker`, with chunk-language and free-form chunker options.
- Added optional extras for `chunking`, `chunking_code`, `model2vec`, and `pylate`.
- Kept `metrics` conflict-free by relying on the built-in CodeBLEU-style implementation instead of pulling the external `codebleu` package by default.
- Updated the CLI and docs for feature weights, richer chunking, and backend routing.

## 0.2.0

- Added shared preprocessing and universal chunking support across the CLI, Python API, and Gradio app.
- Added configurable semantic vector backends: transformer, static hashed vectors, and multivector late interaction.
- Added code-aware metrics: CodeBLEU-style component scoring and CrystalBLEU.
- Added a JSON-driven comparison suite with `matheel compare-suite` and CSV/JSON summary output.
- Added directory-path support for CLI and Python API inputs while keeping the Gradio upload flow ZIP-based.
- Tightened packaging with optional extras for `metrics`, `gradio`, and `dev`.
- Updated documentation for the expanded interface and publication-oriented workflows.
