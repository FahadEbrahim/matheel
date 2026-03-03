# Changelog

## 0.2.0

- Added shared preprocessing and universal chunking support across the CLI, Python API, and Gradio app.
- Added configurable semantic vector backends: transformer, static hashed vectors, and multivector late interaction.
- Added code-aware metrics: CodeBLEU-style component scoring and CrystalBLEU.
- Added a JSON-driven comparison suite with `matheel compare-suite` and CSV/JSON summary output.
- Added directory-path support for CLI and Python API inputs while keeping the Gradio upload flow ZIP-based.
- Tightened packaging with optional extras for `metrics`, `gradio`, and `dev`.
- Updated documentation for the expanded interface and publication-oriented workflows.
