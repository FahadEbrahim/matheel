---
title: Matheel Framework
sdk: gradio
python_version: 3.12.8
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Matheel Framework

Interactive code-similarity analysis with configurable embedding, lexical, chunking, preprocessing, and code-aware metrics.

This Space is aligned with `matheel==0.3.0` and includes:

- Code metrics: CodeBLEU, CrystalBLEU, RUBY, TSED, CodeBERTScore
- Per-metric advanced parameters via dedicated UI fields
- Advanced preprocessing mode support
- Public chunking methods:
  - `none`
  - `code`
  - `chonkie_token`
  - `chonkie_sentence`
  - `chonkie_recursive`
  - `chonkie_fast`

Default embedding model: `huggingface/CodeBERTa-small-v1`.
