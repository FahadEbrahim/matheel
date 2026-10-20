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

The hosted Space installs `matheel==0.6.1`. Checkout features are synchronized to the Space after the matching package code is published. This app includes:

- Pair, collection, comparison-suite, normalized-dataset, dataset-validation, threshold-tuning, visualization, scored-pair explanation, ready-made public leaderboard, custom leaderboard, and leaderboard-inspection workflows
- Metric presets for balanced, lexical, embedding-only, code-aware, and individual offline baselines
- Downloadable artifacts for dataset evaluation, validation reports, threshold sweeps, dataset maps, pair explanations, ready leaderboards, leaderboard reports, and leaderboard detail reports
- Raw and parser-derived lexical tokenization for Winnowing and GST
- All 20 RapidFuzz scalar string metrics, including longest common subsequence (LCS), plus Winnowing and GST
- Relative feature weights normalized automatically, with saved comparison-suite settings
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
