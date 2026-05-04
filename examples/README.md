# Matheel Examples

Examples are grouped by workflow:

- `basic/`: preprocessing, chunking, lexical scoring, vectors, code metrics, and archive ranking.
- `evaluation/`: dataset adapters, validation reports, threshold tuning, comparison suites, calibration reports, dataset maps, pair explanations, leaderboards, and reproducible benchmark outputs.
- `custom/`: custom `score_pair` algorithm modules.
- `notebooks/`: ordered Colab notebooks.
- `sample_data.py`: deterministic generator for the tiny Java sample archive used by examples and docs.

Generate the sample archive from the repository root:

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
```

Then run any archive-based example:

```bash
python examples/basic/sample_pairs_demo.py
```

Run the local visualization and leaderboard examples when you want generated artifacts to inspect:

```bash
python examples/evaluation/visualization_demo.py
python examples/evaluation/leaderboard_demo.py --overwrite
python examples/evaluation/dataset_validation_demo.py
python examples/evaluation/threshold_tuning_demo.py
```

Generated archives and benchmark outputs are ignored by git.
