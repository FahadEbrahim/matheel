# Matheel Examples

Examples are grouped by workflow:

- `basic/`: preprocessing, chunking, lexical scoring, vectors, code metrics, and archive ranking.
- `evaluation/`: dataset adapters, validation reports, threshold tuning, comparison suites, calibration reports, dataset maps, scored pair explanations, leaderboards, and reproducible benchmark outputs.
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

Generate a starter custom algorithm module:

```bash
matheel init-custom-algorithm my_algorithm.py
```

Run the local visualization and leaderboard examples when you want generated artifacts to inspect:

```bash
python examples/evaluation/visualization_demo.py
python examples/evaluation/leaderboard_demo.py --overwrite
python examples/evaluation/dataset_validation_demo.py
python examples/evaluation/threshold_tuning_demo.py
```

Launch the Gradio app from a repository checkout:

```bash
python -m pip install -e ".[gradio]"
python gradio_app/app.py
```

Use `.[all]` instead of `.[gradio]` when you want every semantic, chunking, metric, and visualization backend. The [Gradio notebook](notebooks/04_gradio_app.ipynb) includes a guided lexical pairwise check, while the [Gradio app guide](../docs/gradio.md) covers every tab and common troubleshooting.

Generated archives and benchmark outputs are ignored by git.
