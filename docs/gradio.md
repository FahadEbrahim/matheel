# Gradio App

Matheel's Gradio app provides interactive pairwise, collection, suite, dataset, visualization, and leaderboard workflows. Use the [hosted Hugging Face Space](https://huggingface.co/spaces/buelfhood/matheel-framework), the [Colab launcher](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/notebooks/04_gradio_app.ipynb), or a local checkout.

## Run Locally

From the repository root, create an environment and install the Gradio dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[gradio]"
.venv/bin/python gradio_app/app.py
```

Open the local URL shown in the terminal. The Gradio extra supports the app and lexical workflows. Install `.[all]` when you also need every semantic backend, chunker, code metric, and visualization dependency.

The app targets Gradio 5. Its dependencies intentionally require `gradio>=5,<6` because the Hugging Face model-search component currently requires Gradio below version 6.

## Guided Pairwise Check

Start with a lexical-only comparison so the first run does not need model weights:

1. Open the **Pairwise** tab.
2. Paste this into **Code A**:

    ```python
    def add(a, b):
        return a + b
    ```

3. Paste this into **Code B**:

    ```python
    def add(x, y):
        return x + y
    ```

4. Under **Metrics**, set **Metric Preset** to **Lexical Only**.
5. Select **Run Pair** and inspect the overall score and metric breakdown.

Semantic presets can download model weights and may take longer on their first run.

## Choose a Workflow

| Tab | Use it for |
| --- | --- |
| **Pairwise** | Compare two pasted code snippets and inspect the metric breakdown. |
| **Collection** | Rank similar files from an uploaded ZIP archive. |
| **Suite** | Run multiple saved comparison configurations over the same collection. |
| **Datasets** | Validate normalized dataset ZIPs, evaluate scores, tune thresholds, and export reports. |
| **Visualization** | Generate dataset maps or explain scored pairs. |
| **Leaderboard** | Build a ready leaderboard from normalized data or inspect exported JSON/ZIP artifacts. |

Collection and dataset uploads are ZIP-first. Keep source paths relative inside the archive, and use normalized dataset ZIPs for evaluation, visualization, and ready-leaderboard tasks. Download generated artifacts if you need to retain them after the session ends.

## Troubleshooting

- **The first semantic run is slow:** model files and ML runtimes may need to initialize or download. Try the lexical preset first to verify the app itself.
- **A backend is unavailable:** install `.[all]`, restart the app, and check the terminal for the missing optional dependency.
- **An upload is rejected:** use a ZIP archive and check that the selected workflow expects either a source collection or a normalized dataset.
- **A Colab launch cell keeps running:** this is expected while the server is active. Open the public Gradio link printed by the cell.
- **The hosted Space appears stale:** compare the Matheel version shown in its README with the latest [PyPI release](https://pypi.org/project/matheel/), then report the mismatch in GitHub Issues.

For automated checks and contributor setup, see [Development](development.md). Release-facing notebook references are validated against `pyproject.toml` in CI.
