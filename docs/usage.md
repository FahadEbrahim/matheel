# Usage Guide

This page is the quick-start entry point for installing Matheel and running the main CLI and Python workflows. Detailed parameter guides live in the topic pages linked below.

## Installation

Matheel supports Python `3.10` to `3.13`.

Base install:

```bash
pip install matheel
```

The base package includes the CLI, preprocessing, lexical similarity, and the core comparison workflow. Install optional extras when you need larger semantic backends, chunkers, metric runtimes, or the Gradio app:

```bash
pip install "matheel[semantic]"
pip install "matheel[chunking]"
pip install "matheel[metrics]"
pip install "matheel[gradio]"
pip install "matheel[all]"
```

| Extra | Use it for |
| --- | --- |
| `matheel[semantic]` | Sentence Transformers, Model2Vec, and PyLate semantic scoring backends. |
| `matheel[chunking]` | Chonkie chunkers for splitting code before embedding. |
| `matheel[metrics]` | Optional code metric runtimes such as TSED and CodeBERTScore. |
| `matheel[gradio]` | Dependencies for running the Gradio web app. |
| `matheel[all]` | All supported optional backends in one install. |

Compatibility extras remain available for narrower installs: `sentence_transformers`, `model2vec`, `pylate`, and `chunking_code`.

Examples that use semantic weights assume `matheel[semantic]` or `matheel[all]` is installed. Optional installs can take some time because they may include model and ML runtime dependencies.

## Quick Checks

The repository root includes `sample_pairs.zip`, a small Java archive you can use immediately.

Base CLI check:

```bash
matheel compare sample_pairs.zip \
  --feature-weight levenshtein=1.0 \
  --num 10
```

Base Python check:

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b):\n    return a + b\n",
    "def add(x, y):\n    return x + y\n",
    feature_weights={"levenshtein": 1.0},
)
print(round(score, 4))
```

Semantic CLI check:

```bash
matheel compare sample_pairs.zip \
  --model huggingface/CodeBERTa-small-v1 \
  --feature-weight semantic=0.7 \
  --feature-weight levenshtein=0.3
```

Semantic Python check:

```python
from matheel.similarity import get_sim_list

results = get_sim_list(
    "sample_pairs.zip",
    model_name="huggingface/CodeBERTa-small-v1",
    feature_weights={"semantic": 0.7, "levenshtein": 0.3},
)
print(results.head())
```

## Common Workflows

- Use a ZIP archive or a directory path with `matheel compare`.
- Use `feature_weights` to combine semantic, lexical, and code-aware scores.
- Use `--normalize-semantic-scores` when blending `dot`, `euclidean`, or `manhattan` semantic scores with other 0-1 metrics.
- Add `--preprocess-mode` when code should be normalized before scoring.
- Add `--chunking-method` when large files should be split before embedding.
- Use `matheel compare-suite` with a JSON config for repeatable multi-run comparisons.
- Use `--algorithm-path` when you need a custom `score_pair` implementation.
- Run the Gradio app or notebooks when you want an interactive workflow.

## Demos and Examples

- Hugging Face Space demo: [buelfhood/matheel-framework](https://huggingface.co/spaces/buelfhood/matheel-framework)
- Gradio Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_gradio_colab_demo.ipynb)
- Examples Colab notebook: [Open in Colab](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_examples_colab.ipynb)
- Examples folder: [github.com/FahadEbrahim/matheel/tree/main/examples](https://github.com/FahadEbrahim/matheel/tree/main/examples)

## Documentation Map

- [Preprocessing](preprocessing.md)
- [Tokenization and preprocessing limits](tokenization.md)
- [Chunking](chunking.md)
- [Vectors and routing](vectors.md)
- [Lexical metrics and baselines](lexical.md)
- [Code metrics](code_metrics.md)
- [Scoring and calibration](scoring.md)
- [Custom algorithms](customization.md)
- [Comparison suite](comparison_suite.md)
- [Development](development.md)

## Interface Notes

- CLI and Python API accept either a directory or a ZIP archive.
- Gradio remains ZIP-only for uploads.
- `feature_weights` is the canonical scoring input.
- `vector_backend=auto` uses Hugging Face metadata and tag heuristics when available.
- CLI progress bars write to stderr and default to interactive terminals only. Use `--progress` or `--no-progress` to override.
- Python APIs accept `progress=True` for tqdm bars and `progress_callback=...` for structured progress events.
- Collection results include run metadata in `DataFrame.attrs`, including `elapsed_seconds`, `feature_set`, `vector_backend`, `code_metric`, `chunking_method`, and custom algorithm metadata when applicable.
