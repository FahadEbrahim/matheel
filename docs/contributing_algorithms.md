# Contributing Similarity Algorithms

Matheel supports two contribution paths:

1. A custom algorithm module for local experiments.
2. A built-in Matheel method or preset for reusable package support.

Use the custom path first when prototyping. Move into package code only after the method has a stable contract, deterministic tests, and clear dependencies.

## Start From A Template

Generate a starter module:

```bash
matheel init-custom-algorithm my_algorithm.py
```

Then run it locally:

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare sample_pairs.zip --algorithm-path my_algorithm.py
```

The generated module exposes `score_pair(code_a, code_b, dataset_context=None, row=None, **kwargs)`. It may also define `prepare_dataset(dataset, prepared_texts=None, **kwargs)` for reusable context.

## Custom Algorithm Checklist

- Return a finite numeric score where larger means more similar.
- Accept `row` and `dataset_context` when useful, but do not require them for basic scoring.
- Keep options JSON-serializable so CLI, suite, and reproducibility outputs can store them.
- Avoid network calls and mutable global state during scoring.
- Add tests for direct pair scoring and dataset evaluation when the method is intended for benchmarks.
- Keep private code, credentials, and real datasets out of the repository.

## Built-In Method Checklist

When promoting a method into Matheel:

1. Add the implementation to the smallest relevant module.
2. Expose public helpers only when they are useful outside the implementation.
3. Add deterministic unit tests with tiny synthetic code snippets.
4. Document parameters, dependency requirements, and score range.
5. Add the method to comparison or Gradio controls when it is user-facing.
6. Add an offline leaderboard preset when the method should be benchmarked by default.

## Offline Leaderboard Presets

Leaderboard presets make a method available in manifests, Python, and the Gradio **Reports → Build Leaderboard** workflow.

```python
from matheel import register_leaderboard_algorithm_preset

register_leaderboard_algorithm_preset(
    "My Method",
    {
        "description": "Short benchmark-facing description.",
        "similarity_options": {
            "feature_weights": {"levenshtein": 1.0},
            "code_metric": "codebleu",
            "code_metric_weight": 0.0,
        },
    },
    overwrite=True,
)
```

Use a preset in a leaderboard manifest:

```json
{
  "datasets": [{"name": "pairs", "task": "pair", "path": "./normalized_pairs"}],
  "algorithms": ["My Method"]
}
```

Built-in presets should be added with tests that verify `available_leaderboard_algorithm_presets()`, manifest normalization, and at least one tiny leaderboard run.

## Pull Request Expectations

- Include docs and examples for new user-facing behavior.
- Include tests that run offline.
- Keep optional heavy dependencies import-late or behind extras.
- Run:

```bash
uv run python -m pytest -q
uv run python -m ruff check .
uv run python -m mkdocs build --strict
git diff --check
```
