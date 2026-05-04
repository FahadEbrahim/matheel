# Contributing Dataset Support

Dataset support is split into source resolvers, adapters, and presets.

- A **source resolver** locates or downloads raw files.
- An **adapter** converts a raw layout into normalized Matheel pair or retrieval manifests.
- A **preset** records an approved public dataset source plus its adapter.

Keep these pieces separate so custom users can reuse generic sources and adapters without treating every source as an endorsed dataset.

## Dataset Contribution Checklist

- Confirm the task family: `pair`, `retrieval`, or both.
- Confirm license, provenance, access requirements, and citation information.
- Do not commit real dataset files.
- Do not commit credentials or credential-like fields in manifests.
- Use tiny synthetic fixtures in tests.
- Keep source resolvers deterministic and safe. Archive extraction must reject path traversal.
- Add docs that explain user-provided downloads and source terms.

## Adapter Shape

Adapters should write normalized datasets with existing helpers:

```python
from matheel.datasets import write_pair_dataset


def adapt_my_pairs(source_root, destination, **options):
    files = ...
    pairs = ...
    return write_pair_dataset(
        destination,
        files=files,
        pairs=pairs,
        metadata={
            "name": options.get("name", "my_pairs"),
            "task_type": "plagiarism",
            "dataset_kind": "pair_classification",
        },
    )
```

Prefer `write_retrieval_dataset` for retrieval layouts. Use `document_id` for retrieval documents and keep normalized metadata aligned with existing schemas.

## Registering Custom Pieces

```python
from matheel.datasets import register_dataset_adapter, register_dataset_preset, register_dataset_source

register_dataset_source("internal", resolve_internal_dataset, overwrite=True)
register_dataset_adapter("internal_pair", adapt_my_pairs, overwrite=True)
register_dataset_preset(
    "internal_pairs",
    {
        "source": "internal",
        "identifier": "raw_pairs",
        "adapter": "internal_pair",
        "task_families": ("pair",),
    },
    overwrite=True,
)
```

Built-in presets should be added only for approved datasets. Generic resolvers such as GitHub, Zenodo, Hugging Face, and Kaggle are source mechanisms, not endorsements by themselves.

## Tests

Use synthetic layouts that mirror the expected file structure:

- Adapter test: raw fixture in a temporary directory, run adapter, load the normalized dataset, assert counts and labels.
- Preset test: resolve through a local fake source or monkeypatched resolver, never the network.
- Manifest test: relative paths resolve from the manifest location.
- Security test: unsafe archives or credential fields are rejected when relevant.

## Pull Request Expectations

- Add docs for source terms and user-provided downloads.
- Add a small runnable example only when it does not require real dataset files.
- Run:

```bash
uv run python -m pytest -q
uv run python -m ruff check .
uv run python -m mkdocs build --strict
git diff --check
```
