# Contributing Dataset Support

Dataset support is split into source resolvers, adapters, and presets.

- A **source resolver** locates or downloads raw files.
- An **adapter** converts a raw layout into normalized Matheel pair or retrieval manifests.
- A **preset** records an approved public dataset source plus its adapter.

Keep these pieces separate so custom users can reuse generic sources and adapters without treating every source as an endorsed dataset.

## Dataset Contribution Checklist

- Declare the dataset domain with `task_type`, independently of its protocol. Current public presets use `plagiarism`; a future clone dataset would use `code_clone`. Stable snake_case domains are extensible without changing an enum.
- Confirm the task family: `pair`, `retrieval`, or both. These protocol names are not dataset domains.
- Describe the ground-truth definition and its limits. Record languages and file/function/fragment granularity only when established by the source; do not infer plagiarism from clone labels.
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

If the adapter handles several domains, accept and validate `options.get("task_type")` with `normalize_dataset_task_type` instead of assigning plagiarism unconditionally. Domain-specific adapters should reject a conflicting requested domain. Built-in generic adapters already propagate this option. Loading rejects conflicts between a preset, explicit spec, adapter options, and normalized metadata.

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
        "task_type": "plagiarism",
        "adapter": "internal_pair",
        "task_families": ("pair",),
    },
    overwrite=True,
)
```

Built-in presets should be added only for approved datasets. Generic resolvers such as GitHub, Zenodo, Hugging Face, and Kaggle are source mechanisms, not endorsements by themselves.

## Tests

Use synthetic layouts that mirror the expected file structure:

- Adapter test: raw fixture in a temporary directory, run adapter, load the normalized dataset, assert counts, labels, and domain. Derive generated identifiers from source-relative paths so moving a download cache preserves the normalized dataset fingerprint.
- Category test: verify the domain remains distinct from pair/retrieval protocol, explicit metadata conflicts are rejected, and mixed-domain datasets cannot be merged.
- Preset test: resolve through a local fake source or monkeypatched resolver, never the network.
- Manifest test: relative paths resolve from the manifest location.
- Security test: unsafe archives or credential fields are rejected when relevant.

## Pull Request Expectations

- Add docs for source terms and user-provided downloads.
- Add a small runnable example only when it does not require real dataset files.
- Run:

```bash
python -m pytest -q
python -m ruff check .
python -m mkdocs build --strict
git diff --check
```

See [Development](development.md) for the development and documentation dependency installs.
