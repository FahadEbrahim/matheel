# Vectors and Routing

Matheel supports three public vector backends:

- `sentence_transformers`
- `model2vec`
- `pylate`

You can also use `vector_backend=auto` and let Matheel route based on Hugging Face metadata and tags.

## Parameters

- `vector_backend`
- `max_token_length`
- `similarity_function`
- `pooling_method`
- `static_vector_dim`
- `static_vector_lowercase`
- `multivector_bidirectional`
- `device`

## Backends

### `sentence_transformers`

Dense single-vector embeddings through Sentence Transformers.

Best when:

- you want a widely supported default path
- you want dense semantic similarity
- you want to control `similarity_function` and `pooling_method`

### `model2vec`

Learned static single-vector embeddings via `model2vec`.

Best when:

- you want a true static embedding model
- you want lighter-weight inference than dense transformer pooling in some setups
- your selected Hugging Face model is explicitly a `model2vec` model

### `pylate`

Multivector late-interaction scoring via PyLate/ColBERT-style models.

Best when:

- you want token- or chunk-level late interaction
- you want higher-fidelity multivector matching
- your selected Hugging Face model is tagged for `PyLate` or `ColBERT`

## Auto Routing

`vector_backend=auto` checks model metadata and tags and prefers:

- Sentence Transformers for dense models
- model2vec for static models
- PyLate for multivector models

If metadata is unavailable, Matheel falls back to simple name/tag heuristics and finally defaults to Sentence Transformers.

## Max Token Length

Use `max_token_length` to cap the sequence length used by the selected model.

- `None` or `0` keeps the model default
- a positive integer applies a shorter cap
- values above the detected model limit are clamped to that detected limit

The helper `inspect_model_settings(...)` returns both:

- `detected_max_token_length`
- `configured_max_token_length`

That is the API the Gradio model picker can use to show a safe token-length control after model selection.

## Similarity Functions

These apply to single-vector backends:

- `cosine`
- `dot`
- `euclidean`
- `manhattan`

Notes:

- `cosine` and `dot` are the most common similarity choices.
- `euclidean` and `manhattan` are distance-style scores, so thresholds may need to be lower and can be negative.

## Pooling Methods

These apply to Sentence Transformers single-vector scoring:

- `mean`
- `max`
- `cls`
- `lasttoken`
- `mean_sqrt_len_tokens`
- `weightedmean`

`pooling_method` is ignored by `model2vec` and `pylate`.

## Backend-Specific Parameters

- `static_vector_dim`
  Compatibility fallback dimension used by internal local static-vector fallback paths.
- `static_vector_lowercase`
  Whether local static-token preprocessing lowercases tokens.
- `multivector_bidirectional`
  If enabled, late interaction averages both scoring directions.
- `device`
  Runtime device such as `auto`, `cpu`, `mps`, or `cuda` when supported.

## Python Example

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b): return a + b",
    "def add(x, y): return x + y",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_backend="sentence_transformers",
    max_token_length=256,
    similarity_function="dot",
    pooling_method="max",
    feature_weights={"semantic": 1.0},
)
print(score)
```
