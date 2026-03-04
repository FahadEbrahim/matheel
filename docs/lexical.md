# Edit Distance and Feature Weights

Matheel can blend lexical similarity with semantic and code-aware scores.

## Lexical Features

- `levenshtein`
  Normalized edit-distance similarity.
- `jaro_winkler`
  String similarity that rewards shared prefixes and near matches.

These are most useful when:

- identifier changes are small
- formatting is similar
- you want a lightweight non-embedding metric in the final score

## Parameter

- `feature_weights`

## Canonical Weight Format

Use `feature_weights` as a dictionary or repeated `name=value` entries:

```python
feature_weights = {
    "semantic": 0.6,
    "levenshtein": 0.25,
    "jaro_winkler": 0.15,
}
```

CLI:

```bash
matheel compare sample_pairs.zip \
  --feature-weight semantic=0.6 \
  --feature-weight levenshtein=0.25 \
  --feature-weight jaro_winkler=0.15
```

## Normalization

Matheel normalizes the supplied weights automatically.

That means:

- values only need to be non-negative
- they do not need to sum to `1.0`
- the final blend stays stable after normalization

## Default Blend

If you omit `feature_weights`, Matheel uses:

```python
{
    "semantic": 0.7,
    "levenshtein": 0.3,
    "jaro_winkler": 0.0,
}
```

## Practical Guidance

- Start with semantic-heavy blends for transformer/model runs.
- Increase `levenshtein` when code is very short or formatting is stable.
- Increase `jaro_winkler` when identifier-level similarity matters.
- Add `code_metric` only when you want structure-aware scoring in the final blend.
