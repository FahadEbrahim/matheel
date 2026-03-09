# Lexical Metrics, Baselines, and Feature Weights

Matheel can blend lexical similarity with semantic and code-aware scores.

## Lexical Metrics and Baselines

- `levenshtein`
  Normalized edit-distance similarity.
- `jaro_winkler`
  String similarity that rewards shared prefixes and near matches.
- `winnowing`
  Token fingerprint overlap based on k-grams and sliding-window minima.
- `gst`
  Greedy String Tiling over token sequences.

These are most useful when:

- identifier changes are small
- formatting is similar
- you want a lightweight non-embedding metric in the final score
- you want token-level overlap without loading an embedding model

## Parameters

- `feature_weights`
- `levenshtein_weights`
- `jaro_winkler_prefix_weight`
- `winnowing_kgram`
- `winnowing_window`
- `gst_min_match_length`

## Canonical Weight Format

Use `feature_weights` as a dictionary or repeated `name=value` entries:

```python
feature_weights = {
    "semantic": 0.5,
    "levenshtein": 0.25,
    "jaro_winkler": 0.1,
    "winnowing": 0.1,
    "gst": 0.05,
}
```

CLI:

```bash
matheel compare sample_pairs.zip \
  --feature-weight semantic=0.5 \
  --feature-weight levenshtein=0.25 \
  --feature-weight jaro_winkler=0.1 \
  --feature-weight winnowing=0.1 \
  --feature-weight gst=0.05
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

## Baseline-Specific Options

CLI:

```bash
matheel compare sample_pairs.zip \
  --feature-weight winnowing=1.0 \
  --winnowing-kgram 5 \
  --winnowing-window 4
```

```bash
matheel compare sample_pairs.zip \
  --feature-weight gst=1.0 \
  --gst-min-match-length 5
```

## Practical Guidance

- Start with semantic-heavy blends for transformer/model runs.
- Increase `levenshtein` when code is very short or formatting is stable.
- Increase `jaro_winkler` when identifier-level similarity matters.
- Add `winnowing` when you want a token-fingerprint baseline that stays robust to small local edits.
- Add `gst` when you want to emphasize longer copied token spans.
- Add `code_metric` only when you want structure-aware scoring in the final blend.
