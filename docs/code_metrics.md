# Code Metrics

Matheel includes built-in code-aware metrics that can be blended into the final score or used on their own.

## Parameters

- `code_metric`
- `code_metric_weight`
- `code_language`
- `codebleu_component_weights`
- `crystalbleu_max_order`
- `crystalbleu_trivial_ngram_count`

## Supported Metrics

- `none`
- `codebleu`
- `codebleu_ngram`
- `codebleu_weighted_ngram`
- `codebleu_syntax`
- `codebleu_dataflow`
- `crystalbleu`

## Language Scope

For the strongest and safest interpretation, CodeBLEU-style metrics are scoped to:

- `java`
- `python`
- `c`
- `cpp`

CrystalBLEU is token-based and generally broader, but those four languages are still the best-documented scope for code-aware evaluation.

## Metric Details

### CodeBLEU-style Metrics

Matheel provides a built-in CodeBLEU-style implementation, so no external `codebleu` package is required.

- `codebleu`
  Full weighted blend of the CodeBLEU components.
- `codebleu_ngram`
  Surface n-gram overlap.
- `codebleu_weighted_ngram`
  Keyword-weighted n-gram overlap.
- `codebleu_syntax`
  Syntax-oriented component.
- `codebleu_dataflow`
  Dataflow-oriented component.

`codebleu_component_weights` is a comma-separated string:

```text
ngram,weighted_ngram,syntax,dataflow
```

Default:

```text
0.25,0.25,0.25,0.25
```

### CrystalBLEU

CrystalBLEU discounts frequent “trivial” n-grams.

- `crystalbleu_max_order`
  Maximum n-gram order.
- `crystalbleu_trivial_ngram_count`
  Number of high-frequency n-grams to ignore.

For very small toy examples, set `crystalbleu_trivial_ngram_count` lower than the default or even `0`, otherwise tiny inputs may collapse toward `0.0`.

## Blending Into Final Score

To use code metrics as part of the final score:

- choose a `code_metric`
- set `code_metric_weight`
- include `code_metric` in `feature_weights`, or let Matheel add it automatically when only `code_metric_weight` is provided

## Python Example

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b): return a + b",
    "def sum_two(x, y): return x + y",
    code_metric="codebleu",
    code_metric_weight=0.2,
    code_language="python",
    feature_weights={"semantic": 0.8, "code_metric": 0.2},
)
print(score)
```
