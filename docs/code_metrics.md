# Code Metrics

Matheel includes built-in code-aware metrics that can be blended into the final score or used on their own.

## Parameters

- `code_metric`
- `code_metric_weight`
- `code_language`
- `codebleu_component_weights`
- `crystalbleu_max_order`
- `crystalbleu_trivial_ngram_count`
- `ruby_max_order`
- `ruby_epsilon`
- `ruby_mode`
- `ruby_tokenizer`
- `ruby_denominator`
- `ruby_graph_timeout_seconds`
- `ruby_graph_use_edge_cost`
- `ruby_graph_include_leaf_edges`
- `ruby_tree_max_nodes`
- `ruby_tree_max_depth`
- `ruby_tree_max_children`
- `tsed_delete_cost`
- `tsed_insert_cost`
- `tsed_rename_cost`
- `tsed_max_nodes`
- `tsed_max_depth`
- `tsed_max_children`
- `codebertscore_model`
- `codebertscore_num_layers`
- `codebertscore_batch_size`
- `codebertscore_max_length`
- `codebertscore_device`
- `codebertscore_lang`
- `codebertscore_idf`
- `codebertscore_rescale_with_baseline`
- `codebertscore_use_fast_tokenizer`
- `codebertscore_nthreads`
- `codebertscore_verbose`

## Supported Metrics

- `none`
- `codebleu`
- `codebleu_ngram`
- `codebleu_weighted_ngram`
- `codebleu_syntax`
- `codebleu_dataflow`
- `crystalbleu`
- `ruby`
- `tsed`
- `codebertscore`

## Language Scope

For the strongest and safest interpretation, CodeBLEU-style metrics are scoped to:

- `java`
- `python`
- `c`
- `cpp`

RUBY and TSED follow the same language scope (`java`, `python`, `c`, `cpp`).
CodeBERTScore is language-agnostic at runtime, but use the same four-language scope for consistent code-level comparisons.

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

### RUBY

RUBY uses a staged similarity strategy:

1. graph similarity (when optional graph dependencies are available)
2. tree similarity
3. string similarity as a deterministic fallback

`ruby_mode` controls this behavior (`auto`, `graph`, `tree`, `string`, `ngram`).

- `ruby_max_order`
  Maximum n-gram order (used when `ruby_mode=ngram`).
- `ruby_epsilon`
  Small smoothing value for n-gram mode edge cases.
- `ruby_tokenizer`
  Tokenizer used by string mode (`tranx` or `regex`).
- `ruby_denominator`
  String-mode normalization denominator (`max` or `mean`).
- `ruby_graph_timeout_seconds`
  Per-step timeout for graph-edit search.
- `ruby_graph_use_edge_cost`
  Include edge insertion/deletion costs in graph mode.
- `ruby_graph_include_leaf_edges`
  Add sequential leaf edges in graph mode.
- `ruby_tree_max_nodes`, `ruby_tree_max_depth`, `ruby_tree_max_children`
  Parse-budget controls for tree/graph modes.

### TSED

TSED compares syntax trees using tree edit distance.

- `tsed_delete_cost`
- `tsed_insert_cost`
- `tsed_rename_cost`
- `tsed_max_nodes`
- `tsed_max_depth`
- `tsed_max_children`

TSED requires optional dependencies (`apted` and a tree-sitter runtime package).

### CodeBERTScore

CodeBERTScore uses transformer token alignment to score similarity.

- `codebertscore_model`
- `codebertscore_num_layers`
- `codebertscore_batch_size`
- `codebertscore_max_length`
- `codebertscore_device`
- `codebertscore_lang`
- `codebertscore_idf`
- `codebertscore_rescale_with_baseline`
- `codebertscore_use_fast_tokenizer`
- `codebertscore_nthreads`
- `codebertscore_verbose`

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

codebertscore_only = calculate_similarity(
    "def add(a, b): return a + b",
    "def sum_two(x, y): return x + y",
    code_metric="codebertscore",
    code_metric_weight=1.0,
    codebertscore_model="microsoft/codebert-base",
    feature_weights={"code_metric": 1.0},
)
print(codebertscore_only)
```
