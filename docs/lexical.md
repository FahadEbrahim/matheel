# Lexical Metrics and Feature Weights

Lexical features run with the base Matheel install and can be used alone or blended with semantic and code-aware scores. Start with [Choose similarity metrics](choosing_metrics.md) if you are unsure which comparison fits your task.

## RapidFuzz Distance Families

All ten scalar distance families from [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/index.html) are available as normalized similarities in the `0.0`–`1.0` range. Larger scores mean more similar prepared strings.

| Feature name | What it compares |
| --- | --- |
| `levenshtein` | Insertions, deletions, and substitutions; configurable edit costs. |
| `damerau_levenshtein` | Edits including adjacent transpositions. |
| `osa` | Optimal string alignment, with restricted transpositions. |
| `hamming` | Positional character matches, padding unequal lengths. |
| `indel` | Insertions and deletions. |
| `lcs` | Longest common subsequence, normalized by the longer length. |
| `jaro` | Matching characters and transpositions. |
| `jaro_winkler` | Jaro similarity with a shared-prefix bonus. |
| `prefix` | Shared beginning of the strings. |
| `postfix` | Shared ending of the strings. |

`lcs` uses RapidFuzz's `LCSseq.normalized_similarity`: matching characters may have gaps, but keep their relative order. It is a character subsequence score, rather than a longest contiguous substring or a parser-token comparison. See [RapidFuzz LCSseq](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/LCSseq.html).

Hamming uses `pad=True`, so unequal-length code does not raise a length error. See [RapidFuzz Hamming](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Hamming.html).

## RapidFuzz Fuzzy Ratios

Matheel exposes every scalar scorer in `rapidfuzz.fuzz` with the `rapidfuzz_` prefix. Upstream `0`–`100` scores are divided by `100` for blending.

| Feature name | Matching strategy |
| --- | --- |
| `rapidfuzz_ratio` | Normalized Indel similarity. |
| `rapidfuzz_partial_ratio` | Match the shorter string within the longer one. |
| `rapidfuzz_token_sort_ratio` | Sort whitespace-separated words, then compare. |
| `rapidfuzz_token_set_ratio` | Compare unique and shared words. |
| `rapidfuzz_token_ratio` | Maximum of token-sort and token-set scores. |
| `rapidfuzz_partial_token_sort_ratio` | Partial matching after sorting words. |
| `rapidfuzz_partial_token_set_ratio` | Partial matching of word sets. |
| `rapidfuzz_partial_token_ratio` | Maximum of the partial token scores. |
| `rapidfuzz_wratio` | Upstream weighted combination of ratios. |
| `rapidfuzz_qratio` | Quick ratio with upstream empty-input behavior. |

See the [RapidFuzz fuzzy scorer documentation](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html). Partial matching and token subsets can score highly despite extra code; validate against labeled pairs.

Token ratios use whitespace splitting, independent of `lexical_tokenizer`. Matheel adds no RapidFuzz processor; preprocessing controls case and punctuation. Empty-input behavior is preserved: two empty strings score `1.0` with `rapidfuzz_ratio`, but `0.0` with `rapidfuzz_qratio`. `partial_ratio_alignment` returns alignment data and is not a scalar feature.

## Code-Token Baselines

- `winnowing`: token fingerprints from k-grams and sliding-window minima.
- `gst`: Greedy String Tiling over token sequences.

These use Matheel's code-token regex after preprocessing. Set `lexical_tokenizer="parser"` or CLI `--lexical-tokenizer parser` to compare tree-sitter leaf node types instead. Parser-derived streams reduce sensitivity to identifier spelling; see [Tokenization and limits](tokenization.md).

## Feature Weights

Run `matheel metrics` for the current string metric catalog, including canonical names, families, and score ranges.

Use a dictionary in Python or repeat `--feature-weight name=value` in the CLI. Matheel normalizes finite, non-negative weights; they do not need to sum to one.

```python
from matheel.similarity import calculate_similarity

score = calculate_similarity(
    "def add(a, b): return a + b",
    "def total(x, y): return x + y",
    feature_weights={"lcs": 2, "levenshtein": 1},
    preprocess_mode="basic",
    code_language="python",
)
print(score)
```

From a repository checkout:

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare sample_pairs.zip \
  --feature-weight lcs=2 \
  --feature-weight rapidfuzz_ratio=1
```

`rapidfuzz_ratio` and `indel` measure the same normalized similarity. Use one of them unless duplicating its weight is intentional. Other related scorers also produce correlated evidence; adding more features does not automatically improve quality.

If you omit `feature_weights`, the default remains `semantic=0.7` and `levenshtein=0.3`. That default requires a semantic backend. Set lexical weights explicitly for model-free comparisons. See [Scoring and calibration](scoring.md) before setting thresholds or mixing semantic score scales.

## Baseline Parameters

| Parameter | Used by |
| --- | --- |
| `levenshtein_weights` | Levenshtein insertion, deletion, and substitution costs. |
| `jaro_winkler_prefix_weight` | Jaro-Winkler prefix bonus. |
| `winnowing_kgram`, `winnowing_window` | Winnowing fingerprint generation. |
| `gst_min_match_length` | Minimum GST tile length. |
| `lexical_tokenizer` | Winnowing and GST: `raw` or `parser`. |

```bash
matheel compare sample_pairs.zip \
  --feature-weight winnowing=1 \
  --winnowing-kgram 5 \
  --winnowing-window 4 \
  --lexical-tokenizer raw

matheel compare sample_pairs.zip \
  --feature-weight gst=1 \
  --gst-min-match-length 5 \
  --code-language java \
  --lexical-tokenizer parser
```
