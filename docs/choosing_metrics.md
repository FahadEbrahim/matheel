# Choose Similarity Metrics

Start with a small baseline set and compare it on representative labeled data. A high code-similarity score is evidence for inspection; it does not establish shared behavior, copying, or misconduct by itself.

## Available in Matheel

| Goal | Starting point | What to check |
| --- | --- | --- |
| Compare whole files with small edits | `levenshtein`, `indel`, or `lcs` | Formatting, repeated boilerplate, and identifier spelling affect string scores. |
| Match reordered or repeated words | `rapidfuzz_token_sort_ratio` or `rapidfuzz_token_set_ratio` | Whitespace words are not code-parser tokens; order or repetition may be lost. |
| Find a short snippet inside a larger file | `rapidfuzz_partial_ratio`, `winnowing`, or `gst` | Partial matching can hide substantial extra code; inspect matched regions. |
| Reduce sensitivity to identifier renaming | Winnowing/GST with `lexical_tokenizer="parser"` | Leaf node types preserve less information than full syntax or dataflow. |
| Compare syntax or data dependencies | CodeBLEU syntax/dataflow, TSED, or RUBY | Supported language, parse quality, runtime budgets, and fallback mode. |
| Compare model representations | `semantic` or CodeBERTScore | Model choice, token limits, optional dependencies, and score scale. |
| Rank candidates for a query | Dense, static, or multi-vector semantic scoring with a lexical baseline | Measure retrieval quality and preserve query/document roles. |

The [lexical reference](lexical.md) lists all RapidFuzz scalar distance and fuzzy-ratio features, including LCS. See [code-aware metrics](code_metrics.md) and [embeddings](vectors.md) for the other implemented options. Prefix, postfix, Hamming, and Jaro-family scores are available for targeted experiments; shared scaffolding and positional changes can make them weak whole-file baselines.

## Compare Before Blending

1. Run `levenshtein=1`, `lcs=1`, and `winnowing=1` separately on the same data. Add GST or a partial scorer if local overlap matters.
2. Fix the preprocessing and tokenizer settings while comparing algorithms. Change those settings in a separate experiment.
3. Inspect false matches and missed matches with [pair explanations](visualization.md#pair-explanations).
4. Add one complementary feature at a time and evaluate the blend. Several edit-based scores can largely repeat the same signal.
5. Select thresholds on a calibration split, then report quality on separate held-out data. If submissions share an assignment, author, or code family, group splits to reduce leakage.

Use [comparison suites](comparison_suite.md) for repeated configurations, [datasets and evaluation](datasets.md) for pair/retrieval metrics, and [scoring and calibration](scoring.md) for score ranges and thresholds. Normalize semantic scores when combining raw dot products or distances with bounded lexical features.

## Recommended Future Additions

The following methods are recommendations for new built-ins. They are **not feature names you can pass to `feature_weights` today**. The suggested code applications are design recommendations; they need benchmark validation.

| Priority | Metric or method | Why add it | Main limitation |
| --- | --- | --- | --- |
| 1 | Token/shingle Jaccard | Explainable overlap baseline: intersection divided by union; configurable token n-grams retain local order. | Set overlap discards multiplicity and longer-range order. |
| 1 | Sørensen–Dice on token shingles | A second overlap scale, `2 * intersection / (size_a + size_b)`, easy to inspect alongside Jaccard. | With the same sets it is a monotonic transform of Jaccard; evaluate one baseline before blending both. |
| 1 | TF-IDF cosine over code tokens or character n-grams | Sparse corpus-aware baseline that downweights common material; fit once for repeated comparisons. | Vocabulary and IDF depend on the fitted corpus. |
| 2 | BM25 | Lexical query-to-code retrieval baseline with term-frequency saturation and document-length normalization. | Asymmetric, corpus-dependent relevance score; needs an explicit mapping before use in a bounded blend. |
| 2 | Smith–Waterman local token alignment | Identify matching regions with gaps, useful for partial-code inspection. | Alignment cost and normalization require explicit limits and evaluation. |
| 2 | MinHash with locality-sensitive hashing | Approximate shingle Jaccard to shortlist candidates before expensive scoring. | An approximate retrieval layer, with seed and recall tradeoffs. |
| 3 | SimHash similarity | Compact fingerprints for fast approximate near-duplicate filtering. | Loses detail; fingerprint distance needs code-specific validation. |
| 3 | Program-dependence-graph similarity | A research path for comparing control/data dependencies under code rearrangement. | Language-specific analysis, expensive matching, and extraction failures. |

Jaccard with shingling and MinHash provide a natural exact/approximate pair; the [information-retrieval text's near-duplicate chapter](https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html) describes their relationship. Dice uses the overlap definition documented in the [SciPy Dice reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html), converted from distance to similarity.

For sparse baselines, see the [scikit-learn text feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) for TF-IDF and character n-grams, and the [BM25 reference](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) for retrieval scoring. Fit corpus statistics on the designated training/index data and record that corpus in reproducibility metadata.

Local alignment comes from [Smith and Waterman's original paper](https://cs.brown.edu/courses/csci1820/spring-2022/resources/Smith_Waterman_1981.pdf). Applying it to code tokens is a proposed extension here. For compact filtering, see the authors' [SimHash near-duplicate study](https://research.google.com/pubs/archive/33026.pdf). Dependence-graph matching has a code-specific precedent in [GPlag](https://cmps-people.ok.ubc.ca/bowenhui/analytics/readings/kdd06_gplag.pdf); this would need richer dependency analysis than simply selecting a tree metric.

Prototype corpus-dependent methods through [custom algorithms](customization.md), using `prepare_dataset` to fit reusable state and `score_pair` to score pairs. MinHash/LSH and SimHash indexing would also need a collection-level candidate-selection API to avoid scoring every possible pair.
