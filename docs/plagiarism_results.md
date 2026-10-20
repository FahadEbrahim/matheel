# Plagiarism benchmark results

`Matheel-Plagiarism-v1` compares **29 method configurations** on a frozen
plagiarism benchmark: **2,284 labeled pairs** across five datasets and **1,180
retrieval queries** across 11 task views. **24 configurations completed every
view**; **5 exceeded the declared 1,800-second method budget** and remain visible
with null scores and no rank.

Open the [interactive official leaderboard](https://huggingface.co/spaces/buelfhood/matheel-framework)
under **Reports → Official Benchmark** to inspect every metric, task view, method
configuration, and protocol identity. The separate **Demo** tab retains its sampled
example results. Pair classification and retrieval have separate rankings. Displayed scores are
rounded to four decimals; ranks use full precision and preserve shared ranks for
exact ties.

## Pair classification

The table is ranked by the mean of dataset average precision (AP) across all five
pair datasets.
AUROC and F1 provide additional views of those same configurations. Scores range
from 0 to 1; higher is better. F1 uses a fixed threshold of `0.5`, without fitting
on evaluation labels, so its value also reflects score calibration.

| Rank | Method configuration | Mean AP | Mean AUROC | Mean F1 | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | RapidFuzz Token Set Ratio | 0.7845 | 0.8009 | 0.6632 | Complete |
| 2 | RapidFuzz Token Ratio | 0.7843 | 0.7996 | 0.5235 | Complete |
| 3 | Winnowing | 0.7838 | 0.7971 | 0.3511 | Complete |
| 4 | CodeBLEU | 0.7834 | 0.8140 | 0.6899 | Complete |
| 5 | Code-Aware (MiniLM-L6-v2) | 0.7716 | 0.7881 | 0.5589 | Complete |
| 6 | Indel | 0.7656 | 0.7852 | 0.6732 | Complete |
| 6 | RapidFuzz QRatio | 0.7656 | 0.7852 | 0.6732 | Complete |
| 6 | RapidFuzz Ratio | 0.7656 | 0.7852 | 0.6732 | Complete |
| 9 | RapidFuzz Token Sort Ratio | 0.7650 | 0.7797 | 0.5349 | Complete |
| 10 | Damerau-Levenshtein | 0.7647 | 0.7898 | 0.6424 | Complete |
| 11 | Optimal String Alignment (OSA) | 0.7646 | 0.7898 | 0.6424 | Complete |
| 12 | Levenshtein | 0.7646 | 0.7898 | 0.6424 | Complete |
| 13 | Balanced (MiniLM-L6-v2) | 0.7474 | 0.7640 | 0.4915 | Complete |
| 14 | Longest Common Subsequence (LCS) | 0.7379 | 0.7565 | 0.6438 | Complete |
| 15 | Embedding Only (MiniLM-L6-v2) | 0.6998 | 0.7305 | 0.4793 | Complete |
| 16 | PyLate BERT tiny MaxSim | 0.6992 | 0.7419 | 0.4821 | Complete |
| 17 | Model2Vec multilingual MiniLM 256 | 0.6652 | 0.6988 | 0.4767 | Complete |
| 18 | Jaro | 0.6267 | 0.6999 | 0.4826 | Complete |
| 19 | Jaro-Winkler | 0.6260 | 0.7037 | 0.4826 | Complete |
| 20 | Common Postfix | 0.5814 | 0.5990 | 0.0943 | Complete |
| 21 | Common Prefix | 0.5227 | 0.6295 | 0.0861 | Complete |
| 22 | Hamming | 0.5084 | 0.6136 | 0.1093 | Complete |
| 23 | RapidFuzz Partial Token Ratio | 0.3508 | 0.4996 | 0.4822 | Complete |
| 23 | RapidFuzz Partial Token Set Ratio | 0.3508 | 0.4996 | 0.4822 | Complete |
| — | GST | — | — | — | Timed out |
| — | Lexical Only | — | — | — | Timed out |
| — | RapidFuzz Partial Ratio | — | — | — | Timed out |
| — | RapidFuzz Partial Token Sort Ratio | — | — | — | Timed out |
| — | RapidFuzz WRatio | — | — | — | Timed out |

## Retrieval

The table is ranked by mean average precision (MAP) across all 11 retrieval views.
MAP and MRR use complete rankings; nDCG uses `k = 10`.

| Rank | Method configuration | MAP | MRR | nDCG@10 | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Winnowing | 0.9900 | 0.9909 | 0.9927 | Complete |
| 2 | Code-Aware (MiniLM-L6-v2) | 0.9881 | 0.9883 | 0.9908 | Complete |
| 3 | RapidFuzz Token Sort Ratio | 0.9865 | 0.9877 | 0.9896 | Complete |
| 4 | CodeBLEU | 0.9818 | 0.9829 | 0.9864 | Complete |
| 5 | Indel | 0.9780 | 0.9790 | 0.9825 | Complete |
| 5 | RapidFuzz QRatio | 0.9780 | 0.9790 | 0.9825 | Complete |
| 5 | RapidFuzz Ratio | 0.9780 | 0.9790 | 0.9825 | Complete |
| 8 | Balanced (MiniLM-L6-v2) | 0.9738 | 0.9754 | 0.9771 | Complete |
| 9 | Damerau-Levenshtein | 0.9720 | 0.9731 | 0.9775 | Complete |
| 10 | Levenshtein | 0.9720 | 0.9730 | 0.9774 | Complete |
| 10 | Optimal String Alignment (OSA) | 0.9720 | 0.9730 | 0.9774 | Complete |
| 12 | Longest Common Subsequence (LCS) | 0.9615 | 0.9625 | 0.9687 | Complete |
| 13 | RapidFuzz Token Set Ratio | 0.9470 | 0.9477 | 0.9591 | Complete |
| 14 | RapidFuzz Token Ratio | 0.9470 | 0.9477 | 0.9589 | Complete |
| 15 | Embedding Only (MiniLM-L6-v2) | 0.9187 | 0.9247 | 0.9261 | Complete |
| 16 | Model2Vec multilingual MiniLM 256 | 0.9150 | 0.9214 | 0.9256 | Complete |
| 17 | Jaro | 0.9015 | 0.9068 | 0.9153 | Complete |
| 18 | Jaro-Winkler | 0.8471 | 0.8518 | 0.8696 | Complete |
| 19 | Hamming | 0.7294 | 0.7429 | 0.7672 | Complete |
| 20 | Common Postfix | 0.7234 | 0.7348 | 0.7432 | Complete |
| 21 | Common Prefix | 0.7182 | 0.7322 | 0.7377 | Complete |
| 22 | PyLate BERT tiny MaxSim | 0.6986 | 0.7105 | 0.7340 | Complete |
| 23 | RapidFuzz Partial Token Ratio | 0.1284 | 0.1300 | 0.1518 | Complete |
| 23 | RapidFuzz Partial Token Set Ratio | 0.1284 | 0.1300 | 0.1518 | Complete |
| — | GST | — | — | — | Timed out |
| — | Lexical Only | — | — | — | Timed out |
| — | RapidFuzz Partial Ratio | — | — | — | Timed out |
| — | RapidFuzz Partial Token Sort Ratio | — | — | — | Timed out |
| — | RapidFuzz WRatio | — | — | — | Timed out |

## Scope and limitations

Each declared task view has equal aggregate weight within its evaluation task.
Each pair dataset contributes one fifth of a classification average. The ten
SOCO14 language/scenario views contribute ten elevenths of a retrieval average;
IRPlag contributes one eleventh. Inspect individual views alongside these averages.
There is no combined score across classification and retrieval.

SOCO14 uses valid judged endpoints as its candidate pools, with separate languages
and scenarios and self matches excluded. These pools differ from the original
competition's full candidate corpus. The frozen preparation records 47 annotation
rows referencing missing source files and excludes two views with no valid positive
judgments. IRPlag retrieval includes its 355 positively judged queries; its pair
classification task also retains the 105 queries without positive judgments.

Semantic configurations use pinned trained models on CPU with a 256-token
limit and no chunking or preprocessing. Lexical methods use the raw source text.
These are declared baseline configurations without test-label parameter tuning.
All supplied source labels are evaluation data; this is not a newly constructed
training/test split or a claim of unseen-dataset generalization. Source definitions
and derived label policies are preserved. SOCO14 records code reuse; its inclusion
in this track does not infer misconduct. Future code-clone datasets and detectors
will require their own task definitions and rankings.

Every ranked method has finite scores on all 16 views. Timed-out methods
(GST, Lexical Only, RapidFuzz Partial Ratio, RapidFuzz Partial Token Sort Ratio, RapidFuzz WRatio) have no aggregate rank or partial average. Each method
ran in a separate process with one CPU thread; the supervisor used three workers
and the predeclared 1,800-second wall-clock budget. Wall time includes model
loading and evaluation and depends on host hardware and concurrent execution.
A timeout describes feasibility in this run.
[Performance follow-up #293](https://github.com/FahadEbrahim/matheel/issues/293)
tracks profiling and exact scoring optimizations for these configurations.

## Reproduce and audit

The portable protocol ID is
`8ccc4f217ff16693d99024e959945dfc5c5e9ec403c1890712c778c3b5875692`.
Sources, model revisions, selection rules, thresholds, candidate counts, and
method configurations are frozen and recorded with content fingerprints.

- [Frozen benchmark and preparation recipe](https://github.com/FahadEbrahim/matheel/tree/v0.7.0/benchmarks/plagiarism-v1).
- [Protocol and validation guide](benchmark_protocol.md).
- [Canonical result JSON](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1.json),
  [aggregate CSV](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1_aggregate.csv), and
  [per-dataset CSV](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1_per_dataset.csv).
- [Score-file index](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/score_index.json),
  [publication hashes](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/publication.json), and
  [reference reproducibility metadata](https://github.com/FahadEbrahim/matheel/blob/v0.7.0/gradio_app/benchmark_data/plagiarism-v1/matheel_plagiarism_v1_reproducibility.json).
- [Release assets](https://github.com/FahadEbrahim/matheel/releases/tag/v0.7.0): the
  [audited scored-row ZIP](https://github.com/FahadEbrahim/matheel/releases/download/v0.7.0/matheel-plagiarism-v1-scores.zip) and
  [runtime constraints](https://github.com/FahadEbrahim/matheel/releases/download/v0.7.0/matheel-plagiarism-v1-runtime-constraints.txt).
- [Leaderboard guide](leaderboard.md) for custom experiments and artifact inspection.

The ZIP contains **384 audited score CSVs**, their index,
and the portable protocol. Its SHA-256 is
`5d55316558aa94f5cce2bad20ac20b8d0bf357b9d5caa5b055cceb5d5748c149`.
Only indexed, validated scored rows are included. Source submissions, local
protocols, model caches, partial worker output, checkpoints, and logs are excluded.
The canonical public bundle is stored once under
`gradio_app/benchmark_data/plagiarism-v1/`, which also supplies the hosted app.
