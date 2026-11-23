# Matheel-AIGCodeSet-v1

This benchmark evaluates classifiers on individual source snippets labeled human-authored or AI-generated. It uses the two authoritative AIGCodeSet CSVs at revision `b4d4e69`. The dataset card lists CDLA Permissive 2.0 for the dataset and Apache 2.0 for accompanying source code. [Dataset card](https://huggingface.co/datasets/basakdemirok/AIGCodeSet) · [Versioned tables](https://huggingface.co/datasets/basakdemirok/AIGCodeSet/tree/b4d4e69/data) · [Paper](https://arxiv.org/abs/2412.16594)

## Protocol

The source tables contain 4,755 human and 2,828 generated Python samples across 317 problems. Preparation excludes 206 rows across 103 text conflicts and 26 repeated rows within a problem and cohort. It connects problems that share an exact text, an equal normalized token sequence, or distinct token sequences with at least 95% normalized Indel similarity and eight tokens. Token normalization drops comments and non-significant line breaks while retaining statement breaks and indentation structure. Source without a token signature remains covered by exact text matching. The official run found 314 independent groups after connecting related problems.

A seeded SHA-256 order assigns groups to train, validation, and test partitions. The scratch cohort compares human accepted solutions with AI code generated from the problem description. The models fit on scratch training code only. The validation scratch cohort selects the threshold that maximizes recall with a false-positive ceiling of 5%, breaking ties by lower false-positive rate and then higher threshold. The same threshold is applied to scratch and the two repair cohorts in the test partition. Repair cohorts compare human runtime-error or wrong-answer submissions with AI outputs prompted to repair submissions of the corresponding type. These scenarios use the source's generation context. The benchmark evaluates provenance labels and preserves scenario-level and per-generator breakdowns.

## Reference results

| Scratch test, 501 snippets | Average precision | AUROC | Macro F1 | Recall | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training class prior | 0.3752 | 0.5000 | 0.3845 | 0.0000 | 0.0000 |
| Character TF-IDF and logistic regression | 0.9549 | 0.9702 | 0.8885 | 0.8032 | 0.0447 |

The logistic baseline achieved recall of 0.1919 at a false-positive rate of 0.0654 on runtime repairs, and recall of 0.1149 at a false-positive rate of 0.0300 on wrong-answer repairs. The reference tests problem holdout within this corpus. It has no held-out generator or language track.

The [result manifest](results.json) records source and implementation hashes, split groups, exclusion counts, fitted configuration, runtime environment, rejection threshold, and checksums for the source-free score table. The score table contains validation and test predictions for both baselines. The benchmark data remains in the local source cache.

## Reproduce

The reference used Matheel 0.8.0 and Python 3.14.2 on macOS arm64. Install from the release checkout with the recorded constraints, then fetch the two source CSVs and run the benchmark. Use a writable local cache outside the checkout.

```bash
python -m pip install -c benchmarks/authorship-v1/reference-constraints.txt ".[authorship]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.run_authorship_benchmark run \
  --source-dir /tmp/matheel-authorship-data --output-dir /tmp/matheel-authorship-results \
  --download --compare-to benchmarks/authorship-v1/results.json
```

The runner validates both source file sizes and SHA-256 hashes. An existing local cache can be used offline by omitting `--download`. Exact score digests depend on the Python and numerical-library environment. Compare those versions to the result manifest when assessing runs across machines.

Recompute the operating points and metrics from the published predictions offline.

```bash
python -m scripts.run_authorship_benchmark validate benchmarks/authorship-v1/results.json
```
