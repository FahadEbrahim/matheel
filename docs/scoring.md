# Scoring and Calibration

Matheel scores are useful for ranking pairs inside one run. They are not automatically calibrated probabilities.

Different algorithms expose different evidence:

- embedding backends compare vector geometry
- Levenshtein and Jaro-Winkler compare string similarity
- Winnowing and GST compare token overlap
- code metrics compare code-specific structure, tokens, or model alignments

Use raw scores for inspection and ranking. Use labeled data when you need thresholds that carry a specific meaning such as "likely same solution" or "needs review".

## Score Ranges

Use `feature_score_range(...)` when you need the documented range in code.

| Feature or metric | Default range | Interpretation |
| --- | --- | --- |
| `semantic` with `cosine` | `-1.0` to `1.0` | Larger means closer vector direction. |
| `semantic` with `dot` | unbounded | Larger means stronger dot product, but magnitude depends on vector scale. |
| `semantic` with `euclidean` | negative distance to `0.0` | `0.0` means identical vectors; farther vectors are more negative. |
| `semantic` with `manhattan` | negative distance to `0.0` | `0.0` means identical vectors; farther vectors are more negative. |
| normalized semantic scores | `0.0` to `1.0` | Enable `normalize_semantic_scores=True` before blending raw `dot`, `euclidean`, or `manhattan` scores with other metrics. |
| `levenshtein` | `0.0` to `1.0` | Normalized edit-distance similarity. |
| `jaro_winkler` | `0.0` to `1.0` | String similarity with prefix weighting. |
| `winnowing` | `0.0` to `1.0` | Fingerprint overlap. |
| `gst` | `0.0` to `1.0` | Covered-token overlap from greedy string tiling. |
| `code_metric` | usually `0.0` to `1.0` | Depends on the selected code metric. Built-in CodeBLEU, CrystalBLEU, RUBY, TSED, and CodeBERTScore paths are intended to be bounded. |

Weighted blends are arithmetic blends of selected feature scores after feature weights are normalized. A blended score is only as comparable as the component scales. Matheel guards mixed-feature blends for raw unbounded or distance-style semantic scores; enable semantic normalization when you need a 0-1 blend.

## Thresholds

A threshold should be calibrated for the dataset and workflow that will use it.

Good threshold labels are operational:

- "same assignment solution"
- "manual review needed"
- "known clone pair"
- "not related"

Avoid treating a threshold from one model, language, assignment, or preprocessing setup as universal.

## Calibrating on Labeled Pairs

Use a small labeled sample that matches the workflow:

1. Collect representative code pairs.
2. Label each pair with the decision you care about.
3. Score the pairs with the same Matheel options you plan to use later.
4. Choose the threshold that optimizes the metric you care about.
5. Re-check the threshold when you change models, metrics, languages, preprocessing, chunking, or feature weights.

```python
from matheel.calibration import calibrate_threshold

labeled_scores = [
    (0.95, True),
    (0.82, True),
    (0.61, False),
    (0.40, False),
]

report = calibrate_threshold(labeled_scores)
print(report["threshold"])
print(report["precision"], report["recall"], report["f1"])
```

`calibrate_threshold(...)` accepts either `(score, label)` pairs or mappings:

```python
from matheel.calibration import calibrate_threshold

report = calibrate_threshold(
    [
        {"similarity_score": 0.92, "label": "same"},
        {"similarity_score": 0.48, "label": "different"},
    ],
    score_key="similarity_score",
    label_key="label",
    positive_label="same",
)
```

Use `evaluate_threshold(...)` when you already have a threshold and want confusion counts, precision, recall, F1, and accuracy.

## Threshold Tuning Reports

Use `tune-threshold` for a reproducible threshold sweep over scored pair rows. Unlike the fuller calibration report, this workflow also works for all-positive or all-negative samples and records a warning when ROC or precision-recall summaries are not meaningful.

```bash
matheel tune-threshold scored_pairs.csv \
  --score-column similarity_score \
  --label-column label \
  --optimize f1 \
  --output-dir threshold_tuning
```

The output includes:

- `threshold_tuning_threshold_sweep.csv`: threshold, confusion counts, precision, recall, F1, and accuracy.
- `threshold_tuning_summary.json`: selected threshold and reproducibility settings.
- `threshold_tuning_report.json`: machine-readable report payload.
- `threshold_tuning_report.html`: local HTML summary.

Python usage:

```python
from matheel.calibration import write_threshold_tuning_report_artifacts

report, artifacts = write_threshold_tuning_report_artifacts(
    scored_pairs,
    "threshold_tuning",
    score_key="similarity_score",
    label_key="label",
    optimize="f1",
)

print(report["summary"]["optimized_threshold"]["threshold"])
print(artifacts["threshold_sweep_csv"])
```

## Calibration Reports

For scored pair CSVs, export a threshold sweep, ROC curve, precision-recall curve, and summary JSON:

```bash
matheel calibration-report scored_pairs.csv \
  --score-column similarity_score \
  --label-column label \
  --output-dir calibration_artifacts
```

The output includes:

- `calibration_threshold_sweep.csv`: threshold, confusion counts, precision, recall, F1, and accuracy.
- `calibration_roc.csv`: threshold rows with false-positive and true-positive rates.
- `calibration_precision_recall.csv`: threshold rows with precision and recall.
- `calibration_summary.json`: AUROC, average precision, and the optimized threshold.
- `calibration_report.json`: all curve rows plus the summary in one JSON artifact.

The ROC and precision-recall reports require at least one positive and one negative label. Single-class labeled samples fail clearly because AUROC and average precision are not meaningful there.

```python
from matheel.calibration import calibration_report

report = calibration_report(
    [
        {"similarity_score": 0.95, "label": 1},
        {"similarity_score": 0.82, "label": 1},
        {"similarity_score": 0.30, "label": 0},
        {"similarity_score": 0.10, "label": 0},
    ],
    score_key="similarity_score",
    label_key="label",
)

print(report["summary"]["auroc"])
print(report["summary"]["optimized_threshold"]["threshold"])
```

## Comparing Runs

When comparing algorithms or configurations:

- keep the input set fixed
- keep labels separate from scoring
- compare ranking quality and calibrated threshold quality, not just raw averages
- report active `feature_weights`, `vector_backend`, `code_metric`, preprocessing, and chunking settings
- avoid comparing raw `dot` or negative distance scores directly against 0-1 metrics

The comparison suite records `feature_set`, `vector_backend`, `code_metric`, `chunking_method`, summary score statistics, and elapsed time. Use those fields to keep comparisons auditable.
