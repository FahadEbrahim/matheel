# Code Similarity, Clone Discovery, and AI Authorship

Matheel evaluates labeled code pairs, ranks candidates for a code query, measures recall on detected clone spans, and benchmarks single-snippet AI-authorship classifiers. Each protocol defines its own labels, candidate pools, and metrics. Use the release's [benchmark artifacts and reproduction commands](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/research-080) to compare runs on the same scope.

## Code similarity and retrieval

Use `matheel datasets catalog` to see each dataset's domain, source, and access method. The new adapters prepare locally acquired data. Their dataset cards record upstream terms and revisions. For a plain normalized dataset, use `matheel datasets adapt` and the existing [pair or retrieval evaluation](datasets.md). The bounded benchmark command also records baselines, output hashes, and evaluation details.

| Dataset | Adapter | Evaluation |
| --- | --- | --- |
| [SimilCode v2.4.0](https://doi.org/10.5281/zenodo.23949390) | `similcode_pair` | Java and C# similarity pairs. Retains the source categories, reference groups, and independently supplied derived clone types. Binary views define which categories count as positive. The benchmark also reports score distributions and counts by category and clone type. |
| [POJ-104](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104) | `poj104_retrieval` | C/C++ same-problem retrieval. Uses official problem splits, excludes the query from the candidate pool, and reports MAP@R, where R is the query's relevant-candidate count. |
| [XLCoST](https://github.com/reddy-lab-code-research/XLCoST/tree/main/metadata/codesearch) | `xlcost_retrieval` | Directional Java→Python and Python→Java program retrieval on aligned solutions. Each view fixes a target-language corpus and reports first-hit MRR and Recall@k. The upstream multi-target evaluator averages reciprocal ranks differently and has its own published protocol. |
| [CodeXGLUE BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench) | `codexglue_bcb_pair` | Java method-pair classification with official binary labels, ordered IDs, published splits, and binary F1. Preserve the source's BigCloneBench lineage when interpreting results. |

```bash
python -m pip install "matheel[benchmark]"

matheel benchmark-similarity similcode_pair ./data/similcode ./runs/similcode-java \
  --language java --source-revision v2.4.0

matheel benchmark-similarity poj104_retrieval ./data/poj104 ./runs/poj104 --split test

matheel benchmark-similarity xlcost_retrieval ./data/XLCoST_data ./runs/xlcost-java-python \
  --source-language java --target-language python --split test

matheel benchmark-similarity codexglue_bcb_pair ./data/codexglue-bcb ./runs/codexglue-bcb \
  --split test
```

The benchmark command leaves normalized source files in the run directory. Share its source-free metrics, scored rows, configuration, and fingerprints under the applicable data terms. Keep the `dataset/` directory local. Tune a binary threshold on validation data before passing it to a test run. For a smaller smoke test, record the selected population as its own run. Use `benchmark-similarity` for the full POJ-104 scope. The generic Gradio retrieval form retains every scored candidate and is intended for smaller uploads.

## Clone discovery

Pass the source spans that a detector found and the reference spans supplied by the benchmark. Both CSVs identify two files and inclusive start and end lines per row. The reference CSV also declares `clone_type` and can include `project_scope`, `pretty_lines`, `tokens`, `line_similarity`, and `token_similarity`.

| Reference type | Definition |
| --- | --- |
| `type1` | Differences in layout or comments. |
| `type2` | Renaming or literal changes. BigCloneEval reports `type2_consistent` and `type2_blind` separately. |
| `type3_very_strong`, `type3_strong`, `type3_moderate` | Statement changes that BigCloneEval groups by line and token similarity. |
| `weak_type3_type4` | BigCloneEval's combined band for weak syntactic matches and functionally similar code. |

Use the source's labels as recorded. SimilCode supplies its own derived type for each pair alongside its original category.

```bash
matheel import-clone-report ./native/nicad.xml ./runs/nicad-findings.csv \
  --format nicad --source-map ./native/file-map.json --metadata-out ./runs/nicad.json
matheel evaluate-clones ./reference.csv ./runs/nicad-findings.csv \
  --coverage 0.70 --output ./runs/nicad-recall.json
matheel export-bigcloneeval ./runs/nicad-findings.csv ./runs/bigcloneeval.csv \
  --source-map ./native/bigcloneeval-map.json
```

Use a JSON object with report paths as keys and stable corpus-relative IDs as values for NiCad or PMD CPD. SourcererCC uses a CSV mapping `native_id,file_id,start_line,end_line`. The BigCloneEval export map goes from stable IDs to paths in the reference corpus. See [external detectors](external_detectors.md) for import and runner options.

The evaluator counts a reference pair when a reported pair covers the required proportion of both reference spans. It reports eligible and matched reference counts per type and project scope. Review unmatched discoveries using independently judged data if you need precision. The pinned BigCloneEval v1.1 SQL matcher rejected partial overlaps in the controlled replay that Matheel's fractional coverage accepts. Use the [detector evidence](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/research-080/detectors) for the native and Matheel results, matching parameters, and corpus provenance.

Export a typed reference CSV from a local BigCloneEval v1.1 H2 database. For a bounded first run, use `--max-references` and keep the selected scope in the run record. The extractor preserves the source database. An archive that needs H2 recovery can require temporary disk space for a full private copy.

```bash
matheel extract-bigcloneeval-references ./bigclonebenchdb/bcb.h2.db ./runs/reference.csv \
  --java /tools/jdk/bin/java --h2-jar ./BigCloneEval/libs/h2-1.3.176.jar \
  --max-references 1000
```

## AI authorship

[AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet/tree/b4d4e69/data) provides separate human and generated Python CSVs. Use `human_selected_dataset.csv` and `created_dataset_with_llms.csv` from the pinned author revision. The preparation audit records conflicting identical snippets, duplicates, related problem groups, split membership, and sample counts. The first evaluated task uses human accepted solutions and generated-from-scratch code, with results for the repair cohorts reported separately.

```bash
python -m pip install "matheel[authorship]"
matheel authorship prepare-aigcodeset \
  ./data/human_selected_dataset.csv ./data/created_dataset_with_llms.csv \
  ./runs/aigcodeset-private.json --verify-source
matheel authorship benchmark ./runs/aigcodeset-private.json ./runs/aigcodeset --target-fpr 0.05
matheel authorship verify ./runs/aigcodeset/results.json
```

The baselines are a class-prior control and character n-gram TF-IDF with logistic regression. Feature fitting uses training code only. Threshold selection uses validation predictions. The test results include average precision, AUROC, macro-F1, recall, and the achieved false-positive rate. The normalized dataset contains source code. The evaluation JSON and score CSV contain identifiers, labels, scores, and provenance, and can be audited separately.

## Examples and Gradio

Run `python examples/evaluation/research_benchmarks_demo.py` for a synthetic offline clone and authorship example. In the app, open **Reports → Research Benchmarks** to see published results, evaluate uploaded clone span pairs, or reproduce the AI-authorship run from the two pinned source CSVs. Use **Datasets** to evaluate normalized pair and retrieval ZIPs.
