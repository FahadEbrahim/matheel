# Matheel 0.8.0 research benchmarks

The release records separate scopes for similarity pairs, functional retrieval, directional cross-language retrieval, clone discovery, and AI authorship. Each report identifies its inputs, method, evaluator, and coverage. Compare methods within the same scope and metric.

| Scope | Results and reproduction |
| --- | --- |
| SimilCode Java and C#, POJ-104, XLCoST directions, and CodeXGLUE BigCloneBench | [Similarity datasets](similarity/README.md) |
| Typed BigCloneBench v2 recall, NiCad, PMD CPD, SourcererCC, Copydetect, and archived MOSS | [Detector evidence](detectors/README.md) |
| AIGCodeSet Python origin classification on held-out problem groups | [Authorship benchmark](../authorship-v1/README.md) |

The data acquisition steps are part of each result recipe. Source corpora stay in local working directories. Published tables contain metrics, coordinates or opaque IDs, and checksums. Follow the original dataset terms when accessing or sharing data.

The Research Benchmarks tab in Gradio uses a generated index. From the release checkout, verify it against the published evidence and replay authorship metrics from the saved score table.

```bash
python scripts/build_research_benchmark_summary.py --check
python -m scripts.run_authorship_benchmark validate benchmarks/authorship-v1/results.json
```

To rebuild the index after intentionally refreshing evidence, rerun the summary command without `--check`. The [offline example](../../examples/evaluation/research_benchmarks_demo.py) generates small source-free clone recall and authorship reports from synthetic fixtures.
