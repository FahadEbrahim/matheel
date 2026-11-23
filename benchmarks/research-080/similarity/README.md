# Matheel 0.8.0 similarity evidence

These runs report the released pair and retrieval populations. Each report records the input hashes, normalized dataset hash, method settings, runtime versions, and output hashes. Full score files can be regenerated locally. The published bundle includes the compact SimilCode scores and category summaries. It records hashes and counts for the larger score files.

| Scope | Evaluated population | Primary metric |
| --- | ---: | --- |
| [SimilCode](similcode/results.json) | 120 Java and C# pairs | Binary F1 for the documented relatedness view, plus [category and clone-type score slices](similcode/slices.csv) |
| [SimilCode Java native detectors](similcode-java-native/results.json) | 60 Java pairs | Binary F1 and [score slices](similcode-java-native/slices.csv) for local JPlag and Dolos |
| [SimilCode C# native detector](similcode-csharp-native/results.json) | 60 C# pairs | Binary F1 and [score slices](similcode-csharp-native/slices.csv) for local JPlag |
| [POJ-104](poj104-test/results.json) | 12,000 test queries, 143,988,000 eligible comparisons | MAP@R with 499 same-problem relevant candidates per query |
| [XLCoST Java → Python](xlcost-java-python/results.json) | 882 test queries, 882 target programs | First-hit MRR and Recall@10 for the fixed target-language pool |
| [XLCoST Python → Java](xlcost-python-java/results.json) | 882 test queries, 882 target programs | First-hit MRR and Recall@10 for the fixed target-language pool |
| [CodeXGLUE BigCloneBench](codexglue-bcb-test/results.json) | 415,416 official test pairs | Binary F1 |

The two CodeXGLUE runs passed [reference evaluator parity](reference-checks.json) over the full test scopes. The XLCoST directions are Matheel tasks with target-language pools sourced from the official archive. The upstream mixed-target metric also uses a different reciprocal-rank formula. In SimilCode, the relatedness view maps identical, functional, and structural pairs to the positive label. Its Type 4 cases include independent implementations. Native detector runs keep language-specific corpora and publish counts for every category, including the empty Java Type 2 slice. [XLCoST evaluator](https://github.com/reddy-lab-code-research/XLCoST/blob/main/code/codesearch/code2codesearch/evaluator/evaluator.py) · [SimilCode labels](https://github.com/gleiston-guerrero/SimilCode-Benchmark/tree/v2.4.0#1-corpus-and-ground-truth)

## Inputs

Acquire the datasets according to the source terms. Benchmark directories containing `dataset/` include local copies of the source code. Keep those directories out of published results.

| Input | Pinned source |
| --- | --- |
| SimilCode | [CC BY 4.0 version 2.4.0](https://doi.org/10.5281/zenodo.23949390). The archived Git tag downloaded for the run had SHA-256 `481c9ea731d7b85c0e03764965482b309a09bd44a925d21aeed3cb11d0566855`. Extract it into `./data/similcode`. |
| POJ-104 | [Test Parquet from the pinned mirror](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_poj104/blob/701869dece36b32174ee48277eada423cdcf699c/data/test-00000-of-00001.parquet), SHA-256 `4c1d6389d66a68c2b5ad4c345ac264b8f4062258aa870b74a3fd478b7cce4955`. Save it as `./data/poj104/test.parquet` to reproduce the report metadata. The adapter also accepts original preprocessed JSONL. |
| XLCoST | [Official XLCoST_data.zip](https://drive.google.com/file/d/1Cp3vFITRaUEJwPoeI_uv0cC6KVyvDc4F/view?usp=sharing), SHA-256 `2b822d82197c4ca159a383854f8a3d187f24f970e6209e77641e709220bf2f2b`. Extract the `retrieval/code2code_search/program_level/{Java,Python}` folders into their original relative locations under `./data/XLCoST_data`. |
| BigCloneBench binary pairs | [CodeXGLUE data directory at the reference revision](https://github.com/microsoft/CodeXGLUE/tree/ac74a62802a0dd159b3258c78a2df8ad36cdf2b9/Code-Code/Clone-detection-BigCloneBench/dataset). Save `data.jsonl`, `train.txt`, `valid.txt`, and `test.txt` in `./data/codexglue-bcb`. See the run report for their SHA-256 values. |

## Reproduce

The reference runs used Python 3.14.2. Install Matheel 0.8.0 with its benchmark extra and the recorded constraints to reproduce them. The CPU retrieval baseline fits character TF-IDF on the fixed unlabeled target corpus, scans every eligible candidate, and breaks exact score ties by ascending document ID. The POJ-104 score archive has 5,988,000 rows and occupies about 85 MB when compressed. The default query batch contains 32 rows. Per-query artifacts preserve first-hit ranks for results beyond the saved score cutoff.

```bash
python -m pip install -c benchmarks/research-080/similarity/reference-constraints.txt 'matheel[benchmark]==0.8.0'

matheel benchmark-similarity similcode_pair ./data/similcode ./runs/similcode \
  --source-revision v2.4.0

matheel benchmark-similarity poj104_retrieval ./data/poj104 ./runs/poj104 \
  --source-revision 701869dece36b32174ee48277eada423cdcf699c

XL_REV='zip-sha256:2b822d82197c4ca159a383854f8a3d187f24f970e6209e77641e709220bf2f2b'
matheel benchmark-similarity xlcost_retrieval ./data/XLCoST_data ./runs/xlcost-java-python \
  --source-revision "$XL_REV" --source-language java --target-language python
matheel benchmark-similarity xlcost_retrieval ./data/XLCoST_data ./runs/xlcost-python-java \
  --source-revision "$XL_REV" --source-language python --target-language java

matheel benchmark-similarity codexglue_bcb_pair ./data/codexglue-bcb ./runs/codexglue-bcb \
  --source-revision ac74a62802a0dd159b3258c78a2df8ad36cdf2b9
```

From the Matheel checkout, compare the exact saved predictions against a separate local Microsoft CodeXGLUE checkout at the pinned revision. The parity script verifies the upstream evaluator hashes before loading them.

```bash
python scripts/check_similarity_reference_parity.py ./data/CodeXGLUE \
  --poj-run ./runs/poj104 --bcb-run ./runs/codexglue-bcb \
  --output ./runs/reference-checks.json
```

For the local SimilCode detector runs, use the pinned Java 25, JPlag 6.3.0, Node 22, and Dolos 2.10.0 artifacts described in [External detectors](../../../docs/external_detectors.md). The run report records the actual native file and runtime hashes. Run from the checkout with Matheel installed.

```bash
python scripts/run_similcode_detectors.py ./data/similcode ./runs/similcode-java-native \
  --language java --jplag-jar "$MATHEEL_JPLAG_JAR" --java-path "$MATHEEL_JAVA" \
  --dolos-prefix "$MATHEEL_DOLOS_PREFIX" --node-path "$MATHEEL_NODE"
python scripts/run_similcode_detectors.py ./data/similcode ./runs/similcode-csharp-native \
  --language csharp --jplag-jar "$MATHEEL_JPLAG_JAR" --java-path "$MATHEEL_JAVA"
```
