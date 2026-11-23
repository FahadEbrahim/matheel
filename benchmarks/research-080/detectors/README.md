# Detector evidence for Matheel 0.8.0

[results.json](results.json) records source-free metrics, tool configurations and artifact digests. Local native runs produce sparse clone findings or, for Copydetect, complete pair scores. The evidence records the dataset and scope used for each run.

## Replay

Install Matheel from this checkout and supply the local tool installations and data listed below. Build BigCloneEval at commit `8108126f68f67f4148af91daa0b4126ee4e3cda8` with its upstream Makefile. The script can regenerate individual cases by omitting their options and writing to a separate output file.

```bash
python -m pip install -e ".[benchmark]" copydetect==0.5.0
python scripts/run_research_detectors.py \
  --output detector-results.json \
  --nicad "$NICAD_BIN" --txl "$TXL_BIN" \
  --pmd "$PMD_BIN" --java-home "$JAVA_HOME" \
  --bigcloneeval-root "$BIGCLONEEVAL_ROOT" \
  --bcb-db "$BIGCLONEBENCH_H2_DB" --ija-root "$IJADATASET_BCB_REDUCED" \
  --similcode-dir "$SIMILCODE_REPORTS" --irplag-dir "$IRPLAG_EXTRACTED" \
  --irplag-comparison benchmarks/external-detectors/irplag-java-20261020/scores \
  --sourcerercc-linux benchmarks/research-080/detectors/sourcerercc-linux \
  --sourcerercc-typed benchmarks/research-080/detectors/sourcerercc-typed.json
```

- Native Java clone findings use NiCad 7.0.1 with OpenTxl 11.3.7 and PMD CPD 7.29.0 with Java 21. The synthetic two-file examples measure installation and report handling. The real source-span slice uses a fixed database order and the available IJaDataset source paths. Native XML can contain execution-dependent paths and timestamps. The record hashes the normalized findings. [NiCad 7.0.1](https://github.com/CordyJ/Open-NiCad/releases/tag/v7.0.1), [OpenTxl 11.3.7](https://github.com/CordyJ/OpenTxl/releases/tag/v11.3.7), [PMD CPD 7.29.0](https://github.com/pmd/pmd/releases/tag/pmd_releases%2F7.29.0).
- The pinned BigCloneEval v1.1 fixture exercises BOTH syntax bins, original and pretty line minima of six, a token minimum of 50, and inclusive 70% fractional coverage of both references. Matheel matched four of seven eligible references. The official Java/H2 runtime matched one of seven on those same findings, rejecting the tested 70% and 80% partial overlaps. At Matheel's 100% coverage setting, matched and eligible counts agreed with the official replay in 24 strata. The fixture also exercises reversed spans, excluded internal rows and below-minimum sizes. [BigCloneEval v1.1](https://github.com/jeffsvajlenko/BigCloneEval/tree/v1.1).
- BigCloneBench v2 supplies the typed reference judgments under CC BY-NC 4.0. IJaDataset 2.0 files retain their original licensing. The replay used the pinned CloReCo replication image data layers, which preserve the BigCloneEval H2 database and reduced IJaDataset files extracted by the published build recipe. Supply the local `bcb.h2.db` and the `bcb_reduced` directory containing each functionality's `default`, `sample` and `selected` subdirectories. H2 recovers the archived database using a private temporary copy, which needs around 6 GB of free temporary disk space. The evidence publishes counts and digests only. [BigCloneBench license](https://github.com/clonebench/BigCloneBench#license), [CloReCo recipe](https://github.com/Glopix/Cloreco/blob/37a516e507ab3edb6c339df5e5a4cc113547b75c/images/_big-clone-eval-base/Dockerfile), [CloReCo replication study](https://doi.org/10.5220/0013644900003964).
- Saved MOSS cohort results come from `metadata.csv`, `moss_raw.csv`, `moss_java.html` and `moss_cs.html` in SimilCode v2.4.0. Both directional scores reproduce the author's means for 65 archived matches. Another 55 expected case pairs are absent from the archived HTML and remain unscored. The two reports also include matches outside those case pairs. [SimilCode v2.4.0](https://github.com/gleiston-guerrero/SimilCode-Benchmark/tree/v2.4.0).
- Copydetect 0.5.0 ran on 467 files and 460 labeled IRPlag Java pairs. Extract the upstream `IR-Plag-Dataset` files so Matheel can locate the `case-*` directories. Matheel Winnowing ran on the same labels. The comparison replays archived JPlag and Dolos scores after checking that all labeled pair IDs agree. The results record the Python runtime, Copydetect dependency versions, input fingerprint and score-file digests. The threshold is fixed at 0.5. [Copydetect 0.5.0](https://pypi.org/project/copydetect/0.5.0/), [IRPlag source](https://github.com/oscarkarnalim/sourcecodeplagiarismdataset/tree/f07e95120680b25f78659ed63cd6130fd05b2df8), [archived JPlag and Dolos scores](../../external-detectors/irplag-java-20261020/README.md).
- SourcererCC accepts two-column pairs and four-column project/block findings. The source-map join preserves original spans and deduplicates reversed pairs. The pinned CloReCo Linux image matched the authored Type 1 fixture and passed checks of its original line ranges. The shared 141-file BigCloneBench slice separately used the image's extracted tokenizer and SearchManager JAR. It retained 691 of 694 raw pairs after original-span verification and matched 45 of 137 eligible references. [Linux run](https://github.com/FahadEbrahim/matheel/actions/runs/45279822397) · [Linux scores and hashes](sourcerercc-linux/sourcerercc.json) · [Stratified result](sourcerercc-typed.json)

Run the native fixture on a Docker-enabled `linux/amd64` host.

```bash
python scripts/check_sourcerercc_artifact.py --output ./runs/sourcerercc-linux
```

## Fetch the pinned BigCloneBench and IJaDataset data layers

Run this from an empty local directory with `curl`, `jq`, `shasum` and `tar`. The public registry issues an anonymous read token. The commands verify the image manifest and both compressed layer digests, then extract only the H2 file and the reduced IJaDataset directory. They leave the artifacts in `bcb-data/` below the working directory.

```bash
set -eu
registry_repo=glopix/cloreco-images/big-clone-eval-base
manifest=sha256:34171964a86f33798888b40ab3cfa38d1cf6329a34f20b6a1eae1dccb051ed1d
db=sha256:2350bf0d5f74ceb96bf0a2b2533602946a18ec989b412d505eb59b62b959e237
sources=sha256:03aaf2c598806abc5d889c1c5d756cc721da9406a778dfeb1ca4b6350145e547
token=$(curl -fsS "https://ghcr.io/token?scope=repository:${registry_repo}:pull" | jq -er '.token')
endpoint="https://ghcr.io/v2/$registry_repo"
curl -fsSL -H "Authorization: Bearer $token" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "$endpoint/manifests/$manifest" -o manifest.json
printf '%s  %s\n' "${manifest#sha256:}" manifest.json | shasum -a 256 -c -
jq -e --arg db "$db" --arg sources "$sources" \
  '([.layers[].digest] | index($db) != null and index($sources) != null)' manifest.json
for layer in "$db" "$sources"; do
  archive="${layer#sha256:}.tar.gz"
  curl -fsSL --retry 4 -H "Authorization: Bearer $token" \
    "$endpoint/blobs/$layer" -o "$archive"
  printf '%s  %s\n' "${layer#sha256:}" "$archive" | shasum -a 256 -c -
done
mkdir bcb-data
member=cloneDetection/BigCloneEval
tar -xzf "${db#sha256:}.tar.gz" -C bcb-data "$member/bigclonebenchdb/bcb.h2.db"
tar -xzf "${sources#sha256:}.tar.gz" -C bcb-data "$member/ijadataset/bcb_reduced"
export BIGCLONEBENCH_H2_DB="$PWD/bcb-data/$member/bigclonebenchdb/bcb.h2.db"
export IJADATASET_BCB_REDUCED="$PWD/bcb-data/$member/ijadataset/bcb_reduced"
```

The prefix cohort takes the first 100 non-internal references. The stratified cohort takes the first 20 non-internal references independently in each of the seven disjoint BOTH strata. Both selections use functionality ID, then left and right function ID for ordering. Each detector sees the files for the selected reference pairs. Recall applies the recorded size filters and reports source coverage separately. Use `--bcb-slice prefix` or `--bcb-slice stratified` to run one cohort.

The stratified cohort covers 141 source files and has both source files for all 140 selected references. The size filters leave 137 references. NiCad matched 85, PMD CPD matched 40, and SourcererCC matched 45. Each matched all 20 Type 1 references. The 100-reference prefix covers only weak Type 3/Type 4, and both NiCad and CPD matched zero references there.

For the typed SourcererCC run, install the Java grammar, export the stratified references, and use the same pinned Docker image on a `linux/amd64` host. Matheel checks the selected corpus and reference hashes before starting the detector.

```bash
matheel grammars java
matheel extract-bigcloneeval-references "$BIGCLONEBENCH_H2_DB" ./runs/reference.csv \
  --java "$JAVA_HOME/bin/java" --h2-jar "$BIGCLONEEVAL_ROOT/libs/h2-1.3.176.jar" \
  --max-references-per-type 20
python -m scripts.run_sourcerercc_bigclonebench \
  --ija-root "$IJADATASET_BCB_REDUCED" --references ./runs/reference.csv \
  --output ./runs/sourcerercc-typed.json
```

The published stratified run used the image's tokenizer and JAR on macOS. The result records the runtime and grammar hashes. The detector used a 0.8 threshold and a 65-token minimum. Its tokenizer emitted 1,007 candidate blocks from 139 files. Matheel used Java parse boundaries and the native block hashes to verify 1,004 original method ranges, correcting 12 end coordinates affected by blank lines. Three predicted pairs had unverified endpoints and were excluded. Use `--recorded-native` to score a local saved stats and pair-report directory again.
