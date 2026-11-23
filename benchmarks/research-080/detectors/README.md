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
  --irplag-comparison benchmarks/external-detectors/irplag-java-20261020/scores
```

- Native Java clone findings use [NiCad 7.0.1](https://github.com/CordyJ/Open-NiCad/releases/tag/v7.0.1) with [OpenTxl 11.3.7](https://github.com/CordyJ/OpenTxl/releases/tag/v11.3.7) and [PMD CPD 7.29.0](https://github.com/pmd/pmd/releases/tag/pmd_releases%2F7.29.0) with Java 21. The synthetic two-file examples measure installation and report handling. The real source-span slice uses a fixed database order and the available IJaDataset source paths. Native XML can contain execution-dependent paths and timestamps. The record hashes the normalized findings.
- The pinned [BigCloneEval v1.1](https://github.com/jeffsvajlenko/BigCloneEval/tree/v1.1) fixture exercises BOTH syntax bins, original and pretty line minima of six, a token minimum of 50, and inclusive 70% fractional coverage of both references. Matheel matched four of seven eligible references. The official Java/H2 runtime matched one of seven on those same findings, rejecting the tested 70% and 80% partial overlaps. At Matheel's 100% coverage setting, matched and eligible counts agreed with the official replay in 24 strata. The fixture also exercises reversed spans, excluded internal rows and below-minimum sizes.
- BigCloneBench v2 supplies the typed reference judgments under CC BY-NC 4.0. IJaDataset 2.0 files retain their original licensing. The replay used the pinned CloReCo replication image data layers, which preserve the BigCloneEval H2 database and reduced IJaDataset files extracted by the published build recipe. Supply the local `bcb.h2.db` and the `bcb_reduced` directory containing each functionality's `default`, `sample` and `selected` subdirectories. H2 recovers the archived database using a private temporary copy, which needs around 6 GB of free temporary disk space. The evidence publishes counts and digests only. [BigCloneBench license](https://github.com/clonebench/BigCloneBench#license), [CloReCo recipe](https://github.com/Glopix/Cloreco/blob/37a516e507ab3edb6c339df5e5a4cc113547b75c/images/_big-clone-eval-base/Dockerfile), [CloReCo replication study](https://doi.org/10.5220/0013644900003964).
- Saved MOSS cohort results come from `metadata.csv`, `moss_raw.csv`, `moss_java.html` and `moss_cs.html` in [SimilCode v2.4.0](https://github.com/gleiston-guerrero/SimilCode-Benchmark/tree/v2.4.0). Both directional scores reproduce the author's means for 65 archived matches. Another 55 expected case pairs are absent from the archived HTML and remain unscored. The two reports also include matches outside those case pairs.
- [Copydetect 0.5.0](https://pypi.org/project/copydetect/0.5.0/) ran on 467 files and 460 labeled [IRPlag Java pairs](https://github.com/oscarkarnalim/sourcecodeplagiarismdataset/tree/f07e95120680b25f78659ed63cd6130fd05b2df8). Extract the upstream `IR-Plag-Dataset` files so Matheel can locate the `case-*` directories. Matheel Winnowing ran on the same labels. The comparison replays [archived JPlag and Dolos scores](../../external-detectors/irplag-java-20261020/README.md) after checking that all labeled pair IDs agree. The results record the Python runtime, Copydetect dependency versions, input fingerprint and score-file digests. The threshold is fixed at 0.5.
- [SourcererCC](https://github.com/Mondego/SourcererCC/tree/88314f51fdf3cd89103eb2c19fb56464308846c8) is verified on a generated native-format report and explicit block-to-source mapping. The source-map join preserves source spans and deduplicates reversed pairs.
