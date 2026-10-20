# Plagiarism benchmark results

`Matheel-Plagiarism-v1` compares declared source-code similarity methods on a frozen
plagiarism benchmark. The definition includes **29 method configurations**, **five
pair-classification datasets**, and **11 retrieval task views**: IRPlag and ten
SOCO14 language/scenario views. The pair datasets contain **2,284 labeled rows**;
the retrieval views contain **1,180 queries**.

The initial supervised evaluation is in progress. Rankings will be published only
after every declared method has a completed or explicitly failed checkpoint and
the result bundle passes protocol, coverage, score-file, and portability checks.
The ready-made sampled leaderboard remains labeled **Demo** while this versioned
run is pending.

## What this benchmark measures

Pair average precision and AUROC measure discrimination against the frozen labels.
F1, precision, recall, and accuracy use the fixed threshold of `0.5`. Retrieval MAP
and MRR use complete rankings; nDCG, precision, and recall use `k = 10`.

Each task view has equal aggregate weight within its evaluation task. The ten
SOCO14 views contribute ten elevenths of a retrieval average; IRPlag contributes
one eleventh. Inspect individual views alongside aggregate scores. SOCO14 uses
valid judged endpoints as its candidate pools, with separate languages/scenarios
and self matches excluded. These pools differ from the original competition's
full candidate corpus. All policies and omissions are recorded in the
[benchmark definition and source recipe](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/plagiarism-v1).

A method needs finite scores on every declared task view for an aggregate ranking
to be eligible. Unfinished or unsupported evaluations retain null scores and
explicit errors. Each method has a supervised **1,800-second** wall-clock budget;
timeouts remain visible and unranked. Resource limits describe feasibility under
the recorded conditions.

## Reproduce and inspect

- [Frozen benchmark protocol](benchmark_protocol.md): identities, model pins,
  evaluation rules, result validation, and review procedure.
- [Preparation and execution recipe](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/plagiarism-v1):
  pinned archives, complete task definitions, method configurations, and supervisor commands.
- [Leaderboard guide](leaderboard.md): custom experiments and artifact inspection.

The canonical public artifacts will be saved once in
`gradio_app/benchmark_data/plagiarism-v1/`, which is also available to the hosted
Gradio app. Large audited score CSVs are packaged outside Git for attachment to
the GitHub release. Source submissions, local protocols, model caches, worker
checkpoints, and logs remain outside the published result bundle.
