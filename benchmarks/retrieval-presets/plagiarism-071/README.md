# Retrieval presets on the existing plagiarism views

This separate diagnostic compares seven existing configurations on all eleven
prepared retrieval views. The plan fixes N=1/5/10, k=10 and a 45-second
configuration/view deadline before scoring. It preserves every query, corpus
document and qrel from those views, including the existing self-exclusion rules.
It does not change the frozen plagiarism-v1 artifacts or any labels.

Run from a checkout containing the plan's exact source commit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validate_retrieval_presets.py \
  --manifest prepared/prepared-manifest.json \
  --plan benchmarks/retrieval-presets/plagiarism-071/plan.json \
  --source-root ../matheel-0.7.1 --release-wheel dist/matheel-0.7.1-py3-none-any.whl \
  --cpu-lock /private/tmp/matheel-071-benchmark-cpu.lock \
  --output-dir benchmark_outputs/retrieval-071
```

Use `git worktree add --detach ../matheel-0.7.1 v0.7.1` for the exact release
checkout and `python -m pip download --no-deps matheel==0.7.1 --dest dist` for the
published wheel. The runner recursively verifies that all wheel Python modules
match the execution checkout. Use the existing versioned preparation recipe to obtain the local normalized
inputs; this runner downloads nothing. Public artifacts contain source IDs,
hashes, configurations and numerical evidence. `provenance.json` binds the runner,
plan, package Python files, manifest, normalized tables and source inputs.
`results.json` preserves complete/timeout/failed rows, while `summary.csv` is the
compact per-view comparison. Each completed configuration has its selected-score
CSV and metrics; two-stage results include per-query candidate recall. The
exhaustive control also includes source-free error analysis and 100-round grouped
bootstrap intervals. Original source code and local source locators are omitted.

`scoring_seconds` covers scoring plus metrics; `wall_seconds` additionally includes
process startup, validation and reporting. Every timed cell is a single
observation on the recorded machine. Timeouts have null quality metrics. There is
no partial-scope aggregate ranking or quality-dependent method selection.

See [practical preset guidance](../../../docs/scalable_retrieval.md) for the
candidate-recall ceiling, fixed top-ten denominator and judged-pool limitations.

## Recorded 0.7.1 observations

The exact released source is `64ca1364a08f76904559e19504524f80671b0479`.
All 46 Python modules matched the published wheel
`b53e215f5f95aa7198eec229a3c30127ec4f38d1a1b7ca77460b7e705ca3473f`.
The fixed plan was committed in `92cb4a4`; final replay/coordination code was
committed in `aab90b7` before these measurements. The correct exclusive-lock
interval was 2026-10-20 21:59:53–22:07:10 UTC.

All **77 declared cells** were retained: **76 complete, one timeout, zero failed**.
The timeout was Winnowing top 5 → Partial Ratio on SOCO14 Java A1 at the fixed
45-second deadline. It retains null quality metrics and no partial score CSV.
It is not relabeled N/A. The complete score tables were independently replayed
against all original qrels/queries; all five ranking metrics matched within
1e-12 tolerance.

The following ranges describe individual completed views; they are not pooled
performance estimates or aggregate ranks. The Partial Ratio row has incomplete
scope and must not be ranked against the eleven-view rows.

| Configuration | Complete views | Macro candidate recall range | MAP range | Maximum scoring seconds among completions |
| --- | --- | --- | --- | --- |
| Exhaustive Winnowing | 11/11 | Full corpus | 0.9530–1.0000 | 0.489 |
| Winnowing top 1 → Levenshtein | 11/11 | 0.8192–1.0000 | 0.8192–1.0000 | 2.120 |
| Winnowing top 5 → Levenshtein | 11/11 | 0.9921–1.0000 | 0.9300–1.0000 | 2.184 |
| Winnowing top 10 → Levenshtein | 11/11 | 0.9921–1.0000 | 0.9248–1.0000 | 2.218 |
| LCS top 5 → Levenshtein | 11/11 | 0.9379–1.0000 | 0.9081–1.0000 | 2.103 |
| Winnowing top 5 → GST | 11/11 | 0.9921–1.0000 | 0.9583–1.0000 | 33.054 |
| Winnowing top 5 → Partial Ratio | 10/11 | 0.9921–1.0000 | 0.9327–1.0000 | 43.628 |

A larger shortlist does not guarantee all judged positives or better final order.
Winnowing N=5 and N=10 both retained macro candidate recall 0.992063 on C++ A1;
all positives were retained in the other ten views. On IRPlag, Levenshtein's MAP
fell from 0.929953 at N=5 to 0.924842 at N=10 despite candidate recall remaining
1.0. Keep the exhaustive cheap scorer as a control and inspect the reranker,
rather than selecting by candidate recall alone. GST completed all eleven
N=5 views within the declared budget; Partial Ratio did not.

| View | Exhaustive Winnowing MAP | Winnowing N=10 candidate recall | N=10 → Levenshtein MAP | Winnowing N=5 → GST MAP |
| --- | --- | --- | --- | --- |
| irplag_retrieval | 0.9986 | 1.0000 | 0.9248 | 0.9888 |
| soco14_cpp_a1_judged_retrieval | 0.9530 | 0.9921 | 0.9476 | 0.9583 |
| soco14_cpp_a2_judged_retrieval | 0.9601 | 1.0000 | 0.9503 | 0.9671 |
| soco14_cpp_b1_judged_retrieval | 0.9819 | 1.0000 | 0.9556 | 0.9848 |
| soco14_cpp_b2_judged_retrieval | 1.0000 | 1.0000 | 0.9883 | 1.0000 |
| soco14_cpp_c1_judged_retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soco14_java_a1_judged_retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soco14_java_a2_judged_retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soco14_java_b1_judged_retrieval | 0.9960 | 1.0000 | 0.9792 | 0.9960 |
| soco14_java_b2_judged_retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| soco14_java_c2_judged_retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

IRPlag and SOCO14 C++ C1 each have only seven eligible candidates per query,
so N=10 returns seven results there. All reported precision@10 denominators
remain ten. No configuration, scope, budget or shortlist size was changed in
response to the observed quality values.

## Download and verify the complete evidence

The PR keeps compact JSON, query diagnostics, summary CSV and the full generated
file inventory. The [complete source-free evidence ZIP](https://github.com/FahadEbrahim/matheel/releases/download/v0.7.1/matheel-0.7.1-retrieval-presets-evidence.zip)
contains all 168 inventoried files plus the inventory itself, including every
completed score CSV. Archive size/hash are recorded in `evidence-archive.json`.

After downloading that archive beside this directory, verify it before inspection:

```python
import hashlib
import json
from pathlib import Path
import zipfile

root = Path("benchmarks/retrieval-presets/plagiarism-071")
metadata = json.loads((root / "evidence-archive.json").read_text())
archive = Path(metadata["filename"])
assert archive.stat().st_size == metadata["bytes"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
inventory_bytes = (root / "artifact-manifest.json").read_bytes()
inventory = json.loads(inventory_bytes)["files"]
with zipfile.ZipFile(archive) as bundle:
    assert bundle.read("artifact-manifest.json") == inventory_bytes
    assert set(bundle.namelist()) == set(inventory) | {"artifact-manifest.json"}
    for name, expected in inventory.items():
        content = bundle.read(name)
        assert len(content) == expected["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
```

The archive SHA-256 is
`61426f6583016763f125285f91d82ef37773da07381014aad25dd6e58ebc8300`.
The original frozen v1 artifacts remain unchanged.
