# Controlled robustness diagnostics

Robustness experiments measure how existing scorers respond to a declared change in
their inputs. They produce baseline scores, transformed scores and score deltas,
with the input text, source hashes, transformation parameters and validation evidence.
These are diagnostic comparisons; they do not infer plagiarism or change dataset labels.

## Run the offline example

From an installed checkout with the optional chunking dependency:

```bash
python -m pip install -e ".[chunking]"
python examples/experiments/robustness_and_long_files.py \
  --seed 17 --output-dir benchmark_outputs/experiments
```

The example uses Levenshtein, Winnowing and the existing static hash vector backend.
It downloads no datasets or models. Its second workflow compares
[long-file chunking and aggregation](long_files.md).

The CLI can run the robustness portion using only the base installation:

```bash
matheel experiments robustness --seed 17 --output-dir benchmark_outputs/robustness
```

## What the controlled cases mean

| Configuration | Baseline and change | Validation recorded |
| --- | --- | --- |
| Identity | Compare the same source twice | Fixed behavior assertions |
| Formatting | Rewrite the candidate with Python's `ast.unparse` | Equal Python ASTs and fixture behavior assertions |
| Comments | Add deterministic comments to the candidate | Equal Python ASTs and fixture behavior assertions |
| Identifier renaming | Rename selected local identifiers in a known fixture | Explicit assertions for empty, mixed-sign and repeated-value inputs |
| Partial copy | Append one reference function to a different controlled program | Existing and copied functions satisfy named behavior fixtures; copied byte count/fraction recorded |
| Shared starter code | Prepend the same provided helper to two otherwise distinct programs | Original program and provided-helper behavior fixtures |

Partial copying changes the program's contents. Shared starter code supplies legitimate
common material. Neither configuration is assigned a positive plagiarism label. Their
score deltas are measured relative to the original, explicitly recorded pair.

AST equality checks syntax structure. It does not prove arbitrary observational
equivalence: introspection, comments used by external tools and source line numbers can
matter. Renaming is binding-sensitive, so the generic API requires a caller validator
and a named assertion description. It does not assume that identifier substitution
preserves behavior.

## Use your own source and scoring runs

The CLI generates only formatting and comment cases for supplied Python source. It
parses that source and scores its text; it never executes it:

```bash
matheel experiments robustness --source solution.py \
  --config scoring_runs.json --output-dir benchmark_outputs/source_diagnostics
```

`scoring_runs.json` uses the same declared options as a comparison suite:

```json
{
  "runs": [
    {"run_name": "raw_levenshtein", "feature_weights": {"levenshtein": 1}},
    {
      "run_name": "basic_levenshtein",
      "options": {
        "feature_weights": {"levenshtein": 1},
        "preprocess_mode": "basic",
        "code_language": "python"
      }
    }
  ]
}
```

For binding-sensitive transformations, use explicit assertions through Python:

```python
from matheel.robustness import python_robustness_case, run_robustness_experiment

source = "def add(a, b):\n    total = a + b\n    return total\n"

def validate_known_fixture(before, after):
    # Execute only this known, trusted fixture, never an untrusted submission.
    for text in (before, after):
        namespace = {}
        exec(text, namespace)
        if namespace["add"](2, 3) != 5 or namespace["add"](-2, 2) != 0:
            return False
    return True

case = python_robustness_case(
    source, "identifier_renaming", identifier_map={"total": "result"},
    validator=validate_known_fixture, validation_name="add(2,3)==5; add(-2,2)==0",
)
report, artifacts = run_robustness_experiment(
    [{"run_name": "levenshtein", "feature_weights": {"levenshtein": 1}}],
    cases=[case], output_dir="benchmark_outputs/rename",
)
```

## Read and reproduce the results

`robustness.json` contains complete case inputs and provenance, declared scorer
configurations, deterministic case/run IDs, implementation hashes and the environment
snapshot. `robustness_scores.csv` contains the numeric comparisons. A failed run retains
null scores and its error; the CLI exits nonzero so it cannot look like a successful
zero-score comparison.

Compare like configurations and inspect individual transformations. These fixtures
are small diagnostics, not evidence of population-level detection quality. For model
runs, use a pinned local model snapshot as `model_name` when exact replay matters.
The public frozen plagiarism-v1 protocol and its 256-token configuration remain
separate from these experiments.

## Practical configuration comparisons

The [0.7.1 configuration guidance experiment](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/configuration-guidance/plagiarism-071)
compares existing raw/basic Levenshtein, raw LCS and raw Winnowing settings.
It uses all 293 existing Python pair observations for fixed-threshold error
analysis, the six controlled fixtures above, and formatting/comments applied to
one declared, size-selected Python submission. Supplied code is parsed and scored
without executing it. The selection uses byte counts and source row order, without
looking at labels, scores or measured coverage. Its source-free outputs retain
input hashes, scorer options, transformation provenance, paired uncertainty,
false positives/negatives and length slices.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/compare_existing_configurations.py \
    --manifest prepared/prepared-manifest.json \
    --plan benchmarks/configuration-guidance/plagiarism-071/plan.json \
    --source-root ../matheel-0.7.1 --release-wheel dist/matheel-0.7.1-py3-none-any.whl \
    --cpu-lock /private/tmp/matheel-071-benchmark-cpu.lock \
    --output-dir benchmark_outputs/configuration-guidance-071
```

Use the exact `v0.7.1` source checkout and published wheel described in the
[retrieval replay instructions](scalable_retrieval.md#compare-presets-across-the-existing-judged-views).
The POSIX runner verifies that source against the wheel and gives each of its three
workflows 120 seconds. The long-file workflow needs the pinned MiniLM snapshot
already cached, plus the existing `sentence_transformers` and `chunking` extras.
It downloads nothing. Scorer failures retain null scores and a failed workflow;
they cannot count as successful zero scores.

Use raw/basic comparisons to determine which formatting and comments your
experiment intends to retain. Removing those signals may reduce sensitivity to
surface edits, while also increasing scores for unrelated programs with common
syntax or starter material. Inspect both false-positive and false-negative rows
and paired differences on the same observations. The example threshold of 0.5 is
fixed in advance for diagnosis; it is not fitted to these inputs or recommended
as a deployment decision boundary. Confidence intervals are conditional on the
existing source-sharing groups and labels.

Use the partial-copy and shared-starter cases to expose score behavior, with the
recorded transformation as the explanation. A higher score after adding common
material does not establish plagiarism. Neither these six fixtures nor one
transformed submission establish robustness across datasets, languages or
adversarial transformations.
