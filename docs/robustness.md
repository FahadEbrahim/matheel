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
