# Long-file input coverage and aggregation experiments

A long file may exceed an encoder's input limit, or lose its tail through `max_chunks`
or `first` aggregation. Preflight reports what the configured input pipeline can see;
the comparison workflow then scores the same pair using declared existing strategies.

## Start offline

```bash
python -m pip install -e ".[chunking]"
python examples/experiments/robustness_and_long_files.py \
  --output-dir benchmark_outputs/experiments
matheel experiments long-files \
  benchmark_outputs/experiments/left.py benchmark_outputs/experiments/right.py \
  --config benchmark_outputs/experiments/long_file_runs.json \
  --output-dir benchmark_outputs/long_files
```

The default runs compare the existing `static_hash` backend with no chunking, then
character-tokenizer Chonkie chunks aggregated by `mean`, elementwise `max`, or `first`.
A fifth run caps mean aggregation at two chunks. These use no trained-model downloads.
`static_hash` has no transformer input limit and ignores `max_token_length`.

These are existing embedding-aggregation choices. `max` takes the maximum of each
vector coordinate; it does not select the highest-scoring pair of source chunks.
The multi-vector backend has its own late-interaction behavior and ignores the
single-vector `chunk_aggregation` control.

## Preflight without loading a model

```bash
matheel experiments preflight solution.py --output benchmark_outputs/preflight.json
```

For another declared pipeline, pass a JSON object of scorer options:

```json
{
  "vector_backend": "sentence_transformers",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "max_token_length": 256,
  "chunking_method": "chonkie_token",
  "chunk_size": 128,
  "chunk_overlap": 16,
  "max_chunks": 4,
  "chunker_options": {"tokenizer": "character"},
  "chunk_aggregation": "mean"
}
```

```bash
matheel experiments preflight solution.py --options-file input_options.json \
  --output benchmark_outputs/preflight.json
```

Without `--inspect-model`, preflight does not load models or contact a model hub.
It reports actual UTF-8 bytes, lines, chunk selection and source alignment. Transformer
token counts and truncation coverage remain unknown. A byte or line count is not a
substitute for the model's tokenizer.

## Measure the real tokenizer's truncation

`--inspect-model` explicitly loads the configured model and tokenizer. It may download
weights unless they are already cached; use `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` to require cached weights.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  matheel experiments preflight solution.py --options-file input_options.json \
  --inspect-model --output benchmark_outputs/tokenizer_preflight.json
```

The Python API also accepts an already loaded model or tokenizer:

```python
from matheel.long_files import preflight_long_file

report = preflight_long_file(
    source_text, options=input_options, model=loaded_model,
)
```

With compatible fast-tokenizer offsets, preflight applies the tokenizer's actual
truncation operation. This accounts for special tokens, prompt boundaries and left
or right truncation. The effective input limit is the minimum of the requested cap
and the detected model/tokenizer caps. Query and document limits can be inspected
separately with `is_query=True` or `False`.

The model's default prompt is recorded when available. For another wrapper-specific
query/document prefix, pass the exact `prefix` through the Python API. Preflight cannot
infer arbitrary encoder prompts or additional text normalization. A tokenizer that
supports counting but lacks offsets records count-only evidence; retained source
tokens and bytes are unknown when that input exceeds the cap.
An already truncating Rust tokenizer may not expose an uncapped counting operation;
preflight records that limitation as unknown coverage without mutating it or preventing
the configured scorer from running.

## Interpret coverage

The JSON records original and prepared source hashes/byte/line counts, generated and
selected chunks, omitted chunks, effective chunks, per-chunk input/retained token counts,
and truncation evidence.

`selected_chunk_coverage` measures the union of chunk-aligned prepared-source bytes;
overlap is not double-counted. Chunk-to-source substring alignment is marked as an
estimate when source is repeated or rewritten. Unmatched generated chunk text yields
unknown coverage. Whitespace trimmed by the existing input preparation is recorded in
the separate original/prepared counts.

`effective_model_input_coverage` additionally accounts for truncation and `first`
aggregation. For transformer offsets it counts source bytes spanned by retained tokens,
so whitespace without token offsets is excluded. It measures visible input, not which
tokens influenced a final vector or whether a comparison is semantically correct.

## Compare declared configurations

`matheel experiments long-files` accepts a normal comparison-suite `runs` JSON file.
It records every configuration, full source text/hashes, implementation hashes,
preflight evidence and score in `long_files.json`, plus `long_files_scores.csv`.
Semantic runs inspect their configured model before scoring by default;
`--no-inspect-models` retains byte/line diagnostics without the extra inspection load.
Failed scorers retain null scores and errors, and make the CLI exit nonzero.

Lexical and code-only scorers use the full prepared source; chunking and vector limits
do not apply to those components. Preflight states this when such configurations are
compared. Changing a chunk size or aggregation defines an exploratory configuration:
it does not modify the frozen plagiarism-v1 256-token protocol or its published scores.

## Choose a configuration using measured coverage

The [0.7.1 configuration guidance experiment](https://github.com/FahadEbrahim/matheel/tree/main/benchmarks/configuration-guidance/plagiarism-071)
uses one real Python pair selected by a predeclared byte-count rule. It compares
the same pinned cached MiniLM model at a 256-token cap with raw input, 512-character
chunks/64-character overlap aggregated by mean, max or first, and mean capped at
two chunks. A full-source Winnowing score supplies a lexical comparison. The pair's
existing label is retained in its dataset; this one-pair experiment does not
estimate classification or retrieval quality.

Read each score next to the input-coverage report. Raw input can omit the tail
when the tokenizer limit is exceeded. Unlimited mean/max aggregation can preserve
more input while changing the representation and score. `first` still encodes
selected chunks but uses only the first chunk in the final vector, so its extra
work does not provide final-vector tail coverage. `max_chunks=2` is an explicit
input budget, with measured omissions rather than a claim of complete coverage.

Character chunk sizes are reproducible input boundaries, not model-token counts.
The real tokenizer's per-chunk retained-token measurements determine whether a
512-character chunk itself truncates. Choose chunking and aggregation before
evaluating the intended workload, check effective coverage, and compare detection
metrics on that workload separately. Full input visibility alone does not make a
semantic representation accurate, and these controlled comparisons do not select
a universal aggregation strategy.
