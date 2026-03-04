# Chunking

Chunking splits a file into smaller pieces before embedding. This is useful when files are long, when you want late interaction over chunks, or when you want chunk-aware aggregation.

## Parameters

- `chunking_method`
- `chunk_size`
- `chunk_overlap`
- `max_chunks`
- `chunk_language`
- `chunker_options`
- `chunk_aggregation`

## Public Methods

- `none`
  No chunking. The whole file is treated as one document.
- `code`
  Code-aware chunking through Chonkie’s code chunker.
- `chonkie_token`
  Token-based chunking.
- `chonkie_sentence`
  Sentence/statement-oriented chunking.
- `chonkie_recursive`
  Recursive chunking for progressively splitting larger text.
- `chonkie_fast`
  A lightweight Chonkie path for simple chunking workflows.

## Parameter Details

- `chunk_size`
  Size hint passed to chunkers that support a target chunk length.
- `chunk_overlap`
  Overlap hint between adjacent chunks.
- `max_chunks`
  Maximum number of chunks kept per file. `0` means keep all chunks.
- `chunk_language`
  Language hint used by language-aware chunkers, especially code chunking.
- `chunker_options`
  Extra `name=value` options forwarded to the underlying chunker.
- `chunk_aggregation`
  How chunk embeddings are reduced for single-vector scoring.
  Common choices: `mean`, `max`.

## Install

```bash
pip install "matheel[chunking]"
pip install "matheel[chunking_code]"
```

## Behavior Notes

- Chunking is language-agnostic at the interface level.
- Code-aware chunkers become stronger when `chunk_language` matches the source language.
- If a Chonkie-backed method is selected and Chonkie is not installed, Matheel raises an import error instead of silently switching methods.

## CLI Example

```bash
matheel compare sample_pairs.zip \
  --chunking-method code \
  --chunk-language java \
  --chunk-size 120 \
  --chunk-overlap 20 \
  --max-chunks 8 \
  --chunker-option include_line_numbers=true
```

## Python Example

```python
from matheel.chunking import chunk_text

chunks = chunk_text(
    "public class Demo { int add(int a, int b) { return a + b; } }",
    method="code",
    chunk_size=80,
    chunk_overlap=10,
    max_chunks=4,
    chunk_language="java",
    chunker_options={"include_line_numbers": True},
)
print(chunks)
```
