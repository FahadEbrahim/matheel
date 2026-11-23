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
  Code-aware chunking from a local Tree-sitter grammar with a Chonkie tokenizer.
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

## Set up grammars for offline use

Run `matheel grammars` while connected to provision all 20 languages supported by Matheel's code metrics. For a smaller cache, name the languages you need.

```bash
export TREE_SITTER_LANGUAGE_PACK_CACHE_DIR="$PWD/grammar-cache"
matheel grammars python java
```

Provisioning fetches the upstream platform archive and extracts the selected grammars. The command prints the directory with the extracted libraries. Keep the same cache setting when running offline. To move the cache to another machine, copy the versioned `tree-sitter-language-pack/v<VERSION>/libs` directory to the same relative location on a machine with the same OS, architecture and parser pack version. Without an override, the parser pack uses the OS cache directory, such as `~/.cache` on Linux, `~/Library/Caches` on macOS, and `%LOCALAPPDATA%` on Windows.

The default set is C, C++, C#, Dart, Go, Java, JavaScript, Julia, Kotlin, Lua, Objective-C, PHP, Python, R, Ruby, Rust, Scala, Solidity, Swift and TypeScript. Extra grammars in the upstream pack can be provisioned explicitly for code chunking.

Matheel uses local grammars for code chunking, Tree-sitter lexical tokenization, CodeBLEU, TSED, and the explicit RUBY tree and graph modes. A missing grammar produces an error with a provisioning command. Raw tokenization and text chunkers can run with an empty grammar cache. RUBY's automatic mode can fall back to its string score. The Hugging Face Space provisions the default set during its image build.

The parser pack ships wheels for Python 3.11–3.14 on Linux x86-64 and ARM64 with glibc 2.34 or later, macOS 15 or later on Intel and Apple silicon, and Windows x64 and ARM64. The code-chunking CI exercises Windows x64. See the [parser pack wheel list](https://pypi.org/project/tree-sitter-language-pack/1.25.0/#files).

## Behavior Notes

- Set `chunk_language` explicitly for reproducible code chunking. The default `text` hint and `auto` use a shebang first, then try the locally available Matheel grammars. Adding grammars to the cache can affect that guess.
- Code chunking now uses a UTF-8 byte budget estimated from the configured Chonkie tokenizer's token count. Tree-sitter may group syntax differently from Chonkie 1.6.5. Matheel returns the decoded chunk text and keeps the existing whitespace trimming and `max_chunks` limit.
- A Chonkie-backed method raises an import error if Chonkie is missing.

## CLI Example

```bash
python examples/sample_data.py --output sample_pairs.zip --overwrite
matheel compare sample_pairs.zip \
  --chunking-method code \
  --chunk-language java \
  --chunk-size 120 \
  --max-chunks 8
```

## Python Example

```python
from matheel.chunking import chunk_text

chunks = chunk_text(
    "public class Demo { int add(int a, int b) { return a + b; } }",
    method="code",
    chunk_size=80,
    max_chunks=4,
    chunk_language="java",
)
print(chunks)
```
