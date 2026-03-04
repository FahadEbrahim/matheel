# Documentation Index

Matheel is organized around a few simple pieces: normalize the code, optionally chunk it, encode it, add lexical and code-aware metrics, then compare or batch-run multiple configurations.

Use this page as the main index for the docs.

## Guides

- [Quick usage](usage.md)
- [Preprocessing](preprocessing.md)
- [Chunking](chunking.md)
- [Vectors and routing](vectors.md)
- [Edit distance and feature weights](lexical.md)
- [Code metrics](code_metrics.md)
- [Comparison suite](comparison_suite.md)

## Suggested Reading Order

1. Start with [Quick usage](usage.md).
2. Read [Vectors and routing](vectors.md) to choose your embedding path.
3. Add [Chunking](chunking.md) and [Preprocessing](preprocessing.md) if you need code-aware shaping before scoring.
4. Add [Edit distance and feature weights](lexical.md) and [Code metrics](code_metrics.md) if you want hybrid scoring.
5. Use [Comparison suite](comparison_suite.md) for repeatable multi-run experiments.

## GitHub Pages

Yes, a GitHub Pages docs site is a good next step once the text stabilizes. The current `docs/` structure is already suitable for that:

- keep `/docs` as the documentation source
- use a simple static Pages publish first, or
- switch to MkDocs later if you want navigation/search/versioned docs

For now, keeping the Markdown files in-repo is the right low-friction option.
