# Documentation Index

Use this page as the main in-repo docs index.

Matheel is organized around a simple flow:

1. preprocess or normalize the code
2. optionally chunk it
3. encode it or score it lexically
4. add code-aware metrics when needed
5. compare one pair or rank a whole archive

## Guides

- [Quick usage](usage.md)
- [Preprocessing](preprocessing.md)
- [Chunking](chunking.md)
- [Vectors and routing](vectors.md)
- [Lexical metrics and baselines](lexical.md)
- [Code metrics](code_metrics.md)
- [Comparison suite](comparison_suite.md)

## Demos and Examples

- [README demos](../README.md#demos)
- [Examples Colab notebook](../examples/matheel_examples_colab.ipynb)
- [Examples folder](../examples/)

## Suggested Reading Order

1. Start with [Quick usage](usage.md).
2. Read [Vectors and routing](vectors.md) to choose the embedding path.
3. Add [Chunking](chunking.md) and [Preprocessing](preprocessing.md) if you need code-aware shaping before scoring.
4. Add [Lexical metrics and baselines](lexical.md) and [Code metrics](code_metrics.md) if you want hybrid scoring.
5. Use [Comparison suite](comparison_suite.md) for repeatable multi-run experiments.
