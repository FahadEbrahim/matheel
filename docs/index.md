# Matheel Documentation

Matheel is a Python package and CLI for source-code similarity. It combines semantic embeddings, lexical similarity, chunking, preprocessing, and code evaluation metrics in one workflow.

Matheel is organized around a simple flow:

1. preprocess or normalize the code
2. optionally chunk it
3. encode it or score it lexically
4. add code-aware metrics when needed
5. compare one pair or rank a whole archive

Start with the [usage guide](usage.md) for installation, optional extras, quick checks, demos, and example links.

## Guides

- [Usage](usage.md)
- [Preprocessing](preprocessing.md)
- [Chunking](chunking.md)
- [Vectors and routing](vectors.md)
- [Lexical metrics and baselines](lexical.md)
- [Code metrics](code_metrics.md)
- [Datasets and evaluation](datasets.md)
- [Comparison suite](comparison_suite.md)
- [Development](development.md)
- [Release checklist](release_checklist.md)

## Demos and Examples

- [Hugging Face Space demo](https://huggingface.co/spaces/buelfhood/matheel-framework)
- [Gradio Colab notebook](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_gradio_colab_demo.ipynb)
- [Examples Colab notebook](https://colab.research.google.com/github/FahadEbrahim/matheel/blob/main/examples/matheel_examples_colab.ipynb)
- [Examples folder](https://github.com/FahadEbrahim/matheel/tree/main/examples)

## Suggested Reading Order

1. Start with [Usage](usage.md).
2. Read [Vectors and routing](vectors.md) to choose the embedding path.
3. Add [Chunking](chunking.md) and [Preprocessing](preprocessing.md) if you need code-aware shaping before scoring.
4. Add [Lexical metrics and baselines](lexical.md) and [Code metrics](code_metrics.md) if you want hybrid scoring.
5. Use [Datasets and evaluation](datasets.md) for labeled pair datasets.
6. Use [Comparison suite](comparison_suite.md) for repeatable multi-run experiments.
