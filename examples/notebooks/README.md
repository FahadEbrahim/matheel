# Colab Notebooks

The notebooks are ordered by workflow:

1. `01_core_workflows.ipynb`: preprocessing, chunking, lexical scoring, semantic scoring, code metrics, archive ranking, and comparison suites.
2. `02_datasets_and_reproducibility.ipynb`: pair/retrieval datasets, tabular adapters, resampling, and reproducible benchmark outputs.
3. `03_custom_algorithms.ipynb`: path-loaded custom `score_pair` modules and dataset evaluation.
4. `04_gradio_app.ipynb`: launch the Gradio app in Colab.
5. `05_visualization_and_leaderboard.ipynb`: dataset maps, pair explanations, leaderboard artifacts, and static benchmark reports.

Each notebook installs Matheel in its own runtime so it can be opened independently. The setup cells use a tagged release (`MATHEEL_REF`), which stays synchronized with the package version. They therefore demonstrate that release's APIs. For the expanded LCS and RapidFuzz metrics in the current checkout, start with [the lexical script](../basic/lexical_demo.py) and [the examples guide](../README.md).

The first setup cell requires internet access to install dependencies. Core lexical scoring, local datasets, custom algorithms, PCA maps, and leaderboards use generated data. Semantic scoring and CodeBERTScore can download model weights; the Gradio launch creates a public sharing link.
