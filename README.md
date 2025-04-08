This is the repository for the demonstration paper "Matheel: A Hybrid Source Code Plagiarism Detection Software".

# Matheel

Matheel is a Python package designed to detect source code similarity. It integrates semantic similarity models with traditional edit distance, providing a robust approach to detecting similarities among source code snippets.

---

## Features

- **Semantic Similarity:** Uses Pre-Trained models.
- **Edit-distance Metrics:** Integrates Levenshtein and Jaro-Winkler similarity scores.
- **Combined Weighted Similarity:** Adjustable weights for semantic and syntactic similarity.
- **Easy CLI & Python API:** Suitable for both interactive and automated workflows.
- **Interactive UI:** Gradio-based user interface.

---

## Installation

Install via pip:

```bash
pip install matheel
```
---

## Usage

### CLI Usage

Compare files within a compressed ZIP archive:

```bash
matheel compare codes.zip --model buelfhood/unixcoder-base-unimodal-ST --threshold 0.5 --num 100
```

### Python API Usage

To calculate similarities programmatically:

```python
from matheel.similarity import get_sim_list

# Define parameters
zip_file = "sample_codes.zip"
Ws, Wl, Wj = 0.7, 0.2, 0.1
model_name = "buelfhood/unixcoder-base-unimodal-ST"
threshold = 0.5
number_results = 100

# Get similarity results
results = get_sim_list(zip_file, Ws, Wl, Wj, model_name, threshold, number_results)

# Display results
print(results)
```
---

## Gradio GUI:

The gradio_app folder contains a notebook that allows you to run the Gradio through a Jupyter Notebook.
Also, a demo is available hosted on [Huggingface Spaces](https://huggingface.co/spaces/buelfhood/Matheel).

## Using Gradio API:
The tool can be used through the Gradio API as per the following call:

```python
#pip install gradio_client
from gradio_client import Client, handle_file

client = Client("buelfhood/Matheel")
result = client.predict(
		zipped_file=handle_file('zip file path'),
		Ws=0.7,
		Wl=0.3,
		Wj=0,
		model_name="buelfhood/unixcoder-base-unimodal-ST",
		threshold=0,
		number_results=10,
		api_name="/get_sim_list"
)
print(result)
```

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Acknowledgement:

- The demo uses code written by SBERT. [Webpage](https://www.sbert.net/index.html), [Repo](https://github.com/UKPLab/sentence-transformers).
- The code is built with Gradio. [Webpage](https://www.gradio.app/). [Repo](https://github.com/gradio-app/gradio)
- The code uses RapidFuzz for the edit distance. [Webpage](https://rapidfuzz.github.io/RapidFuzz/). [Repo](https://github.com/rapidfuzz/RapidFuzz)
