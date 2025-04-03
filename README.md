This is the repository for the demonstration paper "Matheel: A Hybrid Source Code Plagiairsm Software".

## Demonstration:
A demo is available hosted on [Huggingface Spaces](https://huggingface.co/spaces/buelfhood/matheel). It can be used on the UI.

A Video recording of how to use the tool is available on [YouTube](https://www.youtube.com/watch?v=P8BYCx8X48I).

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
		model_name="uclanlp/plbart-java-cs",
		threshold=0,
		number_results=10,
		api_name="/get_sim_list"
)
print(result)
```

## Acknowledgement:

- The demo uses code written by SBERT. [Webpage](https://www.sbert.net/index.html), [Repo](https://github.com/UKPLab/sentence-transformers).
- The code is built with Gradio. [Webpage](). [Repo]()

## Reference:
This will be provided later.

