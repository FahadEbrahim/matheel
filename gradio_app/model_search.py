"""Model selection and on-demand search with native Gradio components."""

import gradio as gr
from huggingface_hub import HfApi
from huggingface_hub.errors import OfflineModeIsEnabled


class ModelDropdown(gr.Dropdown):
    is_template = True

    def api_info(self):
        # Model IDs and local paths are valid even when absent from the suggestions.
        return {"type": "string"}


def search_model_choices(query, selected_model):
    query = (query or "").strip()
    if not query:
        return gr.skip(), gr.update(value="Enter a model name or keyword.", visible=True)

    try:
        matches = [model.id for model in HfApi().list_models(search=query, limit=12, token=False)]
    except OfflineModeIsEnabled:
        return gr.skip(), gr.update(
            value="Hugging Face search is offline. You can enter a model ID or local path above.",
            visible=True,
        )
    except OSError:
        return gr.skip(), gr.update(
            value="Hugging Face search is unavailable. Try again or enter a model ID above.",
            visible=True,
        )

    if not matches:
        return gr.skip(), gr.update(value="No matching models found.", visible=True)

    choices = [selected_model, *matches] if selected_model else matches
    return gr.update(choices=list(dict.fromkeys(choices))), gr.update(
        value="Choose a result from the model dropdown above.", visible=True,
    )


def build_model_search(*, value, label):
    model = ModelDropdown(
        choices=[value] if value else [],
        value=value,
        label=label,
        info="Select a model or enter an exact repository ID or local path.",
        allow_custom_value=True,
    )
    with gr.Accordion("Search Hugging Face", open=False):
        with gr.Row():
            query = gr.Textbox(label="Model search query", placeholder="Model name or keyword")
            search = gr.Button("Search models", size="sm", scale=0)
        status = gr.Markdown(visible=False)
    gr.on(
        [search.click, query.submit],
        fn=search_model_choices,
        inputs=[query, model],
        outputs=[model, status],
        api_visibility="private",
        trigger_mode="always_last",
    )
    return model
