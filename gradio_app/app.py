# Import necessary libraries
import gradio as gr
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sentence_transformers import SentenceTransformer, util
from typing import List
import zipfile
import os
import io
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from matheel.similarity import get_sim_list

def calculate_similarity(code1, code2, Ws, Wl, Wj, model_name):
    model = SentenceTransformer(model_name)
    embedding1 = model.encode(code1)
    embedding2 = model.encode(code2)
    sim_similarity = util.cos_sim(embedding1, embedding2).item()
    lev_ratio = Levenshtein.normalized_similarity(code1, code2)
    jaro_winkler_ratio = JaroWinkler.normalized_similarity(code1, code2)
    overall_similarity = Ws * sim_similarity + Wl * lev_ratio + Wj * jaro_winkler_ratio

    return "The similarity score between the two codes is: %.2f" % overall_similarity

def get_sim_list_gradio(zipped_file,Ws, Wl, Wj, model_name,threshold,number_results):
    result = get_sim_list(zipped_file,Ws, Wl, Wj, model_name,threshold,number_results)
    return result

# Define the Gradio app
with gr.Blocks() as demo:
    # Tab for similarity calculation
    with gr.Tab("Code Pair Similarity"):
        # Input components
        code1 = gr.Textbox(label="Code 1")
        code2 = gr.Textbox(label="Code 2")

        model_dropdown = HuggingfaceHubSearch(
                label="Pre-Trained Model to use for Embeddings",
                placeholder="Search for Pre-Trained models on Hugging Face",
                search_type="model",
                #value = "huggingface/CodeBERTa-small-v1"
            )

        # Accordion for weights and models
        with gr.Accordion("Weights and Models", open=False):
            Ws = gr.Slider(0, 1, value=0.7, label="Semantic Search Weight", step=0.1)
            Wl = gr.Slider(0, 1, value=0.3, label="Levenshiern Distance Weight", step=0.1)
            Wj = gr.Slider(0, 1, value=0.0, label="Jaro Winkler Weight", step=0.1)
            

        # Output component
        output = gr.Textbox(label="Similarity Score")

        def update_weights(Ws, Wl, Wj):
            total = Ws + Wl + Wj
            if total != 1:
                Wj = 1 - (Ws + Wl)
            return Ws, Wl, Wj

        # Update weights when any slider changes
        Ws.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])
        Wl.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])
        Wj.change(update_weights, [Ws, Wl, Wj], [Ws, Wl, Wj])

        # Button to trigger the similarity calculation
        calculate_btn = gr.Button("Calculate Similarity")
        calculate_btn.click(calculate_similarity, inputs=[code1, code2, Ws, Wl, Wj, model_dropdown], outputs=output)

    # Tab for file upload and DataFrame output
    with gr.Tab("Code Collection Pair Similarity"):
        # File uploader component
        file_uploader = gr.File(label="Upload a Zip file",file_types=[".zip"])

        model_dropdown = HuggingfaceHubSearch(
                label="Pre-Trained Model to use for Embeddings",
                placeholder="Search for Pre-Trained models on Hugging Face",
                search_type="model",
                #value = "huggingface/CodeBERTa-small-v1"
            )

        with gr.Accordion("Weights and Models", open=False):
            Ws = gr.Slider(0, 1, value=0.7, label="Semantic Search Weight", step=0.1)
            Wl = gr.Slider(0, 1, value=0.3, label="Levenshiern Distance Weight", step=0.1)
            Wj = gr.Slider(0, 1, value=0.0, label="Jaro Winkler Weight", step=0.1)
            
            threshold = gr.Slider(0, 1, value=0, label="Threshold", step=0.01)
            number_results = gr.Slider(1, 1000, value=10, label="Number of Returned pairs", step=1)

        # Output component for the DataFrame
        df_output = gr.Dataframe(label="Results")

        # Button to trigger the file processing
        process_btn = gr.Button("Process File")
        process_btn.click(get_sim_list, inputs=[file_uploader, Ws, Wl, Wj, model_dropdown,threshold,number_results], outputs=df_output)

# Launch the Gradio app with live=True
demo.launch(show_error=True,debug=True)