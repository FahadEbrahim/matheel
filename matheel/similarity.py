import zipfile
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sentence_transformers import SentenceTransformer, util, models
from typing import List

def load_model(model_name: str):
    model = SentenceTransformer(model_name)
    return model

def extract_and_read_compressed_file(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        file_names = z.namelist()
        codes = [z.read(file).decode('utf-8', errors='ignore') for file in file_names]
    return file_names, codes

def paraphrase_mining_with_combined_score(
    model,
    sentences: List[str],
    weight_semantic: float,
    weight_levenshtein: float,
    weight_jaro_winkler: float,
):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    paraphrases = util.paraphrase_mining_embeddings(embeddings, score_function=util.cos_sim)

    results = []
    for score, i, j in paraphrases:
        lev_ratio = Levenshtein.normalized_similarity(sentences[i], sentences[j])
        jw_ratio = JaroWinkler.normalized_similarity(sentences[i], sentences[j])
        combined_score = (
            weight_semantic * score +
            weight_levenshtein * lev_ratio +
            weight_jaro_winkler * jw_ratio
        )
        results.append((combined_score, i, j))
    return sorted(results, reverse=True)

def get_sim_list(zipped_file, Ws, Wl, Wj, model_name,  threshold, number_results):
    file_names, codes = extract_and_read_compressed_file(zipped_file)
    model = load_model(model_name)
    code_pairs = paraphrase_mining_with_combined_score(model, codes, Ws, Wl, Wj)

    pairs_results = [
        {
            'file_name_1': file_names[i],
            'file_name_2': file_names[j],
            'similarity_score': round(score, 2)
        }
        for score, i, j in code_pairs if score >= threshold
    ]

    similarity_df = pd.DataFrame(pairs_results)
    return similarity_df.head(number_results)

def calculate_similarity(code1, code2, Ws, Wl, Wj, model_name):
    model = load_model(model_name)
    embedding1 = model.encode(code1)
    embedding2 = model.encode(code2)
    sim_similarity = util.cos_sim(embedding1, embedding2).item()
    lev_ratio = Levenshtein.normalized_similarity(code1, code2)
    jaro_winkler_ratio = JaroWinkler.normalized_similarity(code1, code2)
    overall_similarity = Ws * sim_similarity + Wl * lev_ratio + Wj * jaro_winkler_ratio

    return overall_similarity
