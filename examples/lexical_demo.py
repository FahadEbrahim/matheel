from matheel.similarity import calculate_similarity

from _sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair


def main():
    code_a, code_b = load_sample_pair()

    print(f"Comparing Code A ({CODE_A_NAME}) with Code B ({CODE_B_NAME})")

    levenshtein_only = calculate_similarity(
        code_a,
        code_b,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        feature_weights={"levenshtein": 1.0},
    )
    jaro_only = calculate_similarity(
        code_a,
        code_b,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        feature_weights={"jaro_winkler": 1.0},
    )
    winnowing_only = calculate_similarity(
        code_a,
        code_b,
        feature_weights={"winnowing": 1.0},
        winnowing_kgram=3,
        winnowing_window=2,
    )
    gst_only = calculate_similarity(
        code_a,
        code_b,
        feature_weights={"gst": 1.0},
        gst_min_match_length=2,
    )
    blended = calculate_similarity(
        code_a,
        code_b,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        feature_weights={
            "semantic": 0.35,
            "levenshtein": 0.3,
            "jaro_winkler": 0.15,
            "winnowing": 0.1,
            "gst": 0.1,
        },
    )

    print("Levenshtein only:", round(levenshtein_only, 4))
    print("Jaro-Winkler only:", round(jaro_only, 4))
    print("Winnowing only:", round(winnowing_only, 4))
    print("GST only:", round(gst_only, 4))
    print("Blended:", round(blended, 4))


if __name__ == "__main__":
    main()
