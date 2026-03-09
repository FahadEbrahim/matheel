from matheel.similarity import calculate_similarity


LEFT = "def normalize(name):\n    return name.strip().lower()\n"
RIGHT = "def normalise(name):\n    return name.strip().lower()\n"


def main():
    levenshtein_only = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        feature_weights={"levenshtein": 1.0},
    )
    jaro_only = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        feature_weights={"jaro_winkler": 1.0},
    )
    winnowing_only = calculate_similarity(
        LEFT,
        RIGHT,
        feature_weights={"winnowing": 1.0},
        winnowing_kgram=3,
        winnowing_window=2,
    )
    gst_only = calculate_similarity(
        LEFT,
        RIGHT,
        feature_weights={"gst": 1.0},
        gst_min_match_length=2,
    )
    blended = calculate_similarity(
        LEFT,
        RIGHT,
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

    print("Levenshtein only:", levenshtein_only)
    print("Jaro-Winkler only:", jaro_only)
    print("Winnowing only:", winnowing_only)
    print("GST only:", gst_only)
    print("Blended:", blended)


if __name__ == "__main__":
    main()
