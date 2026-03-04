from matheel.similarity import calculate_similarity


LEFT = "def normalize(name):\n    return name.strip().lower()\n"
RIGHT = "def normalise(name):\n    return name.strip().lower()\n"


def main():
    levenshtein_only = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        feature_weights={"levenshtein": 1.0},
    )
    jaro_only = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        feature_weights={"jaro_winkler": 1.0},
    )
    blended = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        feature_weights={
            "semantic": 0.4,
            "levenshtein": 0.4,
            "jaro_winkler": 0.2,
        },
    )

    print("Levenshtein only:", levenshtein_only)
    print("Jaro-Winkler only:", jaro_only)
    print("Blended:", blended)


if __name__ == "__main__":
    main()
