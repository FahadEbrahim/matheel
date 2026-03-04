from matheel.similarity import calculate_similarity


LEFT = "def add(a, b):\n    return a + b\n"
RIGHT = "def sum_two(x, y):\n    return x + y\n"


def main():
    codebleu_score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        code_metric="codebleu",
        code_metric_weight=1.0,
        code_language="python",
        feature_weights={"code_metric": 1.0},
    )
    crystalbleu_score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        code_metric="crystalbleu",
        code_metric_weight=1.0,
        code_language="python",
        crystalbleu_trivial_ngram_count=0,
        feature_weights={"code_metric": 1.0},
    )

    print("CodeBLEU:", codebleu_score)
    print("CrystalBLEU:", crystalbleu_score)


if __name__ == "__main__":
    main()
