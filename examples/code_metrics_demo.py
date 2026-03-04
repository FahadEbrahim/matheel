from matheel.similarity import calculate_similarity


LEFT = "def add(a, b):\n    return a + b\n"
RIGHT = "def sum_two(x, y):\n    return x + y\n"
MODEL = "huggingface/CodeBERTa-small-v1"


def main():
    codebleu_score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="codebleu",
        code_metric_weight=1.0,
        code_language="python",
        feature_weights={"code_metric": 1.0},
    )
    crystalbleu_score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="crystalbleu",
        code_metric_weight=1.0,
        code_language="python",
        crystalbleu_trivial_ngram_count=0,
        feature_weights={"code_metric": 1.0},
    )
    ruby_score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="ruby",
        code_metric_weight=1.0,
        code_language="python",
        ruby_max_order=4,
        feature_weights={"code_metric": 1.0},
    )

    print("CodeBLEU:", codebleu_score)
    print("CrystalBLEU:", crystalbleu_score)
    print("RUBY:", ruby_score)

    try:
        tsed_score = calculate_similarity(
            LEFT,
            RIGHT,
            model_name=MODEL,
            vector_backend="sentence_transformers",
            code_metric="tsed",
            code_metric_weight=1.0,
            code_language="python",
            tsed_delete_cost=1.0,
            tsed_insert_cost=1.0,
            tsed_rename_cost=1.0,
            feature_weights={"code_metric": 1.0},
        )
        print("TSED:", tsed_score)
    except ImportError as exc:
        print("TSED skipped:", exc)

    try:
        codebertscore = calculate_similarity(
            LEFT,
            RIGHT,
            model_name=MODEL,
            vector_backend="sentence_transformers",
            code_metric="codebertscore",
            code_metric_weight=1.0,
            codebertscore_model="microsoft/codebert-base",
            codebertscore_batch_size=8,
            feature_weights={"code_metric": 1.0},
        )
        print("CodeBERTScore:", codebertscore)
    except ImportError as exc:
        print("CodeBERTScore skipped:", exc)


if __name__ == "__main__":
    main()
