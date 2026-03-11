from matheel.similarity import calculate_similarity

from _sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair


MODEL = "huggingface/CodeBERTa-small-v1"


def main():
    code_a, code_b = load_sample_pair()

    print(f"Comparing Code A ({CODE_A_NAME}) with Code B ({CODE_B_NAME})")

    codebleu_score = calculate_similarity(
        code_a,
        code_b,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="codebleu",
        code_metric_weight=1.0,
        code_language="java",
        feature_weights={"code_metric": 1.0},
    )
    crystalbleu_score = calculate_similarity(
        code_a,
        code_b,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="crystalbleu",
        code_metric_weight=1.0,
        code_language="java",
        crystalbleu_trivial_ngram_count=0,
        feature_weights={"code_metric": 1.0},
    )
    ruby_score = calculate_similarity(
        code_a,
        code_b,
        model_name=MODEL,
        vector_backend="sentence_transformers",
        code_metric="ruby",
        code_metric_weight=1.0,
        code_language="java",
        ruby_max_order=4,
        feature_weights={"code_metric": 1.0},
    )

    print("CodeBLEU:", round(codebleu_score, 4))
    print("CrystalBLEU:", round(crystalbleu_score, 4))
    print("RUBY:", round(ruby_score, 4))

    try:
        tsed_score = calculate_similarity(
            code_a,
            code_b,
            model_name=MODEL,
            vector_backend="sentence_transformers",
            code_metric="tsed",
            code_metric_weight=1.0,
            code_language="java",
            tsed_delete_cost=1.0,
            tsed_insert_cost=1.0,
            tsed_rename_cost=1.0,
            feature_weights={"code_metric": 1.0},
        )
        print("TSED:", round(tsed_score, 4))
    except ImportError as exc:
        print("TSED skipped:", exc)

    try:
        codebertscore = calculate_similarity(
            code_a,
            code_b,
            model_name=MODEL,
            vector_backend="sentence_transformers",
            code_metric="codebertscore",
            code_metric_weight=1.0,
            codebertscore_model="microsoft/codebert-base",
            codebertscore_batch_size=8,
            feature_weights={"code_metric": 1.0},
        )
        print("CodeBERTScore:", round(codebertscore, 4))
    except ImportError as exc:
        print("CodeBERTScore skipped:", exc)


if __name__ == "__main__":
    main()
