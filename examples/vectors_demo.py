from matheel.model_routing import available_vector_backends, infer_model_capabilities
from matheel.similarity import calculate_similarity
from matheel.vectors import available_pooling_methods, available_similarity_functions

from _sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair


def main():
    code_a, code_b = load_sample_pair()

    print("Vector backends:", available_vector_backends())
    print("Similarity functions:", available_similarity_functions())
    print("Pooling methods:", available_pooling_methods())
    print()
    print(f"Comparing Code A ({CODE_A_NAME}) with Code B ({CODE_B_NAME})")
    print()

    capabilities = infer_model_capabilities("huggingface/CodeBERTa-small-v1")
    print("Auto routing hint:", capabilities)
    print()

    score = calculate_similarity(
        code_a,
        code_b,
        model_name="huggingface/CodeBERTa-small-v1",
        vector_backend="sentence_transformers",
        similarity_function="dot",
        pooling_method="max",
        feature_weights={"semantic": 1.0},
    )
    print("Dense score:", round(score, 4))


if __name__ == "__main__":
    main()
