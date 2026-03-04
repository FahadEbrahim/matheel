from matheel.model_routing import available_vector_backends, infer_model_capabilities
from matheel.similarity import calculate_similarity
from matheel.vectors import available_pooling_methods, available_similarity_functions


LEFT = "def add(a, b):\n    return a + b\n"
RIGHT = "def add(x, y):\n    return x + y\n"


def main():
    print("Vector backends:", available_vector_backends())
    print("Similarity functions:", available_similarity_functions())
    print("Pooling methods:", available_pooling_methods())
    print()

    capabilities = infer_model_capabilities("sentence-transformers/all-MiniLM-L6-v2")
    print("Auto routing hint:", capabilities)
    print()

    score = calculate_similarity(
        LEFT,
        RIGHT,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        similarity_function="dot",
        pooling_method="max",
        feature_weights={"semantic": 1.0},
    )
    print("Dense score:", score)


if __name__ == "__main__":
    main()
