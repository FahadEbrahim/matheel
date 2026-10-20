"""Inspect vector options locally; opt into model downloads with CLI flags."""

import argparse
import sys
from pathlib import Path

from matheel.model_routing import available_vector_backends, infer_model_capabilities
from matheel.similarity import calculate_similarity
from matheel.vectors import (
    available_pooling_methods,
    available_similarity_functions,
    build_static_hash_vectors,
    single_vector_similarity,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="Run a dense embedding model.")
    parser.add_argument("--multivector", action="store_true", help="Run a multi-vector model.")
    args = parser.parse_args()
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

    vectors = build_static_hash_vectors([code_a, code_b])
    score = single_vector_similarity(vectors[0], vectors[1], similarity_function="cosine")
    print("Static hash cosine score:", round(score, 4))
    print("Static hashing scores shared tokens without loading a model.")

    if args.semantic:
        dense_score = calculate_similarity(
            code_a,
            code_b,
            model_name="huggingface/CodeBERTa-small-v1",
            vector_backend="sentence_transformers",
            similarity_function="cosine",
            pooling_method="mean",
            feature_weights={"semantic": 1.0},
        )
        print("Dense semantic score:", round(dense_score, 4))

    if args.multivector:
        multivector_score = calculate_similarity(
            code_a,
            code_b,
            model_name="NeuML/pylate-bert-tiny",
            vector_backend="multivector",
            multivector_bidirectional=True,
            feature_weights={"semantic": 1.0},
        )
        print("Multi-vector score:", round(multivector_score, 4))

    if not (args.semantic or args.multivector):
        print("Use --semantic or --multivector to load a model (requires .[semantic]).")


if __name__ == "__main__":
    main()
