from matheel.vectors import build_static_hash_vector, multivector_similarity


def test_static_hash_vector_is_normalized_for_same_token_bag():
    left = build_static_hash_vector("Alpha beta alpha", dim=64, lowercase=True)
    right = build_static_hash_vector("alpha Alpha beta", dim=64, lowercase=True)

    assert round(float(left.dot(right)), 6) == 1.0


def test_multivector_similarity_is_one_for_identical_vectors():
    left = [[1.0, 0.0], [0.0, 1.0]]
    right = [[1.0, 0.0], [0.0, 1.0]]

    assert multivector_similarity(left, right, bidirectional=True) == 1.0
