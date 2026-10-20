"""Compare all RapidFuzz scorers and blend lexical features without a model."""

import sys
from pathlib import Path

from matheel.lexical import available_lexical_metrics, score_lexical_metric
from matheel.similarity import calculate_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair  # noqa: E402


def main():
    code_a, code_b = load_sample_pair()

    print(f"Comparing Code A ({CODE_A_NAME}) with Code B ({CODE_B_NAME})")

    print("\nRaw-text scores (0 to 1):")
    for name in available_lexical_metrics():
        score = score_lexical_metric(name, code_a, code_b)
        print(f"{name:36} {score:.4f}")

    # calculate_similarity supports the same names as weighted features and
    # can apply source-code preprocessing before scoring.
    blended = calculate_similarity(
        code_a,
        code_b,
        preprocess_mode="basic",
        code_language="java",
        feature_weights={
            "lcs": 0.4,
            "rapidfuzz_ratio": 0.4,
            "winnowing": 0.1,
            "gst": 0.1,
        },
        winnowing_kgram=3,
        winnowing_window=2,
        gst_min_match_length=2,
    )
    print("\nPreprocessed LCS/RapidFuzz/fingerprint blend:", round(blended, 4))


if __name__ == "__main__":
    main()
