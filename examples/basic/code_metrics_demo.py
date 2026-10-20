"""Compare parser-based code metrics; opt into CodeBERTScore model downloads."""

import argparse
import sys
from pathlib import Path

from matheel.similarity import calculate_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import CODE_A_NAME, CODE_B_NAME, load_sample_pair  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebertscore",
        action="store_true",
        help="Run CodeBERTScore (requires .[metrics] and may download model weights).",
    )
    args = parser.parse_args()
    code_a, code_b = load_sample_pair()

    print(f"Comparing Code A ({CODE_A_NAME}) with Code B ({CODE_B_NAME})")

    metric_names = ["codebleu", "crystalbleu", "ruby", "tsed"]
    if args.codebertscore:
        metric_names.append("codebertscore")

    for name in metric_names:
        try:
            score = calculate_similarity(
                code_a,
                code_b,
                code_metric=name,
                code_language="java",
                crystalbleu_trivial_ngram_count=0,
                feature_weights={"code_metric": 1.0},
            )
        except ImportError as exc:
            print(f"{name} skipped: {exc}")
        else:
            print(f"{name}: {score:.4f}")

    if not args.codebertscore:
        print("Use --codebertscore to load its model (requires .[metrics]).")


if __name__ == "__main__":
    main()
