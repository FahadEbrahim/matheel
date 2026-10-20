"""Compare lexical configurations on the sample archive without model downloads."""

import sys
from pathlib import Path

from matheel.comparison_suite import run_comparison_suite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import sample_archive_path  # noqa: E402


RUNS = [
    {
        "run_name": "levenshtein",
        "options": {
            "feature_weights": {"levenshtein": 1.0},
        },
    },
    {
        "run_name": "lcs_rapidfuzz_blend",
        "options": {
            "preprocess_mode": "basic",
            "code_language": "java",
            "feature_weights": {
                "lcs": 0.5,
                "rapidfuzz_ratio": 0.5,
            },
        },
    },
]


def main():
    summary, details = run_comparison_suite(sample_archive_path(), RUNS)
    print(summary.round(4))
    print()
    print(details["lcs_rapidfuzz_blend"].head().round(4))


if __name__ == "__main__":
    main()
