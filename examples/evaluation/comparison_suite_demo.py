import sys
from pathlib import Path

from matheel.comparison_suite import run_comparison_suite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import sample_archive_path  # noqa: E402


RUNS = [
    {
        "run_name": "semantic_levenshtein_blend",
        "options": {
            "model_name": "huggingface/CodeBERTa-small-v1",
            "vector_backend": "sentence_transformers",
            "feature_weights": {
                "semantic": 0.7,
                "levenshtein": 0.3,
            },
        },
    },
    {
        "run_name": "codebleu_java_blend",
        "options": {
            "model_name": "huggingface/CodeBERTa-small-v1",
            "vector_backend": "sentence_transformers",
            "preprocess_mode": "basic",
            "chunking_method": "none",
            "code_metric": "codebleu",
            "code_language": "java",
            "code_metric_weight": 0.2,
            "feature_weights": {
                "semantic": 0.8,
                "code_metric": 0.2,
            },
        },
    },
]


def main():
    summary, details = run_comparison_suite(sample_archive_path(), RUNS)
    print(summary.round(4))
    print()
    print(details["semantic_levenshtein_blend"].head().round(4))


if __name__ == "__main__":
    main()
