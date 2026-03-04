from pathlib import Path

from matheel.comparison_suite import run_comparison_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ARCHIVE = REPO_ROOT / "sample_pairs.zip"


RUNS = [
    {
        "run_name": "dense_baseline",
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
        "run_name": "code_metric_blend",
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
    summary, details = run_comparison_suite(SAMPLE_ARCHIVE, RUNS)
    print(summary)
    print()
    print(details["dense_baseline"].head())


if __name__ == "__main__":
    main()
