import sys
from pathlib import Path

from matheel.similarity import get_sim_list

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import sample_archive_path  # noqa: E402


def main():
    results = get_sim_list(
        sample_archive_path(),
        model_name="huggingface/CodeBERTa-small-v1",
        threshold=0.2,
        number_results=10,
        feature_weights={
            "semantic": 0.7,
            "levenshtein": 0.3,
        },
    )
    print(results.head().round(4))


if __name__ == "__main__":
    main()
