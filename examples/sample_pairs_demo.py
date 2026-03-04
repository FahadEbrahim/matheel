from pathlib import Path

from matheel.similarity import get_sim_list


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ARCHIVE = REPO_ROOT / "sample_pairs.zip"


def main():
    results = get_sim_list(
        SAMPLE_ARCHIVE,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.2,
        number_results=10,
        feature_weights={
            "semantic": 0.7,
            "levenshtein": 0.3,
        },
    )
    print(results.head())


if __name__ == "__main__":
    main()
