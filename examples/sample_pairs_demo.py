from matheel.similarity import get_sim_list

from _sample_data import SAMPLE_ARCHIVE


def main():
    results = get_sim_list(
        SAMPLE_ARCHIVE,
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
