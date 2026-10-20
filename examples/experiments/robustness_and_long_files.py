"""Run from a checkout; uses existing offline scorers and character chunking."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matheel.long_files import compare_long_files, default_long_file_runs, preflight_long_file  # noqa: E402
from matheel.robustness import run_robustness_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs/experiments"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        {"run_name": "levenshtein", "feature_weights": {"levenshtein": 1}},
        {"run_name": "winnowing", "feature_weights": {"winnowing": 1}},
        {"run_name": "static_hash", "feature_weights": {"semantic": 1}, "vector_backend": "static_hash"},
    ]
    robustness, robustness_paths = run_robustness_experiment(
        runs, seed=args.seed, output_dir=args.output_dir / "robustness",
    )
    common = "".join(f"prefix_{index} = {index}\n" for index in range(50))
    left = common + "".join(f"left_branch_{index} = {index * 7}\n" for index in range(100))
    right = common + "".join(f"different_tail_{index} = '{index}'\n" for index in range(100))
    (args.output_dir / "left.py").write_text(left, encoding="utf-8")
    (args.output_dir / "right.py").write_text(right, encoding="utf-8")
    (args.output_dir / "long_file_runs.json").write_text(
        json.dumps({"runs": default_long_file_runs()}, indent=2) + "\n", encoding="utf-8",
    )
    long_files, long_paths = compare_long_files(left, right, output_dir=args.output_dir / "long_files")
    unknown = preflight_long_file(left, options={"vector_backend": "sentence_transformers",
                                                "max_token_length": 256})
    (args.output_dir / "without_tokenizer.json").write_text(
        json.dumps(unknown, indent=2) + "\n", encoding="utf-8",
    )
    if any(row["status"] != "ok" for row in robustness["scores"] + long_files["scores"]):
        raise RuntimeError("A declared scorer failed; inspect the JSON error rows.")
    print("Robustness: run / transformation / score delta")
    for row in robustness["scores"]:
        print(f"{row['run_name']:14} {row['transformation']:22} {row['score_delta']:+.6f}")
    print("\nLong files: run / similarity / selected left chunks / effective input byte fraction")
    for score, diagnostic in zip(long_files["scores"], long_files["preflight"]):
        preflight = diagnostic["files"]["left"]
        coverage = preflight["effective_model_input_coverage"]["byte_fraction"]
        print(f"{score['run_name']:22} {score['similarity_score']:.6f} "
              f"{preflight['selected_chunks']:4} {coverage:.3f}")
    print("\nReports:", json.dumps({**robustness_paths, "long_files": long_paths}, indent=2))
    print("Transformer token coverage without a tokenizer remains unknown; see without_tokenizer.json.")


if __name__ == "__main__":
    main()
