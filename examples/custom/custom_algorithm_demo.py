import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir


_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

import pandas as pd  # noqa: E402

from matheel.algorithms import score_pair_with_algorithm, score_source_pairs_with_algorithm  # noqa: E402
from matheel.datasets import write_pair_dataset  # noqa: E402
from matheel.evaluation import evaluate_pair_dataset  # noqa: E402


def write_custom_algorithm(path):
    Path(path).write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, bias=0.0, **kwargs):",
                "    _ = kwargs",
                "    return {'bias': float(bias)}",
                "",
                "def score_pair(code_a, code_b, dataset_context=None, row=None, **kwargs):",
                "    _ = (row, kwargs)",
                "    base = 1.0 if code_a.strip() == code_b.strip() else 0.0",
                "    return base + dataset_context['bias']",
            ]
        ),
        encoding="utf-8",
    )


def main():
    with TemporaryDirectory(prefix="matheel_custom_algorithm_") as temp_dir:
        workspace = Path(temp_dir)
        algorithm_path = workspace / "my_algorithm.py"
        write_custom_algorithm(algorithm_path)

        source_root = workspace / "codes"
        source_root.mkdir()
        (source_root / "a.py").write_text("print(1)\n", encoding="utf-8")
        (source_root / "b.py").write_text("print(1)\n", encoding="utf-8")
        (source_root / "c.py").write_text("print(2)\n", encoding="utf-8")

        archive_scores = score_source_pairs_with_algorithm(
            source_root,
            algorithm=algorithm_path,
            algorithm_options={"bias": 0.1},
            number_results=3,
        )
        print("Custom archive/directory scoring")
        print(archive_scores[["file_name_1", "file_name_2", "similarity_score"]].to_string(index=False))
        print(archive_scores.attrs["algorithm"]["algorithm_source_fingerprint"]["sha256"][:12])
        print()

        dataset_root = workspace / "pairs"
        dataset = write_pair_dataset(
            dataset_root,
            files=pd.DataFrame(
                [
                    {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                    {"file_id": "b", "text": "print(1)", "suffix": ".py"},
                    {"file_id": "c", "text": "print(2)", "suffix": ".py"},
                ]
            ),
            pairs=pd.DataFrame(
                [
                    {"left_id": "a", "right_id": "b", "label": 1},
                    {"left_id": "a", "right_id": "c", "label": 0},
                ]
            ),
        )
        scored, metrics = evaluate_pair_dataset(
            dataset,
            threshold=0.5,
            algorithm=algorithm_path,
            algorithm_options={"bias": 0.1},
        )

        print("Custom dataset evaluation")
        print(scored[["left_id", "right_id", "label", "similarity_score"]].to_string(index=False))
        print({key: round(value, 4) for key, value in metrics.items()})
        print()

        direct_score = score_pair_with_algorithm(
            "print(1)",
            "print(1)",
            algorithm_path,
            algorithm_options={"bias": 0.1},
            dataset_context={"bias": 0.1},
        )
        print("Direct pair score:", round(direct_score, 4))


if __name__ == "__main__":
    main()
