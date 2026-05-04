import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

# Configure Matplotlib before importing Matheel's evaluation stack.
_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

from matheel.datasets import write_pair_dataset  # noqa: E402
from matheel.visualization import write_dataset_embedding_map, write_pair_dataset_explanation  # noqa: E402


def main():
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        dataset_root = workspace / "normalized_pairs"
        write_pair_dataset(
            dataset_root,
            files=pd.DataFrame(
                [
                    {"file_id": "a", "text": "def add(a, b):\n    return a + b\n", "suffix": ".py"},
                    {"file_id": "b", "text": "def add(x, y):\n    return x + y\n", "suffix": ".py"},
                    {"file_id": "c", "text": "def subtract(a, b):\n    return a - b\n", "suffix": ".py"},
                ]
            ),
            pairs=pd.DataFrame(
                [
                    {"left_id": "a", "right_id": "b", "label": 1},
                    {"left_id": "a", "right_id": "c", "label": 0},
                ]
            ),
            metadata={"name": "tiny_visualization_pairs"},
        )

        projection, artifacts = write_dataset_embedding_map(
            dataset_root,
            workspace / "visualization_artifacts",
            kind="pair",
            method="pca",
        )

        print(projection[["document_id", "x", "y", "role"]].to_string(index=False))
        print("HTML artifact:", artifacts["html"])

        explanation, pair_artifacts = write_pair_dataset_explanation(
            dataset_root,
            workspace / "pair_explanations",
            left_id="a",
            right_id="b",
            segment_mode="line",
        )

        print("Pair matches:", len(explanation["matches"]))
        print("Pair HTML artifact:", pair_artifacts["html"])


if __name__ == "__main__":
    main()
