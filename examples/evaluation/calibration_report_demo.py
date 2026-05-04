import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

# Configure Matplotlib before importing Matheel's evaluation stack.
_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

from matheel.calibration import write_calibration_report_artifacts  # noqa: E402


def main():
    scored_pairs = pd.DataFrame(
        [
            {"left_id": "a", "right_id": "b", "similarity_score": 0.95, "label": 1},
            {"left_id": "c", "right_id": "d", "similarity_score": 0.82, "label": 1},
            {"left_id": "e", "right_id": "f", "similarity_score": 0.30, "label": 0},
            {"left_id": "g", "right_id": "h", "similarity_score": 0.10, "label": 0},
        ]
    )

    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "calibration_artifacts"
        report, artifacts = write_calibration_report_artifacts(
            scored_pairs,
            output_dir,
            score_key="similarity_score",
            label_key="label",
        )

        print("AUROC:", round(report["summary"]["auroc"], 4))
        print("Average precision:", round(report["summary"]["average_precision"], 4))
        print("Best threshold:", report["summary"]["optimized_threshold"]["threshold"])
        print("Summary artifact:", artifacts["summary_json"])


if __name__ == "__main__":
    main()
