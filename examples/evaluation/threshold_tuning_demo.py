import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

from matheel.calibration import write_threshold_tuning_report_artifacts  # noqa: E402


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
        output_dir = Path(tmp) / "threshold_tuning"
        report, artifacts = write_threshold_tuning_report_artifacts(
            scored_pairs,
            output_dir,
            score_key="similarity_score",
            label_key="label",
            optimize="f1",
        )

        best = report["summary"]["optimized_threshold"]
        print("Best threshold:", best["threshold"])
        print("Best F1:", round(best["f1"], 4))
        print("Threshold sweep:", artifacts["threshold_sweep_csv"])


if __name__ == "__main__":
    main()
