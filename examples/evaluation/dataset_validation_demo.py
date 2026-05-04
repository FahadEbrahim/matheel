import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

_MPLCONFIGDIR = Path(gettempdir()) / "matheel_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

from matheel.dataset_validation import write_dataset_validation_report  # noqa: E402
from matheel.datasets import write_pair_dataset  # noqa: E402


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_root = root / "normalized_pairs"
        write_pair_dataset(
            dataset_root,
            files=pd.DataFrame(
                [
                    {"file_id": "a", "text": "print(1)", "suffix": ".py"},
                    {"file_id": "b", "text": "print(2)", "suffix": ".py"},
                ]
            ),
            pairs=pd.DataFrame([{"left_id": "a", "right_id": "b", "label": 0}]),
            metadata={"name": "demo_pairs"},
        )

        report, artifacts = write_dataset_validation_report(
            dataset_root,
            root / "validation_report",
            kind="pair",
        )

        print("Status:", report["status"])
        print("Errors:", report["error_count"])
        print("Warnings:", report["warning_count"])
        print("HTML report:", artifacts["report_html"])


if __name__ == "__main__":
    main()
