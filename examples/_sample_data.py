from pathlib import Path
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ARCHIVE = REPO_ROOT / "sample_pairs.zip"
CODE_A_NAME = "code_1.java"
CODE_B_NAME = "code_3_plag.java"


def load_sample_pair():
    with ZipFile(SAMPLE_ARCHIVE) as archive:
        code_a = archive.read(CODE_A_NAME).decode("utf-8")
        code_b = archive.read(CODE_B_NAME).decode("utf-8")
    return code_a, code_b
