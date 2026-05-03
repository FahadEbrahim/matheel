import sys
from pathlib import Path

from matheel.preprocessing import available_preprocess_modes, preprocess_code

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import CODE_A_NAME, load_sample_pair  # noqa: E402


def main():
    code_a, _ = load_sample_pair()

    print(f"Preprocessing Code A ({CODE_A_NAME})")
    print()

    for mode in available_preprocess_modes():
        print(f"=== {mode} ===")
        print(preprocess_code(code_a, mode=mode))
        print()


if __name__ == "__main__":
    main()
