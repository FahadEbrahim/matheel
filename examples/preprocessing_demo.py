from matheel.preprocessing import available_preprocess_modes, preprocess_code

from _sample_data import CODE_A_NAME, load_sample_pair


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
