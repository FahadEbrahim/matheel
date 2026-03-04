from matheel.preprocessing import available_preprocess_modes, preprocess_code


SOURCE = """
def add(a, b):  # helper
    # keep this readable
    return a + b
"""


def main():
    for mode in available_preprocess_modes():
        print(f"=== {mode} ===")
        print(preprocess_code(SOURCE, mode=mode))
        print()


if __name__ == "__main__":
    main()
