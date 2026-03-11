from matheel.chunking import available_chunking_methods, chunk_text, chunker_parameter_names

from _sample_data import CODE_A_NAME, load_sample_pair


def main():
    code_a, _ = load_sample_pair()

    print("Available chunking methods:", available_chunking_methods())
    print("Code chunker parameters:", chunker_parameter_names("code"))
    print()
    print(f"Loaded Code A from {CODE_A_NAME}")
    print()

    no_chunking = chunk_text(
        code_a,
        method="none",
        chunk_size=120,
        chunk_overlap=0,
        max_chunks=0,
        chunk_language="java",
    )
    print("No-chunking output:")
    print(no_chunking[0])
    print()

    try:
        code_chunks = chunk_text(
            code_a,
            method="code",
            chunk_size=64,
            chunk_overlap=0,
            max_chunks=3,
            chunk_language="java",
        )
    except ImportError as exc:
        print("Code chunking skipped:", exc)
    else:
        print("First code chunks:")
        for index, chunk in enumerate(code_chunks, start=1):
            print(f"Chunk {index}:")
            print(chunk)
            print()


if __name__ == "__main__":
    main()
