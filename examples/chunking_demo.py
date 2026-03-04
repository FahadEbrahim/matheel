from matheel.chunking import available_chunking_methods, chunk_text, chunker_parameter_names


SOURCE = """
public class Demo {
    int add(int a, int b) {
        return a + b;
    }

    int sub(int a, int b) {
        return a - b;
    }
}
"""


def main():
    print("Available chunking methods:", available_chunking_methods())
    print("Code chunker parameters:", chunker_parameter_names("code"))
    print()

    chunks = chunk_text(
        SOURCE,
        method="none",
        chunk_size=120,
        chunk_overlap=0,
        max_chunks=0,
        chunk_language="java",
    )
    print("No-chunking output:")
    print(chunks)
    print()

    print("To test Chonkie-backed methods, install matheel[chunking] or matheel[chunking_code].")


if __name__ == "__main__":
    main()
