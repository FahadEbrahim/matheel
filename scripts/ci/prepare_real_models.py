"""Fetch immutable model revisions before running the integration suite offline."""

import argparse
import json
from pathlib import Path


MODELS = {
    "MATHEEL_MINILM_MODEL": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "MATHEEL_STATIC_MODEL": (
        "Jarbas/m2v-256-paraphrase-multilingual-MiniLM-L12-v2",
        "a6b74a06b9f1b5d436ee427b88550c0944b3bcfb",
    ),
    "MATHEEL_MULTIVECTOR_MODEL": (
        "NeuML/pylate-bert-tiny",
        "9cb92954e00ee9b127df35354e4714ed5ff70bee",
    ),
}


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--metadata-out", required=True, type=Path)
    arguments = parser.parse_args()
    paths = {}
    for variable, (repository, revision) in MODELS.items():
        path = snapshot_download(
            repository,
            revision=revision,
            # Native safetensors and tokenizer/config files; omit other export formats.
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            max_workers=2,
        )
        if "\n" in path or "\r" in path:
            raise ValueError("Model path cannot contain a newline.")
        paths[variable] = path
    with arguments.env_file.open("a", encoding="utf-8") as output:
        output.writelines(f"{variable}={path}\n" for variable, path in paths.items())
    arguments.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata_out.write_text(
        json.dumps(
            [{"repository": repository, "revision": revision} for repository, revision in MODELS.values()],
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
