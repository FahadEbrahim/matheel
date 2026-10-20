"""Record verified, source-free local detector identities for CI artifacts."""

import argparse
import json
import os
from pathlib import Path

from matheel.external_detectors import dolos, jplag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    identities = {
        "jplag": jplag.portable_configuration({
            "language": "java", "jar_path": os.environ["MATHEEL_JPLAG_JAR"],
            "java_path": os.environ["MATHEEL_JAVA"],
        }),
        "dolos": dolos.portable_configuration({
            "language": "java", "dolos_path": os.environ["MATHEEL_DOLOS_PREFIX"],
            "node_path": os.environ["MATHEEL_NODE"],
        }),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
