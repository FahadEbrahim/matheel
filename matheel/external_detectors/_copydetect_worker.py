"""Isolated Copydetect worker with a fixed interpreter hash seed."""

import io
import json
import sys
from itertools import combinations
from pathlib import Path

from copydetect import CodeFingerprint, compare_files


def _slices(native):
    if not len(native):
        return []
    return [[int(start), int(end)] for start, end in zip(native[0], native[1])]


def main(input_path, output_path):
    request = json.loads(Path(input_path).read_text(encoding="utf-8"))
    config = request["configuration"]
    noise = config["noise_threshold"]
    window = config["guarantee_threshold"] - noise + 1

    def fingerprint(alias, content, boilerplate=None):
        return CodeFingerprint(
            alias, noise, window, boilerplate=boilerplate, filter=config["filter_code"],
            language=config["language"], fp=io.StringIO(content),
        )

    boilerplate = set()
    for text in request["boilerplate"]:
        boilerplate.update(fingerprint("boilerplate", text).hashes)
    indexed = {alias: fingerprint(alias, text, sorted(boilerplate)) for alias, text in request["sources"]}

    with Path(output_path).open("w", encoding="utf-8") as stream:
        for left, right in combinations(indexed, 2):
            overlap, (forward, reverse), (left_spans, right_spans) = compare_files(indexed[left], indexed[right])
            row = [left, right, float(forward), float(reverse), int(overlap)]
            if config["include_spans"]:
                row.extend((_slices(left_spans), _slices(right_spans)))
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
