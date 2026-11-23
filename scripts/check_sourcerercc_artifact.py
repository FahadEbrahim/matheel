"""Qualify the published Linux SourcererCC image on a fixed Java method fixture."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from matheel.clone_discovery import CloneReference, SourceSpan, evaluate_clone_discovery, write_clone_findings
from matheel.external_detectors.common import file_sha256, tree_sha256
from matheel.external_detectors.sourcerercc import parse_report


IMAGE = "ghcr.io/glopix/cloreco-images/sourcerer-cc@sha256:0c93a073ec9f2eddfff012437370125dfcded1a78bfaffe22e56ba4fc7bda1f2"
SEARCH_JAR_SHA256 = "57b75ca65b6c60e4fbdf10ffdeb57416932722b3673e52dfceb3dc41eb755bd2"
PLATFORM = "linux/amd64"
EXPECTED_SPANS = {
    "selected/Left.java": (2, 15),
    "default/Right.java": (3, 16),
    "sample/Other.java": (2, 12),
}
MINIMUM_TOKENS = 20
THRESHOLD = 0.8
LEFT = """\
class Left {
    int calculate(int n) {
        int value = n * n;
        for (int i = 0; i < n; i++) {
            value = value + i * n;
            if (value % 3 == 0) {
                value = value / 3;
            }
            value = value + i * i;
        }
        if (value < 0) {
            value = -value;
        }
        return value;
    }
}
"""
OTHER = """\
class Other {
    String expand(String word) {
        StringBuilder text = new StringBuilder();
        String[] bits = word.split("p");
        for (String bit : bits) {
            text.append(bit.trim());
            text.append("purple");
            text.append(bit.toLowerCase());
            text.append("q").append("z");
        }
        return text.toString().replace("a", "xx").strip();
    }
}
"""


_NATIVE_RUN = """\
app=/cloneDetection/Applications/SourcerCC
cd "$app/tokenizers/block-level"
ln -s /fixture tiny
printf '%s\\n' tiny/selected tiny/default tiny/sample > list.txt
python3 tokenizer.py folderblocks
cp file_block_stats/files-stats-0.stats /output/file-block.stats
cat blocks_tokens/* > "$app/clone-detector/input/dataset/blocks.file"

cd "$app/clone-detector"
sed -i -e 's/^MIN_TOKENS=.*/MIN_TOKENS=20/' -e 's/^IS_SHARDING=.*/IS_SHARDING=false/' sourcerer-cc.properties
printf '1\\n' > search_metadata.txt
run_mode() {
  rm -f nodes_completed.txt
  java -XX:ActiveProcessorCount=2 -Xms64m -Xmx768m \
    -Dproperties.rootDir="$PWD/" -Dproperties.location="$PWD/NODE_1/sourcerer-cc.properties" \
    -Dlog4j.configurationFile="$PWD/NODE_1/log4j2.xml" \
    -jar "$PWD/dist/indexbased.SearchManager.jar" "$1" 8
}
./execute.sh 1
run_mode init
run_mode index
./move-index.sh
./execute.sh 1
run_mode search
cp NODE_1/output8.0/query_1clones_index_WITH_FILTER.txt /output/native-findings.csv
{
  uname -sm
  java -version 2>&1
  python3 -V
  sha256sum dist/indexbased.SearchManager.jar | awk '{print $1}'
} > /output/runtime.txt
"""


def _write_fixture(root):
    files = (LEFT, LEFT.replace("class Left {\n", "class Right {\n\n"), OTHER)
    for name, text in zip(EXPECTED_SPANS, files, strict=True):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def _verified_source_map(stats_path):
    current_project = current_file = None
    seen_files, mapping = set(), {}
    with stats_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, strict=True):
            if len(row) == 9 and row[0].startswith("f"):
                current_project = row[0][1:]
                current_file = row[2].removeprefix("tiny/")
                if current_file not in EXPECTED_SPANS or current_file in seen_files:
                    raise ValueError(f"Unexpected SourcererCC source file {current_file!r}.")
                seen_files.add(current_file)
                continue
            if len(row) != 8 or row[0] != f"b{current_project}" or current_file not in seen_files:
                raise ValueError("Unexpected SourcererCC tokenizer statistics row.")
            lines = tuple(int(value) for value in row[-2:])
            if lines != EXPECTED_SPANS[current_file] or (current_project, row[1]) in mapping:
                raise ValueError(f"Native span disagrees with the known Java method in {current_file!r}.")
            mapping[(current_project, row[1])] = SourceSpan(current_file, *lines)
    if len(mapping) != len(EXPECTED_SPANS) or {span.file_id for span in mapping.values()} != set(EXPECTED_SPANS):
        raise ValueError("Expected one original Java method in each fixture file.")
    return mapping


def verify_recorded_fixture(fixture, output):
    native_runtime = (output / "runtime.txt").read_text(encoding="utf-8").splitlines()
    if not native_runtime or native_runtime[-1] != SEARCH_JAR_SHA256:
        raise ValueError("Unexpected SearchManager JAR digest in the native run.")
    mapping = _verified_source_map(output / "file-block.stats")
    mapping_path = output / "source-map.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("native_project_id", "native_id", "file_id", "start_line", "end_line"))
        writer.writerows((project, block, span.file_id, span.start_line, span.end_line)
                         for (project, block), span in sorted(mapping.items()))

    report = parse_report(output / "native-findings.csv", mapping_path, version=IMAGE,
                          threshold=THRESHOLD, minimum_tokens=MINIMUM_TOKENS,
                          corpus_sha256=tree_sha256(fixture), granularity="function")
    left = SourceSpan("selected/Left.java", *EXPECTED_SPANS["selected/Left.java"])
    right = SourceSpan("default/Right.java", *EXPECTED_SPANS["default/Right.java"])
    if len(report.findings) != 1 or {report.findings[0].left, report.findings[0].right} != {left, right}:
        raise ValueError("Expected a single finding connecting the two original Type-1 fixture methods.")
    reference = CloneReference(left, right, "type1", project_scope="inter_project")
    write_clone_findings(output / "findings.csv", report.findings)
    result = {
        "scope": "authored Java method fixture with one Type-1 reference and one unrelated method",
        "image": IMAGE,
        "image_platform": PLATFORM,
        "native_runtime": native_runtime,
        "configuration": {"minimum_tokens": MINIMUM_TOKENS, "similarity_threshold": THRESHOLD, "sharding": False},
        "source_file_sha256": {name: file_sha256(fixture / name) for name in EXPECTED_SPANS},
        "native_report": report.metadata,
        "evaluation": evaluate_clone_discovery((reference,), report.findings),
        "artifact_sha256": {name: file_sha256(output / name) for name in (
            "file-block.stats", "native-findings.csv", "source-map.csv", "findings.csv",
        )},
    }
    (output / "sourcerercc.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(output_dir):
    with TemporaryDirectory(prefix="matheel-sourcerercc-") as directory:
        root = Path(directory)
        fixture, native = root / "fixture", root / "native"
        _write_fixture(fixture)
        native.mkdir()
        # The isolated container keeps no DAC capabilities and may use a different UID.
        native.chmod(0o777)
        subprocess.run(["docker", "pull", "--platform", PLATFORM, IMAGE], check=True, timeout=420)
        subprocess.run([
            "docker", "run", "--rm", "--platform", PLATFORM, "--pull", "never", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,src={fixture},dst=/fixture,readonly",
            "--mount", f"type=bind,src={native},dst=/output",
            "--entrypoint", "/bin/bash", IMAGE, "-euo", "pipefail", "-c", _NATIVE_RUN,
        ], check=True, timeout=180)
        verify_recorded_fixture(fixture, native)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(native.iterdir()):
            shutil.copyfile(path, output_dir / path.name)
    print(f"Verified one source-mapped Type-1 reference across three SourcererCC method blocks in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for source-free evidence.")
    run(parser.parse_args().output)
