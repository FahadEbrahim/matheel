#!/usr/bin/env python3
"""Check the Gradio Space pin and optionally verify its published package source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


MATHEEL_REQUIREMENT_RE = re.compile(r"^matheel(?:\[[^\]]+\])?==([^\s;]+)", re.MULTILINE)


def pinned_matheel_version(requirements_text):
    match = MATHEEL_REQUIREMENT_RE.search(requirements_text)
    if not match:
        raise ValueError("No pinned matheel requirement found in gradio_app/requirements.txt.")
    return match.group(1)


def pinned_matheel_version_from_root(root):
    requirements_path = Path(root) / "gradio_app" / "requirements.txt"
    return pinned_matheel_version(requirements_path.read_text(encoding="utf-8"))


def pypi_release_url(version):
    return f"https://pypi.org/pypi/matheel/{version}/json"


def release_is_available(version, urlopen=urllib.request.urlopen, timeout=10):
    try:
        with urlopen(pypi_release_url(version), timeout=timeout) as response:
            return response.status == 200
    except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError):
        return False


def wheel_package_sources(wheel_bytes):
    """Read Python package sources without importing or extracting wheel contents."""
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        return {
            name: wheel.read(name)
            for name in wheel.namelist()
            if name.startswith("matheel/") and name.endswith(".py")
        }


def release_matches_source(root, version, urlopen=urllib.request.urlopen, timeout=30):
    with urlopen(pypi_release_url(version), timeout=timeout) as response:
        release = json.load(response)
    wheels = sorted(
        (artifact for artifact in release.get("urls", []) if artifact.get("packagetype") == "bdist_wheel"),
        key=lambda artifact: artifact["filename"],
    )
    if not wheels:
        raise ValueError(f"No published wheel found for matheel {version}.")
    with urlopen(wheels[0]["url"], timeout=timeout) as response:
        wheel_bytes = response.read()
    expected_digest = wheels[0].get("digests", {}).get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest):
        raise ValueError(f"No valid SHA256 digest found for matheel {version}'s wheel.")
    if hashlib.sha256(wheel_bytes).hexdigest() != expected_digest.lower():
        raise ValueError(f"SHA256 digest does not match matheel {version}'s published wheel metadata.")
    published = wheel_package_sources(wheel_bytes)
    if not published:
        raise ValueError(f"Published wheel for matheel {version} contains no Python package sources.")
    source_root = Path(root)
    current = {
        path.relative_to(source_root).as_posix(): path.read_bytes()
        for path in (source_root / "matheel").rglob("*.py")
    }
    differences = sorted(
        name for name in current.keys() | published.keys() if current.get(name) != published.get(name)
    )
    if differences:
        names = ", ".join(differences[:5])
        print(f"Package source differs from matheel {version} in {len(differences)} file(s): {names}.")
    return not differences


def write_github_output(path, *, should_sync_space, matheel_version):
    if not path:
        return
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"should_sync_space={str(bool(should_sync_space)).lower()}\n")
        handle.write(f"matheel_version={matheel_version}\n")


def check_pinned_release(
    root,
    *,
    event_name,
    timeout_seconds=600,
    delay_seconds=30,
    availability_check=release_is_available,
    source_check=None,
    sleep=time.sleep,
    github_output=None,
):
    version = pinned_matheel_version_from_root(root)
    event = str(event_name or "").strip()

    def published_release_result():
        print(f"matheel {version} is available on PyPI.")
        matches = True
        if source_check is not None:
            try:
                matches = source_check(root, version)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                print(f"Could not verify the pinned wheel's package source: {exc}", file=sys.stderr)
                matches = False
        write_github_output(github_output, should_sync_space=matches, matheel_version=version)
        if matches:
            return 0
        print(
            "Skipping Space sync until the pinned PyPI release matches the current package source. "
            "Publish a matching Matheel release and update the Space pin before syncing."
        )
        return 0 if event == "push" else 1

    if event == "push":
        if availability_check(version):
            return published_release_result()
        print(
            f"matheel {version} is not available on PyPI yet. "
            "Skipping this push-triggered Space sync; the publish-triggered workflow will sync after release."
        )
        write_github_output(github_output, should_sync_space=False, matheel_version=version)
        return 0

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    delay = max(0.0, float(delay_seconds))
    while True:
        if availability_check(version):
            return published_release_result()

        print(f"matheel {version} is not available on PyPI yet.")
        if time.monotonic() >= deadline:
            write_github_output(github_output, should_sync_space=False, matheel_version=version)
            print(f"Timed out waiting for matheel {version} on PyPI.", file=sys.stderr)
            return 1

        sleep(delay)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Check whether gradio_app/requirements.txt pins an available Matheel release."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of the scripts directory.",
    )
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="GitHub event name. Push events skip successfully when the pin is not published yet.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--delay-seconds", type=float, default=30.0)
    parser.add_argument(
        "--check-source", action="store_true",
        help="Require current matheel Python sources to match the pinned PyPI wheel before syncing.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None,
        help="GitHub output file. Defaults to GITHUB_OUTPUT when present.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return check_pinned_release(
        args.root.resolve(),
        event_name=args.event_name,
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        github_output=args.github_output,
        source_check=release_matches_source if args.check_source else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
