#!/usr/bin/env python3
"""Check whether the Gradio Space Matheel pin is available on PyPI."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
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
    sleep=time.sleep,
    github_output=None,
):
    version = pinned_matheel_version_from_root(root)
    event = str(event_name or "").strip()

    if event == "push":
        if availability_check(version):
            print(f"matheel {version} is available on PyPI.")
            write_github_output(github_output, should_sync_space=True, matheel_version=version)
            return 0
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
            print(f"matheel {version} is available on PyPI.")
            write_github_output(github_output, should_sync_space=True, matheel_version=version)
            return 0

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
    )


if __name__ == "__main__":
    raise SystemExit(main())
