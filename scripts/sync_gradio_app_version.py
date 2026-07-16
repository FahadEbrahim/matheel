#!/usr/bin/env python3
"""Sync release-facing Matheel version references from pyproject.toml."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


GRADIO_README_VERSION_RE = re.compile(r"`matheel==[^`]+`")
PYPI_BADGE_RE = re.compile(
    r"https://img\.shields\.io/pypi/v/matheel\.svg\?cacheSeconds=3600"
    r"(?:&release=v?[^)\s]+)?"
)
REQUIREMENTS_VERSION_RE = re.compile(r"(?m)^matheel\[all\]==[^\s]+$")
PROJECT_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([^"]+)"\s*$')
NOTEBOOK_REF_RE = re.compile(r'(MATHEEL_REF\s*=\s*\\"v)([^\\"]+)(\\")')
RELEASE_NOTEBOOKS = (
    "01_core_workflows.ipynb",
    "02_datasets_and_reproducibility.ipynb",
    "03_custom_algorithms.ipynb",
    "04_gradio_app.ipynb",
    "05_visualization_and_leaderboard.ipynb",
)


def project_version(root):
    pyproject_path = root / "pyproject.toml"
    if tomllib is not None:
        with pyproject_path.open("rb") as pyproject:
            return tomllib.load(pyproject)["project"]["version"]

    project_lines = []
    in_project_section = False
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        if line == "[project]":
            in_project_section = True
            continue
        if in_project_section and line.startswith("["):
            break
        if in_project_section:
            project_lines.append(line)

    match = PROJECT_VERSION_RE.search("\n".join(project_lines))
    if not match:
        raise ValueError("Could not find project.version in pyproject.toml.")
    return match.group(1)


def replace_single(path, pattern, replacement):
    original = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(replacement, original, count=1)
    if count != 1:
        raise ValueError(f"Expected exactly one Matheel version reference in {path}.")
    return original, updated


def planned_updates(root):
    version = project_version(root)
    targets = [
        (
            root / "README.md",
            PYPI_BADGE_RE,
            "https://img.shields.io/pypi/v/matheel.svg"
            f"?cacheSeconds=3600&release=v{version}",
        ),
        (
            root / "gradio_app" / "README.md",
            GRADIO_README_VERSION_RE,
            f"`matheel=={version}`",
        ),
        (
            root / "gradio_app" / "requirements.txt",
            REQUIREMENTS_VERSION_RE,
            f"matheel[all]=={version}",
        ),
    ]
    targets.extend(
        (
            root / "examples" / "notebooks" / notebook_name,
            NOTEBOOK_REF_RE,
            rf"\g<1>{version}\g<3>",
        )
        for notebook_name in RELEASE_NOTEBOOKS
    )

    updates = []
    for path, pattern, replacement in targets:
        original, updated = replace_single(path, pattern, replacement)
        if original != updated:
            updates.append((path, updated))
    return updates


def sync_gradio_app_version(root, check=False):
    updates = planned_updates(root)
    if check:
        if updates:
            for path, _ in updates:
                print(f"{path.relative_to(root)} is not synced with pyproject.toml.", file=sys.stderr)
            return 1
        print("Release version references are synced.")
        return 0

    if not updates:
        print("Release version references are already synced.")
        return 0

    for path, updated in updates:
        path.write_text(updated, encoding="utf-8")
        print(f"Updated {path.relative_to(root)}.")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Sync release-facing Matheel version references from pyproject.toml."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if release-facing version references are stale.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of the scripts directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return sync_gradio_app_version(args.root.resolve(), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
