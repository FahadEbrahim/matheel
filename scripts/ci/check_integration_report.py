"""Fail an opt-in integration job when optional dependencies silently skip tests."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check_report(path, expected_tests, allowed_skips, skip_reason=None):
    cases = ET.parse(path).getroot().findall(".//testcase")
    skipped = [case.find("skipped") for case in cases if case.find("skipped") is not None]
    if len(cases) != expected_tests:
        raise ValueError(f"Expected {expected_tests} integration tests, found {len(cases)}.")
    if any(case.find("failure") is not None or case.find("error") is not None for case in cases):
        raise ValueError("Integration report contains a test failure or error.")
    if len(skipped) != allowed_skips:
        raise ValueError(f"Expected {allowed_skips} intentional skips, found {len(skipped)}.")
    if skipped and (skip_reason is None or any(item.get("message") != skip_reason for item in skipped)):
        raise ValueError("Integration report contains an unexpected skip reason.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--tests", type=int, required=True)
    parser.add_argument("--allowed-skips", type=int, default=0)
    parser.add_argument("--skip-reason")
    arguments = parser.parse_args()
    check_report(arguments.report, arguments.tests, arguments.allowed_skips, arguments.skip_reason)


if __name__ == "__main__":
    main()
