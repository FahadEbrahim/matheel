#!/usr/bin/env python3
"""Bounded, offline GST before/after profiles; optional local real pair inputs."""

import argparse
import ast
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter

from matheel.datasets import load_code_texts, load_pair_dataset
from matheel.similarity import _ranges_overlap, gst_similarity_from_tokens, tokenize_for_lexical_matching


def _baseline(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:matheel/similarity.py"], text=True)
    function = next(item for item in ast.parse(source).body
                    if isinstance(item, ast.FunctionDef) and item.name == "gst_similarity_from_tokens")
    namespace = {"_ranges_overlap": _ranges_overlap}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "baseline_gst.py", "exec"), namespace)
    return namespace["gst_similarity_from_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="v0.7.0", help="Trusted local Git ref containing the old GST implementation.")
    parser.add_argument("--pair-dataset", type=Path, help="Optional normalized local pair dataset; no downloads.")
    parser.add_argument("--seconds", type=float, default=5, help="Per-case worker deadline, including Python startup.")
    parser.add_argument("--worker", choices=["baseline", "current"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive.")
    if args.worker:
        case = json.load(sys.stdin)
        function = _baseline(args.baseline_ref) if args.worker == "baseline" else gst_similarity_from_tokens
        profile = cProfile.Profile()
        started = perf_counter()
        score = profile.runcall(function, case["left"], case["right"])
        print(json.dumps({"score": score, "seconds": perf_counter() - started,
                          "function_calls": pstats.Stats(profile).total_calls}))
        return
    cases = [{"name": "punctuation_750", "left": [token for i in range(150) for token in ("(", f"left{i}", ")", ";", "}")],
              "right": [token for i in range(150) for token in ("(", f"right{i}", ")", ";", "}")]},
             {"name": "identical_1200", "left": [str(i) for i in range(1200)], "right": [str(i) for i in range(1200)]}]
    if args.pair_dataset:
        dataset = load_pair_dataset(args.pair_dataset)
        texts = load_code_texts(dataset)
        pairs = sorted(dataset.pairs.to_dict("records"), key=lambda row: len(texts[row["left_id"]]) + len(texts[row["right_id"]]))
        for position in sorted({0, len(pairs) // 2, len(pairs) - 1}) if pairs else ():
            row = pairs[position]
            cases.append({"name": f"local_pair_{position}", "left_id": row["left_id"], "right_id": row["right_id"],
                          "left": tokenize_for_lexical_matching(texts[row["left_id"]]),
                          "right": tokenize_for_lexical_matching(texts[row["right_id"]])})
    for case in cases:
        encoded = json.dumps(case).encode()
        result = {"case": case["name"], "token_counts": [len(case["left"]), len(case["right"])],
                  "input_sha256": hashlib.sha256(encoded).hexdigest(), "baseline_ref": args.baseline_ref}
        for implementation in ("baseline", "current"):
            try:
                worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", implementation,
                                         "--baseline-ref", args.baseline_ref], input=encoded,
                                        capture_output=True, check=True, timeout=args.seconds)
                result[implementation] = json.loads(worker.stdout)
            except subprocess.TimeoutExpired:
                result[implementation] = {"status": "timeout", "deadline_seconds": args.seconds}
        if "score" in result["baseline"] and "score" in result["current"]:
            assert result["baseline"]["score"] == result["current"]["score"], "GST parity regression"
            result["exact_parity"] = True
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
