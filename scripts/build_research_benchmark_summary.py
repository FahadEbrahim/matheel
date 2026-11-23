#!/usr/bin/env python3
"""Build or check the Gradio index of published 0.8.0 benchmark evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path("gradio_app/benchmark_data/research-080/summary.json")
AUTHORSHIP_PATH = Path("benchmarks/authorship-v1/results.json")
EVIDENCE_DIR = Path("benchmarks/research-080")
DETECTOR_REPORT_PATH = EVIDENCE_DIR / "detectors" / "results.json"
RELEASE = "0.8.0"
_METHOD_LABELS = {
    "class_prior": "Training class prior",
    "char_tfidf_logistic": "Character TF-IDF + logistic regression",
    "character_tfidf_cosine": "Character TF-IDF cosine",
    "dolos": "Dolos",
    "jplag": "JPlag",
    "levenshtein": "Levenshtein",
    "nicad": "NiCad",
    "pmd-cpd": "PMD CPD",
    "sourcerercc": "SourcererCC",
    "winnowing": "Winnowing",
}
_DETECTOR_DATASETS = {
    "nicad-java-smoke": "Java fixture",
    "pmd-cpd-java-smoke": "Java fixture",
    "bigcloneeval-reference-replay": "BigCloneEval reference replay",
    "sourcerercc-report-import": "SourcererCC report fixture",
    "moss-similcode-replay": "SimilCode",
    "copydetect-irplag-java": "IRPlag Java",
}
_PAIR_METRICS = (("f1", "Binary F1"),)
_RETRIEVAL_METRICS = (
    ("mean_average_precision_at_r", "MAP@R"),
    ("mean_reciprocal_rank", "MRR"),
    ("recall_at_k", "Recall@{k}"),
)
_AUTHORSHIP_METRICS = (
    ("average_precision", "Average precision"),
    ("auroc", "AUROC"),
    ("macro_f1", "Macro F1"),
    ("recall", "Recall"),
    ("false_positive_rate", "False-positive rate"),
)


def _rows(dataset, scope, method, metrics, names, *, evaluated, unit, matched=None, tool_version=None):
    if type(evaluated) is not int or evaluated <= 0:
        return
    method = _METHOD_LABELS.get(method, method)
    if tool_version:
        method = f"{method} {tool_version}"
    for key, label in names:
        value = metrics.get(key)
        if value is None:
            continue
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Invalid {label} result for {dataset}.")
        cutoff = metrics.get("k")
        if "{k}" in label and (type(cutoff) is not int or cutoff <= 0):
            raise ValueError(f"Missing retrieval cutoff for {dataset}.")
        row = {
            "dataset": dataset,
            "scope": scope,
            "method": method,
            "metric": label.format(k=cutoff),
            "value": value,
            "evaluated": evaluated,
            "coverage_unit": unit,
        }
        if matched is not None:
            row["matched"] = matched
        yield row


def _authorship_rows(report):
    if report.get("benchmark_id") != "Matheel-AIGCodeSet-v1" or report.get("schema_version") != 1:
        raise ValueError("Unsupported public authorship report.")
    for model in report["results"]:
        entries = (
            ("Scratch test", model["test"], _AUTHORSHIP_METRICS),
            ("Runtime repair test", model["repair_slices"]["runtime_repair"], _AUTHORSHIP_METRICS[-2:]),
            ("Wrong-answer repair test", model["repair_slices"]["wrong_answer_repair"], _AUTHORSHIP_METRICS[-2:]),
        )
        for scope, metrics, names in entries:
            suffix = f"{model['target_false_positive_rate']:.0%} validation FPR ceiling"
            yield from _rows(
                "AIGCodeSet", f"{scope} · {suffix}", model["model"], metrics, names,
                evaluated=metrics["count"], unit="snippets",
            )


def _similarity_rows(report):
    if report.get("benchmark_schema_version") != 1:
        raise ValueError("Unsupported public similarity report.")
    dataset = report["dataset"]
    adapter = dataset["adapter"]
    split = dataset.get("split", "test")
    if adapter == "similcode_pair":
        name = "SimilCode"
        langs = {"java": "Java", "csharp": "C#"}
        languages = " / ".join(langs[lang] for lang in dataset["languages"])
        scope = f"{languages} · {dataset['label_view'].replace('_', ' ')}"
    elif adapter == "codexglue_bcb_pair":
        name = "CodeXGLUE BigCloneBench"
        coverage = "full split" if dataset["official_split_dimensions"] else "subset"
        scope = f"{split.capitalize()} binary pairs · {coverage}"
    elif adapter == "poj104_retrieval":
        name = "POJ-104"
        scope = f"{split.capitalize()} · {'full partition' if dataset['official_split_dimensions'] else 'subset'}"
    elif adapter == "xlcost_retrieval":
        name = "XLCoST"
        direction = " → ".join(dataset[key].capitalize() for key in ("source_language", "target_language"))
        scope = f"{direction} · {split} program retrieval"
    else:
        raise ValueError(f"Unsupported similarity benchmark adapter: {adapter}.")
    for run in report["results"]:
        metrics = run["metrics"]
        tool_version = run.get("configuration", {}).get("version")
        if report["task_family"] == "pair":
            evaluated = metrics["positive_count"] + metrics["negative_count"]
            threshold = metrics.get("threshold")
            pair_scope = f"{scope} · threshold {threshold:g}" if threshold is not None else scope
            yield from _rows(
                name, pair_scope, run["name"], metrics, _PAIR_METRICS,
                evaluated=evaluated, unit="pairs", tool_version=tool_version,
            )
        else:
            names = _RETRIEVAL_METRICS[:1] if adapter == "poj104_retrieval" else _RETRIEVAL_METRICS[1:]
            yield from _rows(
                name, scope, run["name"], metrics, names,
                evaluated=metrics["query_count"], unit="queries", tool_version=tool_version,
            )


def _detector_rows(report):
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported public detector report.")
    for case in report["cases"]:
        if case["status"] != "verified":
            continue
        identifier = case["id"]
        dataset = case.get("dataset") or _DETECTOR_DATASETS[identifier]
        scope, tool, metrics = case["scope"], case["detector"], case["metrics"]
        provenance = case.get("provenance", {})
        tool_version = provenance.get("version") or provenance.get("configuration", {}).get("version")
        if isinstance(tool_version, str) and "@sha256:" in tool_version:
            tool_version = None
        if "recall" in metrics and "eligible" in metrics:
            legacy = metrics.get("official")
            if legacy is not None:
                legacy_scope = f"{scope} · legacy matcher"
                yield from _rows(
                    dataset, legacy_scope, tool, legacy, (("recall", "Clone recall"),),
                    evaluated=legacy["eligible"], unit="reference pairs", matched=legacy["matched"],
                )
                scope = f"{scope} · fractional matcher"
                if identifier == "bigcloneeval-reference-replay":
                    tool = "Matheel span matcher"
            yield from _rows(
                dataset, scope, tool, metrics, (("recall", "Clone recall"),),
                evaluated=metrics["eligible"], unit="reference pairs", matched=metrics["matched"],
                tool_version=tool_version,
            )
            for stratum in metrics.get("strata", ()):
                if stratum["project_scope"] != "all" or stratum["clone_type"] == "all":
                    continue
                clone_type = stratum["clone_type"].replace("_", " ").replace("type", "Type ")
                if stratum["clone_type"] == "weak_type3_type4":
                    clone_type = "Weak Type 3 / Type 4"
                yield from _rows(
                    dataset, f"{scope} · {clone_type}", tool, stratum, (("recall", "Clone recall"),),
                    evaluated=stratum["eligible"], unit="reference pairs", matched=stratum["matched"],
                    tool_version=tool_version,
                )
        elif "expected_pairs" in metrics:
            expected, reported = metrics["expected_pairs"], metrics["reported_pairs"]
            if type(expected) is not int or type(reported) is not int or not 0 <= reported <= expected:
                raise ValueError(f"Invalid detector report coverage for {dataset}.")
            if expected > 0:
                yield from _rows(
                    dataset, scope, tool, {"coverage": reported / expected}, (("coverage", "Report coverage"),),
                    evaluated=expected, unit="requested pairs",
                )
        elif "pair_count" in metrics:
            comparisons = provenance.get("comparisons") or [{"name": tool, "metrics": metrics}]
            for comparison in comparisons:
                measured = comparison["metrics"]
                threshold = measured.get("threshold")
                comparison_scope = " · ".join(filter(None, (scope, comparison.get("scope"))))
                if threshold is not None:
                    comparison_scope += f" · threshold {threshold:g}"
                yield from _rows(
                    dataset, comparison_scope, comparison["name"], measured,
                    (("average_precision", "Average precision"), *_PAIR_METRICS),
                    evaluated=measured["pair_count"], unit="pairs",
                )
        elif "reported_pairs" in metrics:
            label = "Imported findings" if identifier == "sourcerercc-report-import" else "Reported clone pairs"
            yield from _rows(
                dataset, scope, tool, metrics, (("reported_pairs", label),),
                evaluated=metrics["reported_pairs"], unit="reported pairs",
                tool_version=tool_version,
            )


def build_summary(root=ROOT):
    """Return a deterministic projection of the release's available result reports."""
    root = Path(root).resolve()
    authorship = root / AUTHORSHIP_PATH
    similarity = root / EVIDENCE_DIR / "similarity"
    inputs = [
        (authorship, _authorship_rows),
        *((path, _similarity_rows) for path in similarity.rglob("results.json")),
        (root / DETECTOR_REPORT_PATH, _detector_rows),
    ]
    sources, rows = [], []
    for path, project in sorted(inputs, key=lambda entry: entry[0].as_posix()):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("Benchmark report must stay inside the repository.")
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        selected = [{**row, "source": relative} for row in project(json.loads(data))]
        sources.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(selected)})
        rows.extend(selected)
    if not rows:
        raise ValueError("No evaluated metrics found for the Gradio benchmark index.")
    return {"schema_version": 1, "release": RELEASE, "rows": rows, "sources": sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Verify the published index without writing it.")
    args = parser.parse_args(argv)
    output = args.root / OUTPUT_PATH
    contents = json.dumps(build_summary(args.root), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not output.is_file() or output.read_bytes() != contents.encode("utf-8"):
            parser.exit(1, f"{output} is missing or stale. Run build_research_benchmark_summary.py.\n")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".summary-") as temporary:
            pending = Path(temporary) / output.name
            pending.write_bytes(contents.encode("utf-8"))
            pending.replace(output)
    print(output)


if __name__ == "__main__":
    main()
