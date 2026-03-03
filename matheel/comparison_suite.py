import json
import re
from pathlib import Path

import pandas as pd

from .similarity import DEFAULT_MODEL_NAME, get_sim_list


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def load_run_configs(config_path):
    raw_data = Path(config_path).read_text(encoding="utf-8")
    return parse_run_configs(raw_data)


def parse_run_configs(raw_data):
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    if isinstance(raw_data, dict):
        runs = raw_data.get("runs", [])
    else:
        runs = raw_data
    if not isinstance(runs, list):
        raise ValueError("Comparison config must be a list or an object with a 'runs' list.")
    return [normalize_run_config(item, index=index) for index, item in enumerate(runs, start=1)]


def normalize_run_config(config, index=1):
    if not isinstance(config, dict):
        raise ValueError("Each comparison run must be a JSON object.")

    options = dict(config.get("options", {}))
    for key, value in config.items():
        if key not in {"run_name", "options"}:
            options[key] = value

    run_name = str(config.get("run_name") or options.pop("run_name", f"run_{index}"))

    if "model" in options and "model_name" not in options:
        options["model_name"] = options.pop("model")
    if "num" in options and "number_results" not in options:
        options["number_results"] = options.pop("num")
    if "ws" in options and "Ws" not in options:
        options["Ws"] = options.pop("ws")
    if "wl" in options and "Wl" not in options:
        options["Wl"] = options.pop("wl")
    if "wj" in options and "Wj" not in options:
        options["Wj"] = options.pop("wj")

    options.setdefault("model_name", DEFAULT_MODEL_NAME)
    options.setdefault("Ws", 0.7)
    options.setdefault("Wl", 0.3)
    options.setdefault("Wj", 0.0)
    options.setdefault("threshold", 0.0)
    options.setdefault("number_results", 10)

    return {"run_name": run_name, "options": options}


def summary_row_from_results(run_name, options, results):
    if results is None or results.empty:
        return {
            "run_name": run_name,
            "pair_count": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "top_file_1": "",
            "top_file_2": "",
            "top_score": 0.0,
            "vector_backend": options.get("vector_backend", "transformer"),
            "code_metric": options.get("code_metric", "none"),
            "chunking_method": options.get("chunking_method", "none"),
        }

    scores = results["similarity_score"].astype(float)
    top_row = results.iloc[0]
    return {
        "run_name": run_name,
        "pair_count": int(len(results)),
        "mean_score": float(scores.mean()),
        "median_score": float(scores.median()),
        "max_score": float(scores.max()),
        "min_score": float(scores.min()),
        "std_score": float(scores.std(ddof=0)),
        "top_file_1": str(top_row["file_name_1"]),
        "top_file_2": str(top_row["file_name_2"]),
        "top_score": float(top_row["similarity_score"]),
        "vector_backend": options.get("vector_backend", "transformer"),
        "code_metric": options.get("code_metric", "none"),
        "chunking_method": options.get("chunking_method", "none"),
    }


def _slugify(value):
    slug = _SLUG_RE.sub("_", value.strip())
    slug = slug.strip("._")
    return slug or "run"


def write_summary(summary, output_path, output_format="csv"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or path.suffix.lstrip(".") or "csv").lower()
    if fmt == "json":
        path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
        return path
    summary.to_csv(path, index=False)
    return path


def write_run_details(run_name, results, details_dir):
    if details_dir is None:
        return None
    target_dir = Path(details_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{_slugify(run_name)}.csv"
    results.to_csv(target_path, index=False)
    return target_path


def run_comparison_suite(
    zipped_file,
    run_configs,
    summary_out=None,
    details_dir=None,
    output_format="csv",
):
    normalized_runs = [normalize_run_config(config, index=index) for index, config in enumerate(run_configs, start=1)]
    summary_rows = []
    result_frames = {}

    for run in normalized_runs:
        run_name = run["run_name"]
        options = dict(run["options"])
        results = get_sim_list(zipped_file, **options)
        result_frames[run_name] = results
        write_run_details(run_name, results, details_dir=details_dir)
        summary_rows.append(summary_row_from_results(run_name, options, results))

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(by=["max_score", "mean_score"], ascending=False, ignore_index=True)

    if summary_out:
        write_summary(summary, summary_out, output_format=output_format)

    return summary, result_frames
