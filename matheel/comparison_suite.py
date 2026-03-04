import json
import re
from pathlib import Path

import pandas as pd

from .feature_weights import default_feature_weights, parse_feature_weights
from .similarity import DEFAULT_MODEL_NAME, get_sim_list


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_SCORE_DECIMALS = 4
_SCORE_FIELDS = (
    "mean_score",
    "median_score",
    "max_score",
    "min_score",
    "std_score",
    "top_score",
    "similarity_score",
)


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
    _normalize_run_feature_weights(options)

    options.setdefault("model_name", DEFAULT_MODEL_NAME)
    options.setdefault("threshold", 0.0)
    options.setdefault("number_results", 10)

    return {"run_name": run_name, "options": options}


def _normalize_run_feature_weights(options):
    legacy_weight_keys = [key for key in ("ws", "wl", "wj", "Ws", "Wl", "Wj") if key in options]
    if legacy_weight_keys:
        keys = ", ".join(sorted(legacy_weight_keys))
        raise ValueError(
            f"Legacy weight keys are no longer supported ({keys}). Use feature_weights instead."
        )

    resolved = parse_feature_weights(options.get("feature_weights"))

    if resolved:
        options["feature_weights"] = resolved
        return

    options["feature_weights"] = default_feature_weights()


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
            "vector_backend": options.get("vector_backend", "auto"),
            "code_metric": options.get("code_metric", "none"),
            "chunking_method": options.get("chunking_method", "none"),
        }

    scores = results["similarity_score"].astype(float)
    top_row = results.iloc[0]
    return {
        "run_name": run_name,
        "pair_count": int(len(results)),
        "mean_score": round(float(scores.mean()), _SCORE_DECIMALS),
        "median_score": round(float(scores.median()), _SCORE_DECIMALS),
        "max_score": round(float(scores.max()), _SCORE_DECIMALS),
        "min_score": round(float(scores.min()), _SCORE_DECIMALS),
        "std_score": round(float(scores.std(ddof=0)), _SCORE_DECIMALS),
        "top_file_1": str(top_row["file_name_1"]),
        "top_file_2": str(top_row["file_name_2"]),
        "top_score": round(float(top_row["similarity_score"]), _SCORE_DECIMALS),
        "vector_backend": options.get("vector_backend", "auto"),
        "code_metric": options.get("code_metric", "none"),
        "chunking_method": options.get("chunking_method", "none"),
    }


def _slugify(value):
    slug = _SLUG_RE.sub("_", value.strip())
    slug = slug.strip("._")
    return slug or "run"


def _rounded_score_frame(frame):
    rounded = frame.copy()
    for column in _SCORE_FIELDS:
        if column in rounded.columns:
            rounded[column] = rounded[column].astype(float).round(_SCORE_DECIMALS)
    return rounded


def write_summary(summary, output_path, output_format="csv"):
    summary = _rounded_score_frame(summary)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or path.suffix.lstrip(".") or "csv").lower()
    if fmt == "json":
        path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
        return path
    summary.to_csv(path, index=False, float_format=f"%.{_SCORE_DECIMALS}f")
    return path


def write_run_details(run_name, results, details_dir):
    if details_dir is None:
        return None
    results = _rounded_score_frame(results)
    target_dir = Path(details_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{_slugify(run_name)}.csv"
    results.to_csv(target_path, index=False, float_format=f"%.{_SCORE_DECIMALS}f")
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
        results = _rounded_score_frame(get_sim_list(zipped_file, **options))
        result_frames[run_name] = results
        write_run_details(run_name, results, details_dir=details_dir)
        summary_rows.append(summary_row_from_results(run_name, options, results))

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(by=["max_score", "mean_score"], ascending=False, ignore_index=True)

    if summary_out:
        write_summary(summary, summary_out, output_format=output_format)

    return summary, result_frames
