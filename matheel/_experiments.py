"""Shared artifact and scoring plumbing for diagnostic experiments."""

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd

from .comparison_suite import parse_run_configs
from .datasets import PairDataset
from .evaluation import score_pair_dataset
from .reproducibility import collect_reproducibility_snapshot


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_id(value):
    return text_hash(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")))


def experiment_runs(runs):
    normalized = parse_run_configs(runs)
    if not normalized:
        raise ValueError("At least one declared scoring run is required.")
    # The saved configuration must describe executable, reproducible options.
    json.dumps(normalized, sort_keys=True, allow_nan=False)
    return normalized


def scoring_options(run):
    options = dict(run["options"])
    options.pop("threshold", None)
    options.pop("number_results", None)
    return options


def score_experiment_pairs(pairs, run):
    """Use the existing batch scorer without creating classification labels.

    This temporary PairDataset is only an in-process transport for the public
    score_pair_dataset API. No evaluation metrics, labels, or registered data
    are created. Semantic representations are prepared once for the whole run.
    """
    texts = {}
    records = []
    for pair_id, left, right in pairs:
        left_id, right_id = text_hash(left), text_hash(right)
        texts[left_id], texts[right_id] = left, right
        records.append({"pair_id": pair_id, "left_id": left_id, "right_id": right_id})
    with tempfile.TemporaryDirectory(prefix="matheel-diagnostic-") as temporary:
        root = Path(temporary)
        files = []
        for file_id, text in sorted(texts.items()):
            name = f"{file_id}.txt"
            (root / name).write_text(text, encoding="utf-8")
            files.append({"file_id": file_id, "file_path": name})
        dataset = PairDataset(
            root=root, files=pd.DataFrame(files), pairs=pd.DataFrame(records),
            metadata={"purpose": "unlabelled diagnostic experiment"},
        )
        options = scoring_options(run)
        algorithm = options.pop("algorithm_path", options.pop("algorithm", None))
        algorithm_options = options.pop("algorithm_options", None)
        scored = score_pair_dataset(
            dataset, similarity_options=options, algorithm=algorithm,
            algorithm_options=algorithm_options,
        )
    return dict(zip(scored["pair_id"], scored["similarity_score"]))


def experiment_metadata(kind, runs, inputs):
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    identity = {"experiment_type": kind, "runs": runs, "inputs": inputs,
                "implementation": implementation}
    return {
        "schema_version": 1, "experiment_type": kind,
        "experiment_id": content_id(identity), "diagnostic_only": True,
        "runs": runs, "implementation": implementation,
        "reproducibility": collect_reproducibility_snapshot(run_configs=runs),
    }


def write_experiment_artifacts(report, output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = report["experiment_type"]
    report_path = destination / f"{stem}.json"
    scores_path = destination / f"{stem}_scores.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    pd.DataFrame(report.get("scores", [])).to_csv(scores_path, index=False)
    return {"json": str(report_path), "scores_csv": str(scores_path)}
