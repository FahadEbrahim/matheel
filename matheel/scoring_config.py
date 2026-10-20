"""Shared, UI-independent configuration for a similarity run.

Only active algorithms validate their specialist controls. A stale, hidden
control must not prevent an unrelated scorer from running.
"""

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping

from .feature_weights import normalize_feature_weights


def numeric_values(raw, count, label, *, allow_zero=False):
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        try:
            parts = list(raw)
        except TypeError as exc:
            raise ValueError(
                f"{label} must contain exactly {count} comma-separated values."
            ) from exc
    if len(parts) != count:
        raise ValueError(f"{label} must contain exactly {count} comma-separated values.")
    try:
        values = tuple(float(part) for part in parts)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain only numeric values.") from exc
    if any(
        not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) for value in values
    ):
        rule = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must contain {rule} values only.")
    return values


def format_numeric_values(values):
    return ",".join(f"{float(value):.4f}".rstrip("0").rstrip(".") or "0" for value in values)


def normalized_component_weights(raw):
    values = numeric_values(raw, 4, "CodeBLEU weights")
    total = sum(values)
    if not math.isfinite(total) or abs(total - 1.0) > 0.02:
        raise ValueError("CodeBLEU weights must sum to 1.")
    return format_numeric_values(value / total for value in values)


def positive_integer(raw, label, minimum=1):
    try:
        numeric = float(raw)
        value = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if numeric != value:
        raise ValueError(f"{label} must be an integer.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _default(raw, fallback):
    return fallback if raw is None or raw == "" else raw


def _text(raw, fallback):
    return str(raw or fallback).strip() or fallback


@dataclass(frozen=True)
class ScoringConfig:
    """Typed input shared by pair, collection, and saved-suite workflows."""

    feature_weights: Mapping[str, float]
    model_name: str = "huggingface/CodeBERTa-small-v1"
    vector_backend: str = "auto"
    similarity_function: str = "cosine"
    pooling_method: str = "mean"
    max_token_length: int = 256
    device: str = "auto"
    preprocess_mode: str = "none"
    chunking_method: str = "none"
    chunk_size: int = 120
    chunk_overlap: int = 0
    max_chunks: int = 0
    chunk_aggregation: str = "mean"
    chunk_language: str = "text"
    chunker_options: str = ""
    code_metric: str = "none"
    code_language: str = "python"
    codebleu_component_weights: Any = "0.25,0.25,0.25,0.25"
    crystalbleu_max_order: int = 4
    crystalbleu_trivial_ngram_count: int = 50
    ruby_mode: str = "auto"
    ruby_graph_timeout_seconds: float = 1.0
    tsed_costs: Any = "1,1,1"
    codebertscore_model: str = "microsoft/codebert-base"
    codebertscore_max_length: int = 0
    levenshtein_weights: Any = "1,1,1"
    jaro_winkler_prefix_weight: float = 0.1
    winnowing_kgram: int = 5
    winnowing_window: int = 4
    gst_min_match_length: int = 5
    lexical_tokenizer: str = "raw"

    @classmethod
    def from_mapping(cls, values, **overrides):
        """Ignore workflow-only fields while preserving the shared API names."""
        source = {**values, **overrides}
        if "runtime_device" in source and "device" not in source:
            source["device"] = source["runtime_device"]
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in source.items() if key in names})

    def as_options(self):
        weights = {
            name: weight
            for name, weight in normalize_feature_weights(self.feature_weights).items()
            if weight > 0
        }
        if not weights:
            raise ValueError("At least one selected metric must have a positive weight.")
        code_metric = (
            _text(self.code_metric, "none").lower() if weights.get("code_metric") else "none"
        )
        if weights.get("code_metric") and code_metric == "none":
            raise ValueError("Choose a code metric when its weight is positive.")
        chunking_method = _text(self.chunking_method, "none")
        chunking_active = chunking_method != "none"
        options = {
            "feature_weights": weights,
            "model_name": _text(self.model_name, ScoringConfig.model_name),
            "vector_backend": _text(self.vector_backend, "auto"),
            "similarity_function": _text(self.similarity_function, "cosine"),
            "pooling_method": _text(self.pooling_method, "mean"),
            "max_token_length": positive_integer(
                _default(self.max_token_length, 256), "Maximum token length", 8
            )
            if weights.get("semantic")
            else 256,
            "device": _text(self.device, "auto"),
            "preprocess_mode": _text(self.preprocess_mode, "none"),
            "chunking_method": chunking_method,
            "chunk_size": positive_integer(_default(self.chunk_size, 120), "Chunk size", 10)
            if chunking_active
            else 120,
            "chunk_overlap": positive_integer(_default(self.chunk_overlap, 0), "Chunk overlap", 0)
            if chunking_active
            else 0,
            "max_chunks": positive_integer(_default(self.max_chunks, 0), "Maximum chunks", 0)
            if chunking_active
            else 0,
            "chunk_aggregation": _text(self.chunk_aggregation, "mean"),
            "chunk_language": _text(self.chunk_language, "text"),
            "chunker_options": str(self.chunker_options or "").strip(),
            "code_metric": code_metric,
            "code_metric_weight": weights.get("code_metric", 0.0),
            "code_language": _text(self.code_language, "python"),
            "codebleu_component_weights": "0.25,0.25,0.25,0.25",
            "crystalbleu_max_order": 4,
            "crystalbleu_trivial_ngram_count": 50,
            "levenshtein_weights": (1, 1, 1),
            "jaro_winkler_prefix_weight": 0.1,
            "winnowing_kgram": 5,
            "winnowing_window": 4,
            "gst_min_match_length": 5,
            "lexical_tokenizer": _text(self.lexical_tokenizer, "raw"),
        }
        if weights.get("levenshtein"):
            values = numeric_values(self.levenshtein_weights or "1,1,1", 3, "Levenshtein weights")
            options["levenshtein_weights"] = tuple(max(1, int(round(value))) for value in values)
        if weights.get("jaro_winkler"):
            try:
                prefix = float(_default(self.jaro_winkler_prefix_weight, 0.1))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Jaro-Winkler prefix weight must be between 0 and 0.25.") from exc
            if not math.isfinite(prefix) or not 0 <= prefix <= 0.25:
                raise ValueError("Jaro-Winkler prefix weight must be between 0 and 0.25.")
            options["jaro_winkler_prefix_weight"] = prefix
        if weights.get("winnowing"):
            options["winnowing_kgram"] = positive_integer(
                self.winnowing_kgram, "Winnowing k-gram size"
            )
            options["winnowing_window"] = positive_integer(
                self.winnowing_window, "Winnowing window size"
            )
        if weights.get("gst"):
            options["gst_min_match_length"] = positive_integer(
                self.gst_min_match_length, "GST minimum match length"
            )
        if code_metric.startswith("codebleu"):
            options["codebleu_component_weights"] = normalized_component_weights(
                self.codebleu_component_weights
            )
        if code_metric == "crystalbleu":
            options["crystalbleu_max_order"] = positive_integer(
                self.crystalbleu_max_order, "CrystalBLEU maximum order"
            )
            options["crystalbleu_trivial_ngram_count"] = positive_integer(
                self.crystalbleu_trivial_ngram_count, "CrystalBLEU trivial n-gram count", 0
            )
        if code_metric == "ruby":
            mode = _text(self.ruby_mode, "auto").lower()
            if mode not in ("auto", "graph", "tree", "string", "ngram"):
                raise ValueError("RUBY mode must be one of: auto, graph, tree, string, ngram.")
            try:
                timeout = float(self.ruby_graph_timeout_seconds)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("RUBY graph timeout must be at least 0.1 seconds.") from exc
            if not math.isfinite(timeout) or timeout < 0.1:
                raise ValueError("RUBY graph timeout must be at least 0.1 seconds.")
            options.update(ruby_mode=mode, ruby_graph_timeout_seconds=timeout)
        if code_metric == "tsed":
            costs = numeric_values(self.tsed_costs, 3, "TSED costs", allow_zero=True)
            options.update(zip(("tsed_delete_cost", "tsed_insert_cost", "tsed_rename_cost"), costs))
        if code_metric == "codebertscore":
            options["codebertscore_model"] = _text(
                self.codebertscore_model, "microsoft/codebert-base"
            )
            options["codebertscore_max_length"] = positive_integer(
                _default(self.codebertscore_max_length, 0), "CodeBERTScore max length", 0
            )
        return options
