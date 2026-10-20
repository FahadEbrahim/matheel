"""Component bundles with a stable positional API for scoring handlers."""

from typing import NamedTuple


class ModelRoutingControls(NamedTuple):
    group: object
    model: object
    status: object
    vector_backend: object
    similarity_group: object
    similarity_function: object
    pooling_group: object
    pooling_method: object
    max_token_length: object
    runtime_device: object
    semantic_weight: object


class CodeMetricControls(NamedTuple):
    group: object
    metric: object
    weight: object
    language: object
    codebleu_group: object
    codebleu_component_weights: object
    crystal_group: object
    crystalbleu_max_order: object
    crystalbleu_trivial_ngram_count: object
    ruby_group: object
    ruby_mode: object
    ruby_graph_timeout_seconds: object
    tsed_group: object
    tsed_costs: object
    codebertscore_group: object
    codebertscore_model: object
    codebertscore_max_length: object


class PreparationControls(NamedTuple):
    selected: object
    preprocess_group: object
    preprocess_mode: object
    chunk_group: object
    chunking_method: object
    chunk_size: object
    chunk_overlap: object
    max_chunks: object
    chunk_aggregation: object
    chunk_language: object
    chunker_options: object


class ScoringControls(NamedTuple):
    metric_preset: object
    features: object
    model: ModelRoutingControls
    levenshtein_group: object
    levenshtein_weight: object
    levenshtein_weights: object
    jaro_group: object
    jaro_winkler_weight: object
    jaro_prefix_weight: object
    winnowing_group: object
    winnowing_weight: object
    winnowing_kgram: object
    winnowing_window: object
    gst_group: object
    gst_weight: object
    gst_min_match_length: object
    lexical_tokenizer: object
    code: CodeMetricControls
    preparation: PreparationControls
    lexical_weights: object

    def handler_inputs(self):
        """Return the stable Pair/Collection/Suite scoring argument order."""
        return [
            self.features,
            self.model.model,
            self.model.vector_backend,
            self.model.similarity_function,
            self.model.pooling_method,
            self.model.max_token_length,
            self.model.runtime_device,
            self.model.semantic_weight,
            self.levenshtein_weight,
            self.jaro_winkler_weight,
            self.winnowing_weight,
            self.gst_weight,
            self.levenshtein_weights,
            self.jaro_prefix_weight,
            self.winnowing_kgram,
            self.winnowing_window,
            self.gst_min_match_length,
            self.lexical_tokenizer,
            self.code.metric,
            self.code.weight,
            self.code.language,
            self.code.codebleu_component_weights,
            self.code.crystalbleu_max_order,
            self.code.crystalbleu_trivial_ngram_count,
            self.code.ruby_mode,
            self.code.ruby_graph_timeout_seconds,
            self.code.tsed_costs,
            self.code.codebertscore_model,
            self.code.codebertscore_max_length,
            self.preparation.selected,
            self.preparation.preprocess_mode,
            self.preparation.chunking_method,
            self.preparation.chunk_size,
            self.preparation.chunk_overlap,
            self.preparation.max_chunks,
            self.preparation.chunk_aggregation,
            self.preparation.chunk_language,
            self.preparation.chunker_options,
        ]
