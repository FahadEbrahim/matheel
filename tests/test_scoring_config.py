import math

import pytest

from matheel.scoring_config import ScoringConfig
from matheel.similarity import DEFAULT_MODEL_NAME


def test_shared_configuration_preserves_the_library_default_model():
    assert ScoringConfig(feature_weights={"lcs": 1}).as_options()["model_name"] == DEFAULT_MODEL_NAME


def test_shared_configuration_normalizes_relative_weights_without_overflow():
    options = ScoringConfig(
        feature_weights={"lcs": 1e308, "levenshtein": 1e308},
        levenshtein_weights="1,2,3",
        device="cpu",
    ).as_options()

    assert options["feature_weights"] == {"lcs": 0.5, "levenshtein": 0.5}
    assert options["levenshtein_weights"] == (1, 2, 3)
    assert options["device"] == "cpu"


def test_hidden_specialist_options_do_not_block_an_unrelated_metric():
    options = ScoringConfig(
        feature_weights={"lcs": 1.0},
        levenshtein_weights="invalid hidden control",
        winnowing_kgram=-1,
        gst_min_match_length=0,
        code_metric="ruby",
        ruby_graph_timeout_seconds=math.nan,
        codebleu_component_weights="invalid hidden control",
        codebertscore_max_length=-5,
        max_token_length=-1,
        chunk_size="invalid hidden control",
    ).as_options()

    assert options["code_metric"] == "none"
    assert options["feature_weights"] == {"lcs": 1.0}
    assert options["winnowing_kgram"] == 5


@pytest.mark.parametrize("invalid", [0, -1, 1.5, float("inf"), float("nan")])
def test_active_integer_controls_reject_invalid_and_fractional_values(invalid):
    with pytest.raises(ValueError, match="Winnowing k-gram size"):
        ScoringConfig(feature_weights={"winnowing": 1}, winnowing_kgram=invalid).as_options()


def test_zero_valued_controls_are_preserved():
    options = ScoringConfig(
        feature_weights={"code_metric": 1, "jaro_winkler": 1},
        code_metric="crystalbleu",
        crystalbleu_trivial_ngram_count=0,
        jaro_winkler_prefix_weight=0,
    ).as_options()

    assert options["crystalbleu_trivial_ngram_count"] == 0
    assert options["jaro_winkler_prefix_weight"] == 0


@pytest.mark.parametrize("timeout", [0, -1, "bad", float("inf"), float("nan")])
def test_active_ruby_timeout_is_validated(timeout):
    with pytest.raises(ValueError, match="at least 0.1 seconds"):
        ScoringConfig(
            feature_weights={"code_metric": 1},
            code_metric="ruby",
            ruby_graph_timeout_seconds=timeout,
        ).as_options()


def test_tsed_cost_order_and_zero_costs_survive_configuration():
    options = ScoringConfig(
        feature_weights={"code_metric": 1},
        code_metric="tsed",
        tsed_costs="0,2,3",
    ).as_options()

    assert (
        options["tsed_delete_cost"],
        options["tsed_insert_cost"],
        options["tsed_rename_cost"],
    ) == (0, 2, 3)


def test_mapping_translation_ignores_workflow_fields_and_maps_device():
    options = ScoringConfig.from_mapping(
        {
            "feature_weights": {"lcs": 1},
            "runtime_device": "cpu",
            "run_name": "example",
            "threshold": 0.3,
            "progress": object(),
        }
    ).as_options()

    assert options["device"] == "cpu"
    assert "run_name" not in options
    assert "threshold" not in options


def test_nonpositive_and_unknown_feature_selections_are_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        ScoringConfig(feature_weights={"lcs": 0}).as_options()
    with pytest.raises(ValueError, match="Unsupported feature weight"):
        ScoringConfig(feature_weights={"unknown": 1}).as_options()
