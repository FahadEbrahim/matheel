from pathlib import Path

import pytest

from matheel.algorithms import (
    PairAlgorithm,
    normalize_algorithm_options,
    pair_algorithm_metadata,
    prepare_algorithm_dataset,
    resolve_pair_algorithm,
    score_pair_with_algorithm,
    score_source_pairs_with_algorithm,
)


def test_resolve_pair_algorithm_from_callable():
    def score_pair(code_a, code_b):
        return 1.0 if code_a == code_b else 0.0

    algorithm = resolve_pair_algorithm(score_pair)

    assert isinstance(algorithm, PairAlgorithm)
    assert algorithm.name == "score_pair"
    assert score_pair_with_algorithm("x", "x", algorithm) == pytest.approx(1.0)


def test_score_pair_with_algorithm_passes_options_context_and_row():
    def score_pair(code_a, code_b, dataset_context=None, row=None, multiplier=1.0, **kwargs):
        _ = kwargs
        base = 1.0 if code_a == code_b else 0.5
        return base * float(multiplier) + float(dataset_context["offset"]) + float(row["bonus"])

    algorithm = resolve_pair_algorithm(score_pair)
    score = score_pair_with_algorithm(
        "a",
        "a",
        algorithm,
        dataset_context={"offset": 0.25},
        row={"bonus": 0.5},
        algorithm_options={"multiplier": 2.0},
    )

    assert score == pytest.approx(2.75)


def test_algorithm_options_are_validated():
    with pytest.raises(ValueError, match="mapping"):
        normalize_algorithm_options(["bias=0.1"])

    with pytest.raises(ValueError, match="JSON-serializable"):
        normalize_algorithm_options({"bad": object()})

    def score_pair(code_a, code_b):
        _ = (code_a, code_b)
        return 0.0

    with pytest.raises(ValueError, match="reserved"):
        score_pair_with_algorithm("a", "b", score_pair, algorithm_options={"code_a": "override"})


def test_resolve_pair_algorithm_from_module_path_with_prepare_hook(tmp_path):
    module_path = tmp_path / "custom_algorithm.py"
    module_path.write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, bonus=0.0, **kwargs):",
                "    return {'bonus': float(bonus), 'n': len(dataset)}",
                "",
                "def score_pair(code_a, code_b, dataset_context=None, **kwargs):",
                "    return (1.0 if code_a == code_b else 0.0) + float(dataset_context['bonus'])",
            ]
        ),
        encoding="utf-8",
    )

    algorithm = resolve_pair_algorithm(Path(module_path))
    context = prepare_algorithm_dataset(algorithm, [1, 2, 3], algorithm_options={"bonus": 0.2})
    score = score_pair_with_algorithm("same", "same", algorithm, dataset_context=context)
    metadata = pair_algorithm_metadata(algorithm, algorithm_options={"bonus": 0.2})

    assert context["n"] == 3
    assert score == pytest.approx(1.2)
    assert metadata["algorithm_source_fingerprint"]["file_name"] == "custom_algorithm.py"
    assert len(metadata["algorithm_source_fingerprint"]["sha256"]) == 64
    assert metadata["algorithm_options"] == {"bonus": 0.2}


def test_score_source_pairs_with_algorithm_attaches_reproducibility_metadata(tmp_path):
    source_root = tmp_path / "codes"
    source_root.mkdir()
    (source_root / "a.py").write_text("print(1)\n", encoding="utf-8")
    (source_root / "b.py").write_text("print(1)\n", encoding="utf-8")
    (source_root / "c.py").write_text("print(2)\n", encoding="utf-8")
    module_path = tmp_path / "algo.py"
    module_path.write_text(
        "\n".join(
            [
                "def prepare_dataset(dataset, bias=0.0):",
                "    return {'bias': float(bias), 'code_count': dataset['code_count']}",
                "",
                "def score_pair(code_a, code_b, dataset_context=None, row=None):",
                "    return (1.0 if code_a == code_b else 0.0) + dataset_context['bias']",
            ]
        ),
        encoding="utf-8",
    )

    results = score_source_pairs_with_algorithm(
        source_root,
        algorithm=module_path,
        algorithm_options={"bias": 0.1},
        number_results=2,
    )

    assert results.iloc[0]["file_name_1"] == "a.py"
    assert results.iloc[0]["file_name_2"] == "b.py"
    assert results.iloc[0]["similarity_score"] == pytest.approx(1.1)
    assert results.attrs["feature_set"] == "custom"
    assert results.attrs["vector_backend"] == "inactive"
    assert results.attrs["algorithm"]["algorithm_options"] == {"bias": 0.1}
    assert len(results.attrs["algorithm"]["algorithm_source_fingerprint"]["sha256"]) == 64
