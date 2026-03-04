import zipfile

import numpy as np
import pytest

from matheel import similarity


class FakeModel:
    def encode(self, inputs, convert_to_numpy=True):
        if isinstance(inputs, str):
            return np.asarray(_vectorize(inputs), dtype=float)
        return np.asarray([_vectorize(item) for item in inputs], dtype=float)


def _vectorize(text):
    value = text or ""
    return [
        float(sum(1 for char in value if char.isalpha())),
        float(sum(1 for char in value if char.isdigit())),
        float(len(value)),
    ]


def test_get_sim_list_uses_preprocessed_code(tmp_path, monkeypatch):
    pytest.importorskip("chonkie")
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    archive_path = tmp_path / "codes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "value = 1  # note")
        archive.writestr("b.py", "value = 1")

    raw_results = similarity.get_sim_list(
        archive_path,
        model_name="fake",
        threshold=0.0,
        number_results=10,
        feature_weights={"semantic": 0.0, "levenshtein": 1.0, "jaro_winkler": 0.0},
        vector_backend="sentence_transformers",
    )
    clean_results = similarity.get_sim_list(
        archive_path,
        model_name="fake",
        threshold=0.0,
        number_results=10,
        feature_weights={"semantic": 0.0, "levenshtein": 1.0, "jaro_winkler": 0.0},
        preprocess_mode="basic",
        chunking_method="chonkie_token",
        chunk_size=2,
        chunk_overlap=1,
        vector_backend="sentence_transformers",
    )

    assert raw_results.iloc[0]["similarity_score"] < 1.0
    assert clean_results.iloc[0]["similarity_score"] == 1.0


def test_get_sim_list_accepts_directory_source(tmp_path, monkeypatch):
    pytest.importorskip("chonkie")
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("value = 1  # note", encoding="utf-8")
    (source_dir / "b.py").write_text("value = 1", encoding="utf-8")
    (source_dir / ".DS_Store").write_text("ignore", encoding="utf-8")

    clean_results = similarity.get_sim_list(
        source_dir,
        model_name="fake",
        threshold=0.0,
        number_results=10,
        feature_weights={"semantic": 0.0, "levenshtein": 1.0, "jaro_winkler": 0.0},
        preprocess_mode="basic",
        chunking_method="chonkie_token",
        chunk_size=2,
        chunk_overlap=1,
        vector_backend="sentence_transformers",
    )

    assert len(clean_results) == 1
    assert clean_results.iloc[0]["file_name_1"] == "a.py"
    assert clean_results.iloc[0]["file_name_2"] == "b.py"
    assert clean_results.iloc[0]["similarity_score"] == 1.0


def test_calculate_similarity_supports_chunking(monkeypatch):
    pytest.importorskip("chonkie")
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    score = similarity.calculate_similarity(
        "def add(a, b):\n    total = a + b\n    return total\n",
        "def add(a, b):\n    total = a + b\n    return total\n",
        model_name="fake",
        feature_weights={"semantic": 1.0},
        chunking_method="chonkie_token",
        chunk_size=2,
        chunk_overlap=1,
        max_chunks=2,
        chunk_aggregation="mean",
        vector_backend="sentence_transformers",
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_supports_code_metric_without_semantic_weights(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    score = similarity.calculate_similarity(
        "int value = 1;",
        "int value = 1;",
        model_name="fake",
        feature_weights={"code_metric": 1.0},
        code_metric="codebleu",
        code_metric_weight=1.0,
        vector_backend="sentence_transformers",
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_supports_custom_levenshtein_weights():
    default_score = similarity.calculate_similarity(
        "abcd",
        "abxcd",
        vector_backend="static_hash",
        feature_weights={"levenshtein": 1.0},
    )
    heavier_insertion_score = similarity.calculate_similarity(
        "abcd",
        "abxcd",
        vector_backend="static_hash",
        feature_weights={"levenshtein": 1.0},
        levenshtein_weights=(3, 1, 1),
    )

    assert heavier_insertion_score < default_score


def test_calculate_similarity_supports_custom_jaro_winkler_prefix_weight():
    default_score = similarity.calculate_similarity(
        "prefixAlpha",
        "prefixBeta",
        vector_backend="static_hash",
        feature_weights={"jaro_winkler": 1.0},
    )
    stronger_prefix_score = similarity.calculate_similarity(
        "prefixAlpha",
        "prefixBeta",
        vector_backend="static_hash",
        feature_weights={"jaro_winkler": 1.0},
        jaro_winkler_prefix_weight=0.25,
    )

    assert stronger_prefix_score > default_score


def test_calculate_similarity_supports_static_hash_backend():
    score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(b, a):\n    return b + a\n",
        feature_weights={"semantic": 1.0},
        vector_backend="static_hash",
        static_vector_dim=64,
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_supports_alternative_similarity_functions():
    dot_score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b\n",
        feature_weights={"semantic": 1.0},
        vector_backend="static_hash",
        similarity_function="dot",
        static_vector_dim=64,
    )
    euclidean_score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b\n",
        feature_weights={"semantic": 1.0},
        vector_backend="static_hash",
        similarity_function="euclidean",
        static_vector_dim=64,
    )

    assert dot_score == pytest.approx(1.0)
    assert euclidean_score == pytest.approx(0.0)


def test_calculate_similarity_falls_back_when_model2vec_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: None,
    )

    score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(b, a):\n    return b + a\n",
        model_name="Jarbas/m2v-256-paraphrase-multilingual-MiniLM-L12-v2",
        feature_weights={"semantic": 1.0},
        vector_backend="model2vec",
        static_vector_dim=64,
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_supports_multivector_backend(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    score = similarity.calculate_similarity(
        "def normalize(name):\n    return name.strip().lower()\n",
        "def normalize(name):\n    return name.strip().lower()\n",
        model_name="fake",
        feature_weights={"semantic": 1.0},
        vector_backend="pylate",
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_normalizes_custom_feature_weights(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )

    score = similarity.calculate_similarity(
        "public int square(int value) {\n    return value * value;\n}\n",
        "public int square(int value) {\n    return value * value;\n}\n",
        model_name="fake",
        vector_backend="sentence_transformers",
        feature_weights={"semantic": 3.0, "levenshtein": 1.0},
    )

    assert score == pytest.approx(1.0)


def test_calculate_similarity_passes_similarity_and_pooling_to_loader(monkeypatch):
    captured = {}

    def fake_load_backend_model(
        model_name,
        vector_backend="auto",
        device="auto",
        similarity_function="cosine",
        pooling_method="mean",
        max_token_length=None,
    ):
        captured["similarity_function"] = similarity_function
        captured["pooling_method"] = pooling_method
        captured["max_token_length"] = max_token_length
        return FakeModel()

    monkeypatch.setattr(similarity, "load_backend_model", fake_load_backend_model)
    monkeypatch.setattr(similarity, "configure_sentence_transformer_pooling", lambda model, pooling_method="mean": model)

    score = similarity.calculate_similarity(
        "def clean_name(name):\n    return name.strip().lower()\n",
        "def clean_name(name):\n    return name.strip().lower()\n",
        model_name="fake",
        feature_weights={"semantic": 1.0},
        vector_backend="sentence_transformers",
        similarity_function="manhattan",
        pooling_method="max",
    )

    assert score == 0.0
    assert captured["similarity_function"] == "manhattan"
    assert captured["pooling_method"] == "max"
    assert captured["max_token_length"] is None


def test_inspect_model_settings_reports_detected_and_configured_max_token_length(monkeypatch):
    monkeypatch.setattr(similarity, "load_hf_model_info", lambda model_name: None)
    monkeypatch.setattr(similarity, "detect_model_max_token_length", lambda model=None, model_name=None, default=512: 256)

    settings = similarity.inspect_model_settings(
        "sentence-transformers/all-MiniLM-L6-v2",
        vector_backend="sentence_transformers",
        device="cpu",
        similarity_function="dot",
        pooling_method="max",
        max_token_length=512,
    )

    assert settings["resolved_vector_backend"] == "sentence_transformers"
    assert settings["runtime_device"] == "cpu"
    assert settings["similarity_function"] == "dot"
    assert settings["pooling_method"] == "max"
    assert settings["detected_max_token_length"] == 256
    assert settings["configured_max_token_length"] == 256
    assert settings["supports_custom_max_token_length"] is True


def test_get_sim_list_auto_backend_uses_routing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto", similarity_function="cosine", pooling_method="mean", max_token_length=None: FakeModel(),
    )
    monkeypatch.setattr(similarity, "load_hf_model_info", lambda model_name: object())
    monkeypatch.setattr(
        similarity,
        "validate_vector_options",
        lambda vector_backend, static_vector_dim, model_name=None, model_info=None: ("sentence_transformers", 256),
    )

    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("print(1)", encoding="utf-8")
    (source_dir / "b.py").write_text("print(1)", encoding="utf-8")

    results = similarity.get_sim_list(
        source_dir,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.0,
        number_results=5,
        feature_weights={"semantic": 1.0},
        vector_backend="auto",
    )

    assert len(results) == 1


def test_detect_default_device_prefers_mps_when_cuda_is_unavailable(monkeypatch):
    class FakeMPS:
        @staticmethod
        def is_available():
            return True

    class FakeCUDA:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = FakeMPS()

    class FakeTorch:
        cuda = FakeCUDA()
        backends = FakeBackends()

    monkeypatch.setattr(similarity, "load_torch", lambda: FakeTorch())

    assert similarity.detect_default_device() == "mps"
    assert similarity.normalize_device("auto") == "mps"


def test_normalize_device_rejects_unavailable_accelerator(monkeypatch):
    class FakeMPS:
        @staticmethod
        def is_available():
            return False

    class FakeCUDA:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = FakeMPS()

    class FakeTorch:
        cuda = FakeCUDA()
        backends = FakeBackends()

    monkeypatch.setattr(similarity, "load_torch", lambda: FakeTorch())

    try:
        similarity.normalize_device("mps")
    except ValueError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("Expected normalize_device to reject unavailable mps.")
