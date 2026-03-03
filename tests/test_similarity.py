import zipfile

import numpy as np

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
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    archive_path = tmp_path / "codes.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.py", "value = 1  # note")
        archive.writestr("b.py", "value = 1")

    raw_results = similarity.get_sim_list(
        archive_path,
        0.0,
        1.0,
        0.0,
        "fake",
        0.0,
        10,
        vector_backend="sentence_transformers",
    )
    clean_results = similarity.get_sim_list(
        archive_path,
        0.0,
        1.0,
        0.0,
        "fake",
        0.0,
        10,
        preprocess_mode="basic",
        chunking_method="tokens",
        chunk_size=2,
        chunk_overlap=1,
        vector_backend="sentence_transformers",
    )

    assert raw_results.iloc[0]["similarity_score"] < 1.0
    assert clean_results.iloc[0]["similarity_score"] == 1.0


def test_get_sim_list_accepts_directory_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    source_dir = tmp_path / "codes"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("value = 1  # note", encoding="utf-8")
    (source_dir / "b.py").write_text("value = 1", encoding="utf-8")
    (source_dir / ".DS_Store").write_text("ignore", encoding="utf-8")

    clean_results = similarity.get_sim_list(
        source_dir,
        0.0,
        1.0,
        0.0,
        "fake",
        0.0,
        10,
        preprocess_mode="basic",
        chunking_method="tokens",
        chunk_size=2,
        chunk_overlap=1,
        vector_backend="sentence_transformers",
    )

    assert len(clean_results) == 1
    assert clean_results.iloc[0]["file_name_1"] == "a.py"
    assert clean_results.iloc[0]["file_name_2"] == "b.py"
    assert clean_results.iloc[0]["similarity_score"] == 1.0


def test_calculate_similarity_supports_chunking(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    score = similarity.calculate_similarity(
        "def add(a, b):\n    total = a + b\n    return total\n",
        "def add(a, b):\n    total = a + b\n    return total\n",
        1.0,
        0.0,
        0.0,
        "fake",
        chunking_method="tokens",
        chunk_size=2,
        chunk_overlap=1,
        max_chunks=2,
        chunk_aggregation="mean",
        vector_backend="sentence_transformers",
    )

    assert score == 1.0


def test_calculate_similarity_supports_code_metric_without_semantic_weights(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    score = similarity.calculate_similarity(
        "int value = 1;",
        "int value = 1;",
        0.0,
        0.0,
        0.0,
        "fake",
        code_metric="codebleu",
        code_metric_weight=1.0,
        vector_backend="sentence_transformers",
    )

    assert score == 1.0


def test_calculate_similarity_supports_static_hash_backend():
    score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(b, a):\n    return b + a\n",
        1.0,
        0.0,
        0.0,
        "unused",
        vector_backend="static_hash",
        static_vector_dim=64,
    )

    assert score == 1.0


def test_calculate_similarity_falls_back_when_model2vec_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": None,
    )

    score = similarity.calculate_similarity(
        "def add(a, b):\n    return a + b\n",
        "def add(b, a):\n    return b + a\n",
        1.0,
        0.0,
        0.0,
        "Jarbas/m2v-256-paraphrase-multilingual-MiniLM-L12-v2",
        vector_backend="model2vec",
        static_vector_dim=64,
    )

    assert score == 1.0


def test_calculate_similarity_supports_multivector_backend(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    score = similarity.calculate_similarity(
        "def normalize(name):\n    return name.strip().lower()\n",
        "def normalize(name):\n    return name.strip().lower()\n",
        1.0,
        0.0,
        0.0,
        "fake",
        vector_backend="pylate",
        chunking_method="tokens",
        chunk_size=2,
        chunk_overlap=0,
    )

    assert score == 1.0


def test_calculate_similarity_normalizes_custom_feature_weights(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
    )

    score = similarity.calculate_similarity(
        "public int square(int value) {\n    return value * value;\n}\n",
        "public int square(int value) {\n    return value * value;\n}\n",
        5.0,
        0.0,
        0.0,
        "fake",
        vector_backend="sentence_transformers",
        feature_weights={"semantic": 3.0, "levenshtein": 1.0},
    )

    assert score == 1.0


def test_get_sim_list_auto_backend_uses_routing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        similarity,
        "load_backend_model",
        lambda model_name, vector_backend="auto", device="auto": FakeModel(),
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
        1.0,
        0.0,
        0.0,
        "sentence-transformers/all-MiniLM-L6-v2",
        0.0,
        5,
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
