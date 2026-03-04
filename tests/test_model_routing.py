from matheel.model_routing import (
    available_vector_backends,
    infer_model_backend,
    infer_model_capabilities,
    resolve_vector_backend,
)


class FakeModelInfo:
    def __init__(self, library_name="", tags=None):
        self.library_name = library_name
        self.tags = list(tags or [])


def test_available_vector_backends_hides_deprecated_static_hash():
    assert available_vector_backends() == ("auto", "sentence_transformers", "model2vec", "pylate")
    assert available_vector_backends(include_deprecated=True) == (
        "auto",
        "sentence_transformers",
        "model2vec",
        "pylate",
        "static_hash",
    )


def test_infer_model_backend_prefers_library_name():
    assert infer_model_backend(
        "any/model",
        model_info=FakeModelInfo(library_name="PyLate", tags=["sentence-similarity"]),
    ) == "pylate"

    assert infer_model_backend(
        "any/model",
        model_info=FakeModelInfo(library_name="model2vec"),
    ) == "model2vec"


def test_resolve_vector_backend_auto_uses_hf_metadata():
    backend = resolve_vector_backend(
        "auto",
        model_name="Jarbas/m2v-256-paraphrase-multilingual-MiniLM-L12-v2",
        model_info=FakeModelInfo(library_name="model2vec", tags=["static-embeddings"]),
    )

    assert backend == "model2vec"


def test_static_sentence_transformer_tags_stay_on_sentence_transformers():
    capabilities = infer_model_capabilities(
        "minishlab/potion-base-8M",
        model_info=FakeModelInfo(
            library_name="sentence-transformers",
            tags=["sentence-transformers", "static-embeddings"],
        ),
    )

    assert capabilities["preferred_backend"] == "sentence_transformers"
    assert capabilities["supports_static"] is True
    assert capabilities["supports_multivector"] is False


def test_static_embeddings_tag_without_library_prefers_sentence_transformers():
    capabilities = infer_model_capabilities(
        "some/static-model",
        model_info=FakeModelInfo(tags=["static-embeddings", "sentence-transformers"]),
    )

    assert capabilities["preferred_backend"] == "sentence_transformers"
    assert capabilities["supports_static"] is True
    assert capabilities["supports_multivector"] is False


def test_colbert_tag_marks_model_as_multivector():
    capabilities = infer_model_capabilities(
        "some/late-model",
        model_info=FakeModelInfo(tags=["ColBERT", "sentence-similarity"]),
    )

    assert capabilities["preferred_backend"] == "pylate"
    assert capabilities["supports_static"] is False
    assert capabilities["supports_multivector"] is True
