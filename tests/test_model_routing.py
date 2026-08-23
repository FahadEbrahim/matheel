import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import matheel.model_routing as model_routing_module
from matheel.model_routing import (
    available_vector_backends,
    infer_model_backend,
    infer_model_capabilities,
    load_hf_model_info,
    normalize_vector_backend_name,
    resolve_vector_backend,
)


class FakeModelInfo:
    def __init__(self, library_name="", tags=None, config=None):
        self.library_name = library_name
        self.tags = list(tags or [])
        self.config = dict(config or {})


def test_available_vector_backends_hides_deprecated_static_hash():
    assert available_vector_backends() == (
        "auto",
        "sentence_transformers",
        "model2vec",
        "multivector",
    )
    assert available_vector_backends(include_deprecated=True) == (
        "auto",
        "sentence_transformers",
        "model2vec",
        "multivector",
        "static_hash",
    )


def test_infer_model_backend_prefers_library_name():
    assert infer_model_backend(
        "any/model",
        model_info=FakeModelInfo(library_name="PyLate", tags=["sentence-similarity"]),
    ) == "multivector"

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

    assert capabilities["preferred_backend"] == "multivector"
    assert capabilities["supports_static"] is False
    assert capabilities["supports_multivector"] is True


def test_native_multivector_tag_and_colbert_architecture_are_detected():
    tagged = infer_model_capabilities(
        "lightonai/LateOn",
        model_info=FakeModelInfo(
            library_name="sentence-transformers",
            tags=["multi-vector"],
        ),
    )
    architecture = infer_model_capabilities(
        "vendor/model",
        model_info=FakeModelInfo(
            library_name="transformers",
            config={"architectures": ["HF_ColBERT"]},
        ),
    )

    assert tagged["preferred_backend"] == "multivector"
    assert architecture["preferred_backend"] == "multivector"


def test_pylate_backend_name_remains_a_multivector_compatibility_alias():
    assert normalize_vector_backend_name("pylate") == "multivector"


def test_hf_model_info_cache_is_shared_across_concurrent_loads(monkeypatch):
    calls = []
    model_info = FakeModelInfo(library_name="sentence-transformers")

    class FakeHfApi:
        def model_info(self, model_name):
            calls.append(model_name)
            time.sleep(0.01)
            return model_info

    model_routing_module._HF_MODEL_INFO_CACHE.clear()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_hf_model_info, ["demo/model"] * 8))

    assert results == [model_info] * 8
    assert calls == ["demo/model"]
