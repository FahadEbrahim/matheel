"""Explicit opt-in validation with already cached real model weights."""

import pytest

from matheel.long_files import compare_long_files


@pytest.mark.integration
def test_cached_minilm_raw_truncation_and_chunked_tail_coverage(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("chonkie")
    from huggingface_hub import snapshot_download

    try:
        model_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    except OSError as exc:
        pytest.skip(f"MiniLM weights are not already cached: {exc}")
    common = ("def common(values):\n    total = 0\n    for value in values:\n"
              "        total += value\n    return total\n") * 4
    left = common + "alpha = [1, 2, 3]\n" * 40
    right = common + "while True:\n    handle_error('network failed')\n" * 40
    options = {"feature_weights": {"semantic": 1}, "model_name": model_path,
               "vector_backend": "sentence_transformers", "device": "cpu", "max_token_length": 64}
    runs = [
        {"run_name": "raw", "options": options},
        {"run_name": "chunks", "options": {
            **options, "chunking_method": "chonkie_token", "chunk_size": 32, "chunk_overlap": 4,
            "chunker_options": {"tokenizer": "character"}, "chunk_aggregation": "mean",
        }},
    ]
    report, _ = compare_long_files(left, right, runs, output_dir=tmp_path)
    assert all(row["status"] == "ok" for row in report["scores"])
    raw, chunked = report["preflight"]
    assert raw["files"]["left"]["chunks"][0]["token_measurement"] == "tokenizer_offsets"
    assert raw["files"]["left"]["truncated_chunks"] == 1
    assert raw["files"]["left"]["chunks"][0]["retained_input_tokens"] == 64
    assert chunked["files"]["left"]["truncated_chunks"] == 0
    assert chunked["files"]["left"]["selected_chunks"] > 10
    assert report["scores"][0]["similarity_score"] == pytest.approx(1, abs=1e-6)
    assert report["scores"][1]["similarity_score"] < report["scores"][0]["similarity_score"] - 0.01
