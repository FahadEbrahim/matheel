import pandas as pd
import pytest

pytest.importorskip("gradio")

from gradio_app.result_analysis import analyze_results_gradio


def test_analysis_callback_reports_errors_and_downloads(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame([
        {"left_id": "a", "right_id": "b", "label": 1, "similarity_score": .2},
        {"left_id": "c", "right_id": "d", "label": 0, "similarity_score": .9},
    ]).to_csv(path, index=False)
    summary, metrics, errors, slices, queries, files = analyze_results_gradio(
        str(path), None, "pair", .5, 10, "", 25)
    assert "Result analysis" in summary
    assert not metrics.empty
    assert set(errors["outcome"]) == {"false_positive", "false_negative"}
    assert files and len(files) == 6
    assert slices.empty and queries.empty
    assert set(errors["score_row_index"]) == {0, 1}


def test_analysis_callback_clears_outputs_on_invalid_reference(tmp_path):
    outputs = analyze_results_gradio(None, None, "auto", .5, 10, "", 25)
    assert "Upload scored" in outputs[0]
    assert all(frame.empty for frame in outputs[1:5])
    assert outputs[-1] is None
