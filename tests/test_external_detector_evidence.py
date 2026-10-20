"""Verify checked-in diagnostic evidence without local detectors, sources, or network."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from matheel.benchmark_protocol import _validate_public_metadata
from matheel.calibration import calibration_report
from matheel.evaluation import pair_classification_metrics, retrieval_ranking_metrics

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/external-detectors/irplag-java-20261020'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def test_public_detector_diagnostic_is_complete_portable_and_source_free():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob('*') if path.is_file()}
    assert set(manifest['files']) == actual - {'manifest.json'}
    for relative, expected in manifest['files'].items():
        content = (ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected['sha256']
        assert len(content) == expected['bytes']
        if relative.endswith(('.json', '.csv')):
            text = content.decode()
            for private in ('/Users/', '/private/', '/tmp/', 'file://', 'public class Main', 'jar_path', 'java_path', 'dolos_path', 'node_path'):
                assert private not in text
    result = json.loads((ROOT / 'results.json').read_text())
    _validate_public_metadata(result)
    assert result['metadata']['diagnostic_only'] is True
    assert result['metadata']['official_benchmark'] is False
    assert result['metadata']['strict_coverage'] is True
    assert result['metadata']['pair_threshold_policy'] == {'mode': 'fixed', 'value': .5, 'test_label_tuning': False}
    assert result['metadata']['matheel_source']['sha256'] == _digest(result['metadata']['matheel_source']['files'])
    identities = result['metadata']['method_configurations']
    assert len(identities) == 2
    assert all(_digest(config) == method_id for method_id, config in identities.items())
    assert {item['algorithm_id'] for item in result['per_dataset']} == set(identities)
    protocol = json.loads((ROOT.parents[1] / 'plagiarism-v1/protocol.json').read_text())
    frozen = {entry['name']: entry for entry in protocol['datasets']}
    for entry in result['metadata']['inputs']:
        assert entry['content'] == frozen[entry['name']]['content']
        assert entry['source_revision'] == frozen[entry['name']]['source_revision']
    assert len(result['per_dataset']) == len(result['aggregate']) == 12
    assert all(row['eligible'] and math.isfinite(row['score']) for row in result['per_dataset'])
    assert all(row['eligible'] and row['coverage'] == 1 and row['complete_coverage'] for row in result['aggregate'])


def test_public_diagnostic_scores_recompute_metrics_and_full_candidate_coverage():
    result = json.loads((ROOT / 'results.json').read_text())
    index = json.loads((ROOT / 'scores/index.json').read_text())
    assert index['artifacts'] == result['metadata']['score_artifacts']
    assert len(index['artifacts']) == 4
    assert {path.name for path in (ROOT / 'scores').glob('*.csv')} == {item['filename'] for item in index['artifacts']}
    frames = {}
    for artifact in index['artifacts']:
        path = ROOT / 'scores' / artifact['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
        frame = pd.read_csv(path, float_precision='round_trip')
        assert len(frame) == artifact['rows'] == (460 if artifact['task_family'] == 'pair' else 2485)
        assert list(frame.columns) == artifact['columns']
        assert frame.similarity_score.map(math.isfinite).all()
        assert frame.similarity_score.between(0, 1).all()
        if artifact['task_family'] == 'pair':
            assert set(frame.columns) == {'left_id', 'right_id', 'label', 'similarity_score'}
            metrics = pair_classification_metrics(frame, threshold=.5)
            metrics.update(calibration_report(frame, score_key='similarity_score')['summary'])
            structural = frame.drop(columns='similarity_score')
        else:
            assert set(frame.columns) == {'query_id', 'document_id', 'query_file_id', 'document_file_id', 'relevance', 'similarity_score'}
            assert frame.query_id.nunique() == 355
            assert frame.document_id.nunique() == 7
            assert frame.groupby('query_id').document_id.nunique().eq(7).all()
            assert not frame.duplicated(['query_id', 'document_id']).any()
            metrics = retrieval_ranking_metrics(frame, k=10)
            structural = frame.drop(columns='similarity_score')
        previous = frames.setdefault(artifact['task_family'], structural)
        pd.testing.assert_frame_equal(previous, structural)
        for row in result['per_dataset']:
            if row['algorithm_id'] == artifact['algorithm_id'] and row['task_family'] == artifact['task_family']:
                assert row['score'] == pytest.approx(metrics[row['metric']], abs=1e-12)
                assert row['sample_count'] == len(frame)
