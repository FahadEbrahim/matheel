"""Explicit local integration tests; no downloads and no network services."""

import os

import pandas as pd
import pytest

from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.evaluation import score_pair_dataset, score_retrieval_dataset
from matheel.external_detectors import dolos, jplag
from matheel.external_detectors.common import DetectorError

pytestmark = pytest.mark.integration

SOURCES = [
    'public class Main { public static int sum(int n) { int value=0; for(int i=0;i<n;i++){value+=i;} return value; } public static void main(String[] args){System.out.println(sum(20));}}',
    'public class Main { public static int sum(int n) { int value=0; for(int i=0;i<n;i++){value+=i;} return value; } public static void main(String[] args){System.out.println(sum(20));}}',
    'public interface Main { String getName(); boolean isValid(); void run(); }',
]


@pytest.fixture(params=['jplag', 'dolos'])
def local_detector(request):
    if request.param == 'jplag':
        values = {key: os.environ.get(env) for key, env in [('jar_path', 'MATHEEL_JPLAG_JAR'), ('java_path', 'MATHEEL_JAVA')]}
        module = jplag
    else:
        values = {key: os.environ.get(env) for key, env in [('dolos_path', 'MATHEEL_DOLOS_PREFIX'), ('node_path', 'MATHEEL_NODE')]}
        module = dolos
    if not all(values.values()):
        pytest.skip('Local detector tool paths are not configured; this test never downloads tools.')
    return module, {'language': 'java', **values}


def test_real_detector_pair_and_retrieval_candidate_coverage(tmp_path, local_detector):
    module, options = local_detector
    files = pd.DataFrame([{'file_id': str(i), 'text': code, 'suffix': '.java'} for i, code in enumerate(SOURCES)])
    write_pair_dataset(tmp_path / 'pairs', files, pd.DataFrame([
        {'left_id': '0', 'right_id': '1', 'label': 1}, {'left_id': '0', 'right_id': '2', 'label': 0},
        {'left_id': '1', 'right_id': '2', 'label': 0}]))
    scored = score_pair_dataset(tmp_path / 'pairs', algorithm=module, algorithm_options=options)
    assert len(scored) == 3
    assert scored.similarity_score.tolist() == [1, 0, 0]
    identity = scored.attrs['algorithm']['algorithm_options']
    assert identity['version'] == module.VERSION
    assert str(tmp_path) not in str(identity)
    write_retrieval_dataset(tmp_path / 'retrieval', files,
        pd.DataFrame([{'query_id': 'q0', 'file_id': '0'}, {'query_id': 'q1', 'file_id': '1'}]),
        pd.DataFrame([{'document_id': f'd{i}', 'file_id': str(i)} for i in range(3)]),
        pd.DataFrame([{'query_id': 'q0', 'document_id': 'd1', 'relevance': 1},
                      {'query_id': 'q1', 'document_id': 'd0', 'relevance': 1}]))
    retrieval = score_retrieval_dataset(tmp_path / 'retrieval', algorithm=module,
                                        algorithm_options=options, exclude_self_matches=True)
    assert set(zip(retrieval.query_id, retrieval.document_id)) == {('q0', 'd1'), ('q0', 'd2'), ('q1', 'd0'), ('q1', 'd2')}
    assert retrieval.similarity_score.tolist() == [1, 0, 1, 0]


def test_real_detector_rejects_unparseable_input(local_detector):
    module, options = local_detector
    with pytest.raises(DetectorError):
        module.prepare_dataset({}, prepared_texts={'a': SOURCES[0], 'b': 'public class ??? { ??'}, algorithm_options=options)


def test_real_detector_cli_import_name_exports_portable_provenance(tmp_path, local_detector):
    import json
    from click.testing import CliRunner
    from matheel.cli import main
    module, options = local_detector
    files = pd.DataFrame([{'file_id': str(i), 'text': code, 'suffix': '.java'} for i, code in enumerate(SOURCES)])
    write_pair_dataset(tmp_path / 'pairs', files, pd.DataFrame([
        {'left_id': '0', 'right_id': '1', 'label': 1}, {'left_id': '0', 'right_id': '2', 'label': 0}]))
    arguments = ['evaluate-pairs', str(tmp_path / 'pairs'), '--algorithm', module.__name__,
                 '--scores-out', str(tmp_path / 'scores.csv'), '--metrics-out', str(tmp_path / 'metrics.json'),
                 '--reproducibility-out', str(tmp_path / 'reproducibility.json')]
    for key, value in options.items():
        arguments.extend(['--algorithm-option', f'{key}={value}'])
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, str(result.exception) + result.output
    assert pd.read_csv(tmp_path / 'scores.csv').similarity_score.tolist() == [1, 0]
    assert json.loads((tmp_path / 'metrics.json').read_text())['pair_count'] == 2
    provenance = json.loads((tmp_path / 'reproducibility.json').read_text())
    assert str(tmp_path) not in json.dumps(provenance)
    assert 'jar_path' not in json.dumps(provenance)
    assert 'node_path' not in json.dumps(provenance)
    assert module.VERSION in json.dumps(provenance)


def test_real_jplag_preserves_asymmetric_query_coverage(local_detector):
    module, options = local_detector
    if module is not jplag:
        pytest.skip('Dolos exposes a symmetric score; directional mode belongs to JPlag.')
    extended = SOURCES[0].rsplit('}', 1)[0] + ' public static boolean hasValue(String s) { return s != null && !s.isEmpty(); } }'
    context = module.prepare_dataset({}, prepared_texts={'short': SOURCES[0], 'extended': extended},
                                     algorithm_options={**options, 'score_mode': 'query'})
    short = context.score({'query_file_id': 'short', 'document_file_id': 'extended'})
    extended_score = context.score({'query_file_id': 'extended', 'document_file_id': 'short'})
    # Pinned JPlag matches16 of17 short-submission tokens; the class-end
    # boundary is outside the min-token match after adding the new method.
    assert short == pytest.approx(16 / 17)
    assert 0 < extended_score < short
