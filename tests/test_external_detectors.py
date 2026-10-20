import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from matheel.algorithms import pair_algorithm_metadata, public_algorithm_configuration
from matheel.benchmark_protocol import method_configuration_id
from matheel.datasets import PairDataset, RetrievalDataset
from matheel.evaluation import score_pair_dataset, score_retrieval_dataset
from matheel.external_detectors import dolos, jplag
from matheel.external_detectors.common import (
    DetectorError, DetectorTimeoutError, complete_context, export_corpus,
    file_sha256, run_local, score_pair, tree_sha256,
)
from matheel.leaderboard import leaderboard_payload, run_leaderboard
from matheel.reproducibility import collect_reproducibility_snapshot


def _fake_runtime(path, version):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho \'{version}\' >&2\n')
    path.chmod(0o755)
    return path


def _fake_jplag(root, monkeypatch):
    jar = root / 'jplag.jar'
    root.mkdir(parents=True)
    jar.write_bytes(b'pinned jar fixture')
    monkeypatch.setattr(jplag, 'JAR_SHA256', file_sha256(jar))
    java = _fake_runtime(root / 'jdk/bin/java', 'openjdk version "25.0.1"')
    (root / 'jdk/lib').mkdir()
    (root / 'jdk/release').write_text('JAVA_VERSION="25.0.1"')
    (root / 'jdk/lib/modules').write_bytes(b'java modules fixture')
    return {'jar_path': str(jar), 'java_path': str(java), 'language': 'java'}


def _fake_dolos(root):
    installation = root / 'dolos'
    package = installation / 'node_modules/@dodona/dolos/package.json'
    package.parent.mkdir(parents=True)
    package.write_text(json.dumps({'name': '@dodona/dolos', 'version': dolos.VERSION}))
    (installation / 'package-lock.json').write_text(json.dumps({'packages': {'node_modules/@dodona/dolos': {
        'version': dolos.VERSION, 'integrity': dolos.NPM_INTEGRITY}}}))
    node = _fake_runtime(root / 'node/bin/node', 'v22.23.3')
    return {'dolos_path': str(installation), 'node_path': str(node), 'language': 'java'}


def _jplag_report(path, aliases=('s000000', 's000001'), comparisons=None, **info):
    if comparisons is None:
        comparisons = [{'firstSubmissionId': aliases[0], 'secondSubmissionId': aliases[1],
                        'firstSimilarity': .75, 'secondSimilarity': .25, 'similarities': {'AVG': .375, 'MAX': .75}}]
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('runInformation.json', json.dumps({'version': {'major': 6, 'minor': 3, 'patch': 0},
                          'failedSubmissions': [], 'totalComparisons': len(aliases) * (len(aliases) - 1) // 2, **info}))
        archive.writestr('submissionFileIndex.json', json.dumps({'fileIndexes': {alias: {'Main.java': {'tokenCount': 20}} for alias in aliases}}))
        for i, comparison in enumerate(comparisons):
            archive.writestr(f'comparisons/{i}.json', json.dumps(comparison))
    return path


def test_complete_context_maps_directions_and_never_infers_zero():
    aliases = {'a': 'query', 'b': 'document'}
    context = complete_context([('b', 'a', .2, .8)], aliases, {})
    assert context.score({'query_file_id': 'query', 'document_file_id': 'document'}) == .8
    assert context.score({'left_id': 'document', 'right_id': 'query'}) == .2
    assert context.score({'file_name_1': 'query', 'file_name_2': 'document'}) == .8
    with pytest.raises(DetectorError, match='Incomplete'):
        complete_context([], aliases, {})
    with pytest.raises(DetectorError, match='self comparisons'):
        context.score({'left_id': 'query', 'right_id': 'query'})
    with pytest.raises(DetectorError, match='does not cover'):
        context.score({'left_id': 'query', 'right_id': 'absent'})


@pytest.mark.parametrize('records', [
    [('a', 'a', 1, 1)], [('a', 'c', 0, 0)], [('a', 'b', 0, 0), ('b', 'a', 0, 0)],
    [('a', 'b', float('nan'), 0)], [('a', 'b', float('inf'), 0)], [('a', 'b', -.1, 0)], [('a', 'b', 1.1, 0)],
])
def test_complete_context_rejects_invalid_pairs_and_scores(records):
    with pytest.raises(DetectorError):
        complete_context(records, {'a': 'one', 'b': 'two'}, {})


def test_export_only_opaque_source_inputs_and_all_corpus_members(tmp_path):
    dataset = {'labels': ['SECRET_LABEL'], 'qrels': ['SECRET_QREL']}
    aliases = export_corpus(dataset, {'private/name.java': 'class A {}', 'other': 'class B {}',
                                     'unjudged': 'class C {}'}, tmp_path / 'inputs', '.java')
    assert list(aliases) == ['s000000', 's000001', 's000002']
    assert set(aliases.values()) == {'private/name.java', 'other', 'unjudged'}
    files = sorted((tmp_path / 'inputs').rglob('*'))
    assert len([p for p in files if p.is_file()]) == 3
    assert {p.name for p in files if p.is_file()} == {'Main.java'}
    assert 'SECRET_' not in ''.join(p.read_text() for p in files if p.is_file())
    assert 'private' not in ''.join(p.name for p in files)


@pytest.mark.parametrize('texts,limits,match', [
    ({'a': 'one'}, {}, 'at least two'), ({'a': '', 'b': 'two'}, {}, 'Empty'),
    ({'a': 'one', 'b': 'two'}, {'max_files': 1}, 'max_files'),
    ({'a': 'one', 'b': 'two'}, {'max_source_bytes': 2}, 'max_source_bytes'),
    ({'a': 'one', 'b': 'two', 'c': 'three'}, {'max_comparisons': 1}, 'max_comparisons'),
])
def test_export_rejects_unsupported_input_and_limits(tmp_path, texts, limits, match):
    with pytest.raises(DetectorError, match=match):
        export_corpus({}, texts, tmp_path / 'inputs', '.java', **limits)


def test_bounded_subprocess_timeout_failure_output_and_no_secret_leak(tmp_path):
    started = time.monotonic()
    with pytest.raises(DetectorTimeoutError):
        run_local([sys.executable, '-c', 'import time; time.sleep(10)'], cwd=tmp_path, timeout_seconds=.1)
    assert time.monotonic() - started < 3
    with pytest.raises(DetectorError, match='max_output_bytes'):
        run_local([sys.executable, '-c', 'print("x"*50000)'], cwd=tmp_path, timeout_seconds=3, max_output_bytes=100)
    with pytest.raises(DetectorError, match='status 7') as failure:
        run_local([sys.executable, '-c', 'print("PRIVATE_SOURCE_AND_PATH");raise SystemExit(7)'], cwd=tmp_path, timeout_seconds=3)
    assert 'PRIVATE_' not in str(failure.value)
    output = tmp_path / 'results.json'
    with pytest.raises(DetectorError, match='max_result_bytes'):
        run_local([sys.executable, '-c', f'from pathlib import Path;import time;Path({str(output)!r}).write_text("x"*1000);time.sleep(2)'],
                  cwd=tmp_path, timeout_seconds=3, result_paths=[output], max_result_bytes=100)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process group cleanup')
def test_timeout_kills_spawned_descendant(tmp_path):
    marker = tmp_path / 'escaped'
    child = f'import time;from pathlib import Path;time.sleep(.5);Path({str(marker)!r}).write_text("escaped")'
    command = f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);time.sleep(10)'
    with pytest.raises(DetectorTimeoutError):
        run_local([sys.executable, '-c', command], cwd=tmp_path, timeout_seconds=.1)
    time.sleep(.7)
    assert not marker.exists()


@pytest.mark.parametrize('mode,forward,reverse', [('average', .375, .375), ('maximum', .75, .75), ('query', .75, .25)])
def test_jplag_real_schema_directional_mapping(tmp_path, mode, forward, reverse):
    path = _jplag_report(tmp_path / 'report.jplag')
    context = jplag._parse_report(path, {'s000000': 'A', 's000001': 'B'}, {'configuration': {'score_mode': mode}}, 100000)
    assert context.score({'left_id': 'A', 'right_id': 'B'}) == forward
    assert context.score({'query_file_id': 'B', 'document_file_id': 'A'}) == reverse


@pytest.mark.parametrize('info,comparisons,match', [
    ({'failedSubmissions': ['s000001']}, None, 'rejected'), ({'totalComparisons': 0}, None, 'full declared'),
    ({}, [], 'Incomplete'),
])
def test_jplag_rejects_parser_failures_and_missing_coverage(tmp_path, info, comparisons, match):
    path = _jplag_report(tmp_path / 'report.jplag', comparisons=comparisons, **info)
    with pytest.raises(DetectorError, match=match):
        jplag._parse_report(path, {'s000000': 'A', 's000001': 'B'}, {'configuration': {'score_mode': 'average'}}, 100000)


def test_jplag_reverifies_pinned_jar_and_portable_method_identity(tmp_path, monkeypatch):
    first = _fake_jplag(tmp_path / 'first', monkeypatch)
    second_root = tmp_path / 'second'
    shutil.copytree(tmp_path / 'first', second_root)
    second = {**first, 'jar_path': str(second_root / 'jplag.jar'), 'java_path': str(second_root / 'jdk/bin/java')}
    method = {'algorithm_path': 'matheel.external_detectors.jplag', 'algorithm_options': first}
    other = {**method, 'algorithm_options': second}
    assert method_configuration_id(method) == method_configuration_id(other)
    assert method_configuration_id(method) != method_configuration_id({**method, 'algorithm_options': {**first, 'min_tokens': 12}})
    public = public_algorithm_configuration(method)
    assert str(tmp_path) not in json.dumps(public)
    assert method_configuration_id(method) == method_configuration_id(public)
    assert str(tmp_path) not in json.dumps(pair_algorithm_metadata(method['algorithm_path'], first))
    assert str(tmp_path) not in json.dumps(collect_reproducibility_snapshot(run_configs=[method]))
    Path(first['jar_path']).write_bytes(b'tampered')
    with pytest.raises(DetectorError, match='SHA256'):
        jplag.portable_configuration(first)


def test_dolos_verifies_installed_bytes_and_portable_identity(tmp_path):
    first = _fake_dolos(tmp_path / 'one')
    shutil.copytree(tmp_path / 'one', tmp_path / 'two')
    second = {'language': 'java', 'dolos_path': str(tmp_path / 'two/dolos'), 'node_path': str(tmp_path / 'two/node/bin/node')}
    method = {'algorithm_path': 'matheel.external_detectors.dolos', 'algorithm_options': first}
    assert method_configuration_id(method) == method_configuration_id({**method, 'algorithm_options': second})
    public = public_algorithm_configuration(method)
    assert str(tmp_path) not in json.dumps(public)
    assert method_configuration_id(method) == method_configuration_id(public)
    assert method_configuration_id(method) != method_configuration_id({**method, 'algorithm_options': {**first, 'kgram_length': 7}})
    identity = dolos.portable_configuration(first)
    pinned = {**first, 'expected_installation_sha256': identity['installation_sha256']}
    assert dolos.portable_configuration(pinned) == identity
    (Path(first['dolos_path']) / 'node_modules/@dodona/dolos/changed.js').write_text('changed')
    assert dolos.portable_configuration(first)['installation_sha256'] != identity['installation_sha256']
    with pytest.raises(DetectorError, match='expected_installation_sha256'):
        dolos.portable_configuration(pinned)


def test_tree_fingerprint_includes_native_runtime_and_rejects_escape(tmp_path):
    (tmp_path / 'build/Release').mkdir(parents=True)
    native = tmp_path / 'build/Release/parser.node'
    native.write_bytes(b'one')
    before = tree_sha256(tmp_path)
    native.write_bytes(b'two')
    assert before != tree_sha256(tmp_path)
    (tmp_path / 'external').symlink_to(sys.executable)
    with pytest.raises(DetectorError, match='symbolic'):
        tree_sha256(tmp_path)


def test_dolos_rejects_missing_pairs_files_and_invalid_fingerprints(tmp_path):
    path = tmp_path / 'report.json'
    data = {'schema_version': 1, 'files': [{'alias': 'a', 'kgrams': 4}, {'alias': 'b', 'kgrams': 5}],
            'pairs': [{'left': 'a', 'right': 'b', 'similarity': 0}]}
    path.write_text(json.dumps(data))
    context = dolos._parse_report(path, {'a': 'A', 'b': 'B'}, {}, 10000)
    assert context.score({'left_id': 'A', 'right_id': 'B'}) == 0
    for field, value, match in [('pairs', [], 'Incomplete'), ('files', data['files'][:1], 'every declared')]:
        path.write_text(json.dumps({**data, field: value}))
        with pytest.raises(DetectorError, match=match):
            dolos._parse_report(path, {'a': 'A', 'b': 'B'}, {}, 10000)
    data['files'][0]['kgrams'] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(DetectorError, match='without fingerprints'):
        dolos._parse_report(path, {'a': 'A', 'b': 'B'}, {}, 10000)


@pytest.mark.parametrize('module,options,match', [
    (jplag, {'language': 'text'}, 'language'), (dolos, {'language': 'char'}, 'language'),
    (jplag, {'language': 'java', 'timeout_seconds': float('nan')}, 'finite'),
    (jplag, {'language': 'python3', 'normalize': True}, 'normalize'),
    (dolos, {'language': 'java', 'include_comments': 'false'}, 'boolean'),
    (dolos, {'language': 'java', 'kgram_length': 1.5}, 'integer'),
    (jplag, {'language': 'java', 'unknown_flag': 'oops'}, 'Unknown'),
])
def test_explicit_support_and_config_contract(module, options, match):
    with pytest.raises(DetectorError, match=match):
        module._configuration(options)


def test_once_per_corpus_for_pair_and_directed_retrieval(tmp_path, monkeypatch):
    calls = []
    aliases = {'s000000': 'A', 's000001': 'B', 's000002': 'C'}
    context = complete_context([('s000000', 's000001', .8, .2), ('s000000', 's000002', 0, 0),
                                ('s000001', 's000002', .5, .5)], aliases, {})
    def prepare(dataset, prepared_texts=None, **kwargs):
        calls.append(set(prepared_texts))
        return context
    monkeypatch.setattr(jplag, 'prepare_dataset', prepare)
    monkeypatch.setattr(jplag, 'portable_configuration', lambda algorithm_options=None: {'fixture': True})
    for key in 'ABC':
        (tmp_path / f'{key}.java').write_text(f'class {key} {{}}')
    files = pd.DataFrame([{'file_id': key, 'file_path': f'{key}.java'} for key in 'ABC'])
    pair = PairDataset(tmp_path, files, pd.DataFrame([{'left_id': 'A', 'right_id': 'B', 'label': 1},
                                                    {'left_id': 'A', 'right_id': 'C', 'label': 0}]), {})
    scored = score_pair_dataset(pair, algorithm=jplag)
    assert scored.similarity_score.tolist() == [.8, 0]
    assert calls == [{'A', 'B', 'C'}]
    retrieval = RetrievalDataset(tmp_path, files, pd.DataFrame([{'query_id': 'qb', 'file_id': 'B'}]),
                                 pd.DataFrame([{'document_id': f'd{key}', 'file_id': key} for key in 'ABC']),
                                 pd.DataFrame([{'query_id': 'qb', 'document_id': 'dA', 'relevance': 1}]), {})
    scored = score_retrieval_dataset(retrieval, algorithm=jplag, exclude_self_matches=True)
    assert scored.document_id.tolist() == ['dA', 'dC']
    assert scored.similarity_score.tolist() == [.2, .5]
    assert calls == [{'A', 'B', 'C'}, {'A', 'B', 'C'}]
    with pytest.raises(DetectorError, match='exclude_self_matches'):
        score_retrieval_dataset(retrieval, algorithm=jplag)


def test_score_pair_requires_context_and_stable_ids():
    with pytest.raises(DetectorError, match='prepare_dataset'):
        score_pair('same', 'same')
    context = complete_context([('a', 'b', .3, .3)], {'a': 'A', 'b': 'B'}, {})
    with pytest.raises(DetectorError, match='stable file IDs'):
        score_pair('same', 'same', dataset_context=context, row={'label': 1})


def test_leaderboard_artifacts_publish_hashes_without_runtime_paths(tmp_path, monkeypatch):
    options = _fake_jplag(tmp_path / 'runtime', monkeypatch)
    root = tmp_path / 'dataset'
    root.mkdir()
    for key in 'ABC':
        (root / f'{key}.java').write_text(f'class {key} {{}}')
    pd.DataFrame([{'file_id': key, 'file_path': f'{key}.java'} for key in 'ABC']).to_csv(root / 'files.csv', index=False)
    pd.DataFrame([{'left_id': 'A', 'right_id': 'B', 'label': 1},
                  {'left_id': 'A', 'right_id': 'C', 'label': 0}]).to_csv(root / 'pairs.csv', index=False)
    (root / 'metadata.json').write_text(json.dumps({'dataset_kind': 'pair_classification', 'task_type': 'plagiarism'}))
    calls = []
    def prepare(dataset, **kwargs):
        calls.append(True)
        return complete_context([('a', 'b', 1, 1), ('a', 'c', 0, 0), ('b', 'c', .2, .2)],
                                {'a': 'A', 'b': 'B', 'c': 'C'}, {})
    monkeypatch.setattr(jplag, 'prepare_dataset', prepare)
    report, artifacts = run_leaderboard({'datasets': [{'name': 'tiny', 'task': 'pair', 'path': str(root)}],
        'algorithms': [{'name': 'JPlag', 'algorithm_path': 'matheel.external_detectors.jplag', 'algorithm_options': options}]},
        output_dir=tmp_path / 'report')
    payload = leaderboard_payload(report)
    assert calls == [True]
    assert payload['aggregate'][0]['mean_score'] == 1
    assert str(tmp_path) not in json.dumps(payload)
    published = (tmp_path / 'report/leaderboard.json').read_text()
    assert 'jar_path' not in published and 'java_path' not in published
    assert jplag.JAR_SHA256 in published
    assert str(tmp_path) not in artifacts['reproducibility_json'].read_text()


def test_algorithm_cli_reference_accepts_modules_and_validates_files(tmp_path):
    from click.testing import CliRunner
    from matheel.cli import main
    (tmp_path / 'a.java').write_text('class A {}')
    result = CliRunner().invoke(main, ['compare', str(tmp_path), '--algorithm', 'missing_detector_xyz.module'])
    assert result.exit_code == 2
    assert 'module could not be found' in result.output
    result = CliRunner().invoke(main, ['compare', str(tmp_path), '--algorithm-path', str(tmp_path / 'absent.py')])
    assert result.exit_code == 2
    assert 'existing Python file' in result.output


def test_portable_hook_cannot_publish_private_paths(tmp_path):
    from types import ModuleType
    module = ModuleType('private_identity')
    module.score_pair = lambda code_a, code_b: 0
    module.portable_configuration = lambda algorithm_options=None: {'runtime': str(tmp_path / 'private')}
    with pytest.raises(ValueError, match='local absolute paths'):
        pair_algorithm_metadata(module)


def test_tree_fingerprint_includes_executable_javascript_inside_build(tmp_path):
    runtime = tmp_path / 'cliui/build/lib/index.js'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('export const version = 1;')
    before = tree_sha256(tmp_path)
    runtime.write_text('export const version = 2;')
    assert tree_sha256(tmp_path) != before
    before = tree_sha256(tmp_path)
    (runtime.parent.parent / 'Makefile').write_text('generated machine path')
    assert tree_sha256(tmp_path) == before


@pytest.mark.parametrize('detector', ['jplag', 'dolos'])
def test_portable_identity_requires_every_generated_provenance_field(tmp_path, monkeypatch, detector):
    module = jplag if detector == 'jplag' else dolos
    options = _fake_jplag(tmp_path / 'tool', monkeypatch) if detector == 'jplag' else _fake_dolos(tmp_path / 'tool')
    identity = module.portable_configuration(options)
    for field in identity:
        with pytest.raises(DetectorError):
            module.portable_configuration({key: value for key, value in identity.items() if key != field})
    for field in identity['runtime']:
        changed = {**identity, 'runtime': {key: value for key, value in identity['runtime'].items() if key != field}}
        with pytest.raises(DetectorError):
            module.portable_configuration(changed)


@pytest.mark.parametrize('module', [jplag, dolos])
def test_unexpected_tool_failures_do_not_publish_private_paths_or_source(module, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError('/Users/private/runtime SECRET_SOURCE_CODE')
    monkeypatch.setattr(module, 'required_file', fail)
    options = {'language': 'java'}
    if module is dolos:
        options['dolos_path'] = '/private/local/dolos'
    with pytest.raises(DetectorError) as failure:
        module.portable_configuration(options)
    assert 'PermissionError' in str(failure.value)
    assert '/Users' not in str(failure.value) and 'SECRET_' not in str(failure.value)
    monkeypatch.setattr(module, 'portable_configuration', fail)
    with pytest.raises(DetectorError) as failure:
        module.prepare_dataset({}, prepared_texts={'a': 'class A {}', 'b': 'class B {}'}, algorithm_options=options)
    assert 'PermissionError' in str(failure.value)
    assert '/Users' not in str(failure.value) and 'SECRET_' not in str(failure.value)
