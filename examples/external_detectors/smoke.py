"""Run real optional local detectors on synthetic Java, never benchmark data.

Example: python examples/external_detectors/smoke.py --detector jplag \
  --jar-path /tools/jplag-6.3.0.jar --java-path /tools/jdk-25/bin/java
"""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from matheel.datasets import write_pair_dataset, write_retrieval_dataset
from matheel.evaluation import score_pair_dataset, score_retrieval_dataset

SOURCES = [
    'public class Main { public static int sum(int n) { int value=0; for(int i=0;i<n;i++){value+=i;} return value; } public static void main(String[] args){System.out.println(sum(20));}}',
    'public class Main { public static int sum(int n) { int value=0; for(int i=0;i<n;i++){value+=i;} return value; } public static void main(String[] args){System.out.println(sum(20));}}',
    'public interface Main { String getName(); boolean isValid(); void run(); }',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=['jplag', 'dolos'], required=True)
    parser.add_argument('--java-path')
    parser.add_argument('--jar-path')
    parser.add_argument('--node-path')
    parser.add_argument('--dolos-path')
    args = parser.parse_args()
    if args.detector == 'jplag':
        options = {'jar_path': args.jar_path, 'java_path': args.java_path, 'language': 'java', 'min_tokens': 5}
    else:
        options = {'dolos_path': args.dolos_path, 'node_path': args.node_path, 'language': 'java'}
    algorithm = f'matheel.external_detectors.{args.detector}'
    files = pd.DataFrame([{'file_id': str(i), 'text': code, 'suffix': '.java'} for i, code in enumerate(SOURCES)])
    with tempfile.TemporaryDirectory(prefix='matheel-detector-example-') as temporary:
        root = Path(temporary)
        write_pair_dataset(root / 'pairs', files, pd.DataFrame([
            {'left_id': '0', 'right_id': '1', 'label': 1},
            {'left_id': '0', 'right_id': '2', 'label': 0},
            {'left_id': '1', 'right_id': '2', 'label': 0},
        ]), metadata={'name': 'synthetic-local-detector-smoke', 'task_type': 'plagiarism'})
        scored = score_pair_dataset(root / 'pairs', algorithm=algorithm, algorithm_options=options)
        assert scored.iloc[0].similarity_score == 1
        assert len(scored) == 3
        write_retrieval_dataset(root / 'retrieval', files,
            pd.DataFrame([{'query_id': 'q0', 'file_id': '0'}, {'query_id': 'q1', 'file_id': '1'}]),
            pd.DataFrame([{'document_id': f'd{i}', 'file_id': str(i)} for i in range(3)]),
            pd.DataFrame([{'query_id': 'q0', 'document_id': 'd1', 'relevance': 1},
                          {'query_id': 'q1', 'document_id': 'd0', 'relevance': 1}]),
            metadata={'name': 'synthetic-local-detector-smoke', 'task_type': 'plagiarism'})
        retrieval = score_retrieval_dataset(root / 'retrieval', algorithm=algorithm,
                                           algorithm_options=options, exclude_self_matches=True)
        assert len(retrieval) == 4
        print('Synthetic smoke only; these scores are not a plagiarism benchmark.')
        print(scored.to_string(index=False))
        print(retrieval.to_string(index=False))
        print(json.dumps(scored.attrs['algorithm'], indent=2))


if __name__ == '__main__':
    main()
