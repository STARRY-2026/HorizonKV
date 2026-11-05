"""Exercise CSV-to-summary behavior without loading a model."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


class SummaryIntegrationTests(unittest.TestCase):
    def make_case(self, root, model, method, benchmark, native=False):
        shard = root/'shard_00'
        shard.mkdir()
        (root/'annotations').mkdir()
        task = 'OJR' if benchmark == 'ovobench' else 'Object Recognition'
        question = dict(question='Which object is visible?', answer='A cup.', choices=['A cup.', 'A ball.'],
                        correct_choice='A', end_time=6, task=task)
        video = dict(video_id=1, video_path='/not-needed-for-summary.mp4', conversations=[question])
        (root/'annotations/shard_00.json').write_text(json.dumps([video]))
        row = dict(video_id='1', question=question['question'], choices=str(question['choices']),
                   answer=question['answer'], correct_choice='A', pred_answer='A', pred_choice='A',
                   qa_acc=100, task=task, duration_category='')
        name = '1_0.csv' if native else 'results.csv'
        with (shard/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        manifest = dict(model=model, method=method, benchmark=benchmark, gpus=['4'],
                        cohort='heldout', videos=1, questions=1, fps=1,
                        visual_input='native' if native else 'dscache448-video', max_videos=1)
        filename = 'unified_manifest.json' if benchmark == 'ovobench' else 'manifest.json'
        (root/filename).write_text(json.dumps(manifest))

    def test_streaming_qwen_reads_the_actual_result_file(self):
        for backend, method, native in [('qwen', 'adakv', False), ('qwen', 'pyramidkv', False),
                                         ('qwen_full2495', 'adakv', False), ('qwen', 'adakv', True)]:
            with self.subTest(backend=backend, method=method, native=native), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_case(root, 'qwen25vl', method, 'streamingbench', native)
                result = subprocess.run([sys.executable, str(PROJECT/backend/'summarize.py'), str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((root/'summary.json').read_text())['correct'], 1)

    def test_llava_ovo_summary_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_case(root, 'llavaov', 'adakv', 'ovobench')
            result = subprocess.run([sys.executable, str(PROJECT/'llava/summarize.py'), str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root/'summary.json').read_text())['micro_accuracy'], 100)


if __name__ == '__main__':
    unittest.main()
