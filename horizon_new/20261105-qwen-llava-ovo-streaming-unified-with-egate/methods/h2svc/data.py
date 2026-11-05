"""H2SVC result ledger with the unified benchmark answer fields."""
import ast
import csv
import os
from pathlib import Path

from methods.horizonkv_egate.data import accuracy, correct_letter, task_of, validate_annotations


FIELDS = ['video_id', 'question_index', 'question', 'choices', 'answer', 'correct_choice',
          'pred_answer', 'pred_choice', 'qa_acc', 'task', 'duration_category', 'sampled_frames',
          'seconds', 'visual_slots', 'max_visual_per_head', 'max_retained_frames_per_head']


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_rows(rows, data, equivalent=False, complete=False):
    expected = [(video, index, question) for video in data
                for index, question in enumerate(video['conversations'])]
    if len(rows) > len(expected) or (complete and len(rows) != len(expected)):
        raise ValueError('Incomplete or excessive H2SVC result rows')
    for row, (video, index, question) in zip(rows, expected):
        choices = row['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        if (row['video_id'] != str(video['video_id']) or int(row['question_index']) != index or
            row['question'] != question['question'] or choices != question['choices'] or
            row['answer'] != question['answer'] or row['task'] != task_of(video, question) or
            row['correct_choice'] != correct_letter(question) or
            row['pred_answer'] != row['pred_choice'] or
            float(row['qa_acc']) != accuracy(question, row['pred_choice'], equivalent)):
            raise ValueError('H2SVC result row does not match annotation or score')
        if int(row['max_retained_frames_per_head']) > 64:
            raise ValueError('H2SVC head exceeded paper frame budget')
    return expected
