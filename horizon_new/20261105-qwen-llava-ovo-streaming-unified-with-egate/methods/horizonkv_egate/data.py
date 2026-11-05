"""Small MCQA result ledger; no per-token cache trace is written."""
import ast
import csv
import json
import math
import os
from pathlib import Path
from .protocol import route_answer

FIELDS = ['video_id', 'question_index', 'question', 'choices', 'answer', 'correct_choice',
          'pred_answer', 'pred_choice', 'qa_acc', 'task', 'duration_category', 'recent_answer',
          'history_answer', 'gate_source', 'sampled_frames', 'seconds', 'recent_slots',
          'history_slots', 'max_visual_per_head', 'history_transfers']


def task_of(video, question):
    return question.get('task', question.get('question_type', video.get('task', ''))) or ''


def correct_letter(q):
    return q.get('correct_choice') or 'ABCDEFGH'[q['choices'].index(q['answer'])]


def accuracy(q, prediction, equivalent=False):
    letters = 'ABCDEFGH'[:len(q['choices'])]
    if prediction not in letters:
        return 0.0
    return 100.0 * (q['choices'][letters.index(prediction)] == q['answer'] if equivalent else prediction == correct_letter(q))


def validate_annotations(data):
    seen = set()
    for v in data:
        if str(v['video_id']) in seen or not Path(v['video_path']).is_file():
            raise ValueError('Duplicate video ID or inaccessible media: ' + str(v['video_id']))
        seen.add(str(v['video_id']))
        last = 0
        if not v.get('conversations'):
            raise ValueError('Video has no questions')
        for q in v['conversations']:
            if not isinstance(q.get('question'), str) or not q['question'].strip():
                raise ValueError('Missing question text')
            choices = q.get('choices')
            if not isinstance(choices, list) or not 1 <= len(choices) <= 8 or q.get('answer') not in choices:
                raise ValueError('Invalid choices or answer')
            letter = correct_letter(q)
            if letter not in 'ABCDEFGH'[:len(choices)] or choices[ord(letter) - 65] != q['answer']:
                raise ValueError('Indexed answer does not match answer text')
            end = float(q['end_time'])
            if not math.isfinite(end) or end <= 0 or end < last:
                raise ValueError('Question times must be positive and nondecreasing')
            last = end
    if not data:
        raise ValueError('No selected videos')


def validate_rows(rows, data, equivalent=False, complete=False):
    expected = [(v, i, q) for v in data for i, q in enumerate(v['conversations'])]
    if len(rows) > len(expected) or complete and len(rows) != len(expected):
        raise ValueError('Result ledger is incomplete or contains extra rows')
    for row, (v, i, q) in zip(rows, expected):
        choices = ast.literal_eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
        if (row['video_id'], str(row['question_index']), row['question'], choices, row['answer'], row['correct_choice'], row['task']) != (
                str(v['video_id']), str(i), q['question'], q['choices'], q['answer'], correct_letter(q), task_of(v, q)):
            raise ValueError('Result ledger annotation mismatch')
        result, source = route_answer(row['recent_answer'], row['history_answer'], len(q['choices']))
        if result != row['pred_choice'] or source != row['gate_source'] or row['pred_answer'] != result:
            raise ValueError('Result gate routing is inconsistent')
        if float(row['qa_acc']) != accuracy(q, result, equivalent) or int(row['max_visual_per_head']) > 500:
            raise ValueError('Incorrect result score or budget')
    return expected


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def save_exact(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Resume configuration changed: ' + str(path))
    if not path.exists():
        path.write_text(text)
