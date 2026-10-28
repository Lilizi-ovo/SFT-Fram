"""Stable answer-word strata and final-answer scoring for native thinking outputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata


def normalize_answer(answer):
    """NFKC + casefold + whitespace only; keep punctuation and Chinese characters."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', answer).casefold()).strip()


def gold_answer(row):
    value = row['messages'][-1]['content']
    if value.startswith('<think>'):
        if value.count('</think>') != 1:
            raise ValueError('Bad thinking target')
        value = value.split('</think>', 1)[1].strip()
    elif value.lstrip().startswith('{'):
        value = json.loads(value)['answer']
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('Missing clean gold answer')
    return value


def final_answer(text):
    if not isinstance(text, str):
        raise ValueError('Prediction is not text')
    if text.count('</think>') > 1 or text.count('<think>') > 1:
        raise ValueError('Repeated thinking block')
    if '</think>' in text:
        value = text.split('</think>', 1)[1].strip()
    elif '<think>' in text:
        raise ValueError('Unclosed thinking block')
    else:
        value = text.strip()  # Engines may separate reasoning_content and content.
    if not value or '\n' in value or value.startswith(('{', '```', 'Answer:', 'Final answer:')):
        raise ValueError('Final answer must be one plain word or phrase')
    return value


def family_from_user(text):
    tasks = {
        'Solve this Chinese emoji puzzle.': 'zh_emoji',
        'Solve this Chinese picture puzzle.': 'zh_single_image_idiom',
        'Solve this Chinese visual wordplay puzzle.': 'zh_paired_homophone',
        'Solve this English rebus.': 'en_rebus',
        'Solve this English 2 Pics 1 Word puzzle.': 'en_2pics1word',
    }
    matched = [family for phrase, family in tasks.items() if phrase in text]
    if len(matched) != 1:
        raise ValueError('Unknown or ambiguous puzzle family; do not guess its prompt')
    return matched[0]


def request_key(row):
    """Fallback for Swift versions dropping custom IDs; never pair by line order."""
    users = [m['content'] for m in row.get('messages', []) if m['role'] == 'user']
    images = [PurePosixPath(str(p).replace('\\', '/')).name for p in row.get('images', [])]
    payload = dict(user=users[-1].strip() if users else '', images=images)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def resolve_images(row, root):
    result = deepcopy(row)
    root = Path(root).resolve()
    resolved = []
    for relative in row['images']:
        p = PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or '\\' in relative or not relative.startswith('image/'):
            raise ValueError('Expected safe dataset-relative image path')
        absolute = (root / relative).resolve()
        if root not in absolute.parents or not absolute.is_file():
            raise FileNotFoundError(absolute)
        resolved.append(absolute.as_posix())
    result['images'] = resolved
    return result
