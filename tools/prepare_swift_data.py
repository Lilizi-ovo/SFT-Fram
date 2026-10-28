"""Resolve image paths on the training host without editing the portable dataset."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath


def split_target(content):
    if not content.startswith('<think>\n') or content.count('<think>') != 1 or content.count('</think>') != 1:
        raise ValueError('Expected exactly one native thinking block')
    reasoning, final = content[len('<think>\n'):].split('\n</think>\n\n', 1)
    if not reasoning.strip() or reasoning != reasoning.strip() or not final.strip() or final != final.strip():
        raise ValueError('Reasoning and answer must be nonempty and trimmed')
    if any(t in final for t in ['<think>', '</think>', '\n', '{', '}']):
        raise ValueError('Expected one plain final answer')
    return reasoning, final


def resolve_row(row, root):
    root = Path(root).resolve()
    result = deepcopy(row)
    if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
        raise ValueError('Expected one labelled system/user/assistant sample')
    split_target(row['messages'][-1]['content'])
    if row['messages'][1]['content'].count('<image>') != len(row['images']):
        raise ValueError('Image placeholders do not match images')
    images = []
    for p in row['images']:
        relative = PurePosixPath(p)
        if relative.is_absolute() or '..' in relative.parts or '\\' in p or not p.startswith('image/'):
            raise ValueError(f'Unsafe relative image path: {p}')
        absolute = (root / p).resolve()
        if root not in absolute.parents or not absolute.is_file():
            raise FileNotFoundError(absolute)
        images.append(absolute.as_posix())
    result['images'] = images
    return result


def extract_final_answer(text, *, raw_thinking_generation=False):
    """For metrics use final content, never search the reasoning for a gold answer.

    Raw generate() tokens may omit the opening <think> already in the prompt.
    An unfinished raw thinking generation must fail rather than count as an answer.
    If a serving engine separates reasoning_content and content, pass content only.
    """
    if text.count('</think>') > 1 or text.count('<think>') > 1:
        raise ValueError('Repeated thinking boundaries')
    if '</think>' in text:
        final = text.split('</think>', 1)[1].strip()
    elif raw_thinking_generation or '<think>' in text:
        raise ValueError('Thinking did not finish')
    else:
        final = text.strip()
    if not final or '\n' in final or '<think>' in final or '</think>' in final:
        raise ValueError('Missing or malformed final answer')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = (args.dataset_root / 'data.jsonl').resolve()
    dest = args.output.resolve()
    if source == dest:
        raise ValueError('Output must be a separate host-local cache, not data.jsonl')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    count, ids = 0, set()
    with source.open(encoding='utf-8') as f, temporary.open('w', encoding='utf-8', newline='\n') as out:
        for line in f:
            row = json.loads(line)
            if row['id'] in ids:
                raise ValueError(f"Duplicate ID: {row['id']}")
            ids.add(row['id'])
            out.write(json.dumps(resolve_row(row, args.dataset_root), ensure_ascii=False, separators=(',', ':')) + '\n')
            count += 1
    temporary.replace(dest)
    print(json.dumps(dict(records=count, output=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()), ensure_ascii=False))


if __name__ == '__main__':
    main()
