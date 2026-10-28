"""Build a CoT-mode evaluation view and frozen answer-word strata, without editing gold data."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from eval_protocol import family_from_user, gold_answer, normalize_answer, request_key, resolve_images


def read_rows(root):
    return [json.loads(line) for line in (root / 'data.jsonl').open(encoding='utf-8')]


def prepare(train_root, test_root, templates_path, output_dir, near_duplicate_decisions=None):
    train, test = read_rows(train_root), read_rows(test_root)
    templates = json.loads(templates_path.read_text(encoding='utf-8'))
    words = {gold_answer(r) for r in train}
    normalized = {normalize_answer(a) for a in words}
    train_ids = {r['id'] for r in train}
    assert not train_ids.intersection(r['id'] for r in test), 'Train/test ID overlap'
    near = {}
    if near_duplicate_decisions:
        for line in near_duplicate_decisions.open(encoding='utf-8'):
            d = json.loads(line)
            if not any(r['id'] in train_ids for r in d['train_refs']):
                continue
            if d['decision'] in ['duplicate', 'related_visual_variant', 'unresolved']:
                for r in d['test_refs']:
                    near.setdefault(r['id'], set()).add(d['decision'])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests, keys = [], set()
    paths = ['test.cot.labeled.absolute.jsonl', 'test.cot.requests.absolute.jsonl', 'eval_manifest.jsonl']
    files = [(output_dir / p).open('w', encoding='utf-8', newline='\n') for p in paths]
    try:
        for row in test:
            answer = gold_answer(row)
            user = next(m['content'] for m in row['messages'] if m['role'] == 'user')
            family = family_from_user(user)
            result = resolve_images(row, test_root)
            result['messages'][0] = dict(role='system', content=templates[family])
            suffix = 'Return only the Chinese answer.' if family.startswith('zh_') else 'Return only the English answer.'
            if user.count(suffix) != 1:
                raise ValueError('Unexpected answer-only output clause; review before adapting it')
            result['messages'][1]['content'] = user.replace(suffix,
                'Write the checkable explanation in English inside <think>...</think>, followed by the answer alone.')
            result['messages'][-1] = dict(role='assistant', content=answer)
            assert result['messages'][1]['content'].split('\n\n',1)[0] == user.split('\n\n',1)[0]
            assert result['messages'][1]['content'].count('<image>') == len(row['images'])
            key = request_key(result)
            if key in keys:
                raise ValueError('Ambiguous test request key')
            keys.add(key)
            statuses = near.get(row['id'], set())
            info = dict(id=row['id'], answer=answer, normalized_answer=normalize_answer(answer),
                family=family, seen_exact=answer in words, seen_normalized=normalize_answer(answer) in normalized,
                image_overlap_status=('confirmed_overlap' if statuses.intersection({'duplicate', 'related_visual_variant'}) else 'pending_review' if statuses else 'no_confirmed_overlap'),
                user_prompt_change='output contract only; hints, image order and puzzle content retained',
                request_key=key)
            files[0].write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
            request = deepcopy(result)
            request['messages'].pop()
            assert all(m['role'] != 'assistant' for m in request['messages'])
            files[1].write(json.dumps(request, ensure_ascii=False, separators=(',', ':')) + '\n')
            files[2].write(json.dumps(info, ensure_ascii=False) + '\n')
            manifests.append(info)
    finally:
        for f in files:
            f.close()
    summary = dict(train_records=len(train), test_records=len(test), train_unique_answers=len(words),
        seen_exact=sum(r['seen_exact'] for r in manifests), unseen_exact=sum(not r['seen_exact'] for r in manifests),
        seen_normalized=sum(r['seen_normalized'] for r in manifests), unseen_normalized=sum(not r['seen_normalized'] for r in manifests),
        normalized_protocol='NFKC + Unicode casefold + whitespace collapse; punctuation preserved',
        by_family={family:dict(samples=sum(r['family']==family for r in manifests),
            seen_exact=sum(r['family']==family and r['seen_exact'] for r in manifests),
            seen_normalized=sum(r['family']==family and r['seen_normalized'] for r in manifests)) for family in sorted({r['family'] for r in manifests})},
        image_overlap_status=Counter(r['image_overlap_status'] for r in manifests),
        train_sha256=hashlib.sha256((train_root/'data.jsonl').read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((test_root/'data.jsonl').read_bytes()).hexdigest(),
        predictions_available=False, accuracy=None)
    (output_dir / 'strata_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-root', required=True, type=Path)
    p.add_argument('--test-root', required=True, type=Path)
    p.add_argument('--templates', type=Path, default=Path(__file__).with_name('cot_prompt_templates.json'))
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--near-duplicate-decisions', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.train_root, a.test_root, a.templates, a.output_dir, a.near_duplicate_decisions), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
