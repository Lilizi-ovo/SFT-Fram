"""Score final answers by seen/unseen answer words, family and image overlap status."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from eval_protocol import final_answer, normalize_answer, request_key


def prediction_text(row):
    if isinstance(row.get('response'), str):
        return row['response']
    if row.get('choices'):
        return row['choices'][0]['message']['content']
    if isinstance(row.get('content'), str):
        return row['content']
    raise ValueError('No prediction text in response / choices.message.content / content')


def score(manifest, predictions, *, allow_partial=False):
    gold = {r['id']:r for r in manifest}
    if len(gold) != len(manifest):
        raise ValueError('Duplicate manifest ID')
    keys = {r['request_key']:r['id'] for r in manifest}
    if len(keys) != len(manifest):
        raise ValueError('Ambiguous request key')
    found, detail = {}, []
    for row in predictions:
        candidates = [row.get(k) for k in ['sample_id', 'custom_id', 'id'] if row.get(k) in gold]
        sample_id = candidates[0] if candidates else keys.get(request_key(row))
        if not sample_id:
            raise ValueError('Prediction cannot be mapped to a test ID; never align by row order')
        if sample_id in found:
            raise ValueError(f'Duplicate prediction ID: {sample_id}')
        try:
            answer = final_answer(prediction_text(row))
            error = None
        except ValueError as exc:
            answer, error = None, str(exc)
        expected = gold[sample_id]
        info = dict(id=sample_id, predicted_answer=answer, gold=expected['answer'], format_error=error,
                    correct_exact=answer == expected['answer'],
                    correct_normalized=answer is not None and normalize_answer(answer) == expected['normalized_answer'])
        found[sample_id] = info
    missing = sorted(set(gold)-set(found))
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} predictions missing; explicitly use --allow-partial for a coverage report')
    groups = defaultdict(list)
    for sample_id, row in gold.items():
        item = found.get(sample_id)
        detail.append(dict(**row, prediction=item))
        for key in ['overall', 'answer_exact/' + ('seen' if row['seen_exact'] else 'unseen'),
                    'answer_normalized/' + ('seen' if row['seen_normalized'] else 'unseen'),
                    'family/' + row['family'], 'family_answer_exact/' + row['family'] + '/' + ('seen' if row['seen_exact'] else 'unseen'),
                    'family_answer_normalized/' + row['family'] + '/' + ('seen' if row['seen_normalized'] else 'unseen'),
                    'image_overlap/' + row['image_overlap_status']]:
            groups[key].append(item)
    report = {}
    for group, rows in sorted(groups.items()):
        observed = [r for r in rows if r is not None]
        report[group] = dict(total=len(rows), predictions=len(observed), missing=len(rows)-len(observed),
            correct_exact=sum(r['correct_exact'] for r in observed), correct_normalized=sum(r['correct_normalized'] for r in observed),
            accuracy_exact=sum(r['correct_exact'] for r in observed)/len(rows) if observed else None,
            accuracy_normalized=sum(r['correct_normalized'] for r in observed)/len(rows) if observed else None,
            format_errors=sum(r['format_error'] is not None for r in observed), complete=len(observed)==len(rows))
    return dict(complete=not missing, missing_ids=missing, groups=report,
                accuracy_denominator='All required samples in each group; missing/invalid outputs count as incorrect. An unrun group has null accuracy.'), detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    report, detail = score([json.loads(l) for l in a.manifest.open(encoding='utf-8')],
                           [json.loads(l) for l in a.predictions.open(encoding='utf-8')], allow_partial=a.allow_partial)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with a.output.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['group', 'total', 'predictions', 'missing', 'correct_exact', 'correct_normalized',
                  'accuracy_exact', 'accuracy_normalized', 'format_errors', 'complete']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, values in report['groups'].items():
            writer.writerow(dict(group=group, **values))
    with a.output.with_suffix('.details.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for r in detail:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
