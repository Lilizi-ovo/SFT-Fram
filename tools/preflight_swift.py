"""Check real ms-swift multimodal encoding on the training host, without weights.

Requires the same ms-swift 4.x, Transformers, model processor and image settings
as training. No GPU model weights are loaded. This is not a training run.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path

from prepare_swift_data import resolve_row, split_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--dataset-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-length', default=8192, type=int)
    p.add_argument('--limit', default=0, type=int, help='0 checks all records; a positive value is only a smoke check')
    args = p.parse_args()
    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '1024')
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(args.model, model_type='qwen3_5')
    template = get_template(processor, template_type='qwen3_5', max_length=args.max_length,
                            truncation_strategy='raise', template_backend='swift',
                            loss_scale='default+ignore_empty_think', enable_thinking=True,
                            preserve_thinking=True, add_non_thinking_prefix=False)
    template.set_mode('train')
    tokenizer = processor.tokenizer
    counts, errors, maximum = Counter(), [], 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.dataset_root / 'data.jsonl').open(encoding='utf-8') as f, args.output.open('w', encoding='utf-8', newline='\n') as out:
        for n, line in enumerate(f):
            if args.limit and n >= args.limit:
                break
            row = json.loads(line)
            try:
                target = row['messages'][-1]['content']
                reasoning, answer = split_target(target)
                request = resolve_row(row, args.dataset_root)
                request.pop('id', None)
                encoded = template.encode(request)
                ids, labels = encoded['input_ids'], encoded['labels']
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                assert len(ids) == len(labels) and len(ids) <= args.max_length
                supervised = [i for i, label in enumerate(labels) if label != -100]
                assert supervised and all(labels[i] == ids[i] for i in supervised)
                first = supervised[0]
                assert first > 0 and all(label == -100 for label in labels[:first])
                # The complete target must appear verbatim in the unmasked decode.
                learned = tokenizer.decode([labels[i] for i in supervised], skip_special_tokens=False)
                normalizer = tokenizer.backend_tokenizer.normalizer
                normalized_target = normalizer.normalize_str(target) if normalizer is not None else target
                normalized_answer = normalizer.normalize_str(answer) if normalizer is not None else answer
                assert learned.startswith(normalized_target), 'Reasoning or final-answer supervision changed'
                assert learned.count('<think>') == 1 and learned.count('</think>') == 1
                assert learned.endswith(normalized_answer) or learned.endswith(normalized_answer + '<|im_end|>') or learned.endswith(normalized_answer + '<|im_end|>\n')
                assert all(label != -100 for label in labels[first:]), 'Internal loss mask in the assistant target'
                for image_id in [248056, 248057]:
                    assert all(labels[i] == -100 for i, t in enumerate(ids) if t == image_id)
                scale = encoded.get('loss_scale')
                if scale is not None:
                    if hasattr(scale, 'tolist'):
                        scale = scale.tolist()
                    assert all(float(scale[i]) > 0 for i in supervised)
                info = dict(id=row['id'], ok=True, tokens=len(ids), prompt_masked_tokens=first,
                            supervised_tokens=len(supervised), reasoning_preserved=True, final_answer=answer)
                maximum = max(maximum, len(ids))
                counts['passed'] += 1
            except Exception as exc:
                info = dict(id=row['id'], ok=False, error=f'{type(exc).__name__}: {exc}')
                errors.append(info)
                counts['failed'] += 1
            out.write(json.dumps(info, ensure_ascii=False) + '\n')
            if (n + 1) % 100 == 0:
                out.flush()
                print(json.dumps(dict(checked=n+1, **counts)), flush=True)
    summary = dict(counts=counts, max_tokens=maximum, full_dataset_check=(args.limit == 0),
                   packages={k:version(k) for k in ['ms-swift', 'transformers', 'torch']},
                   image_max_token_num=os.environ['IMAGE_MAX_TOKEN_NUM'],
                   no_model_weights_loaded=True, errors=errors[:20])
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
