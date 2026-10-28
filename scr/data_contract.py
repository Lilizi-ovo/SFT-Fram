"""Reject local nonempty-thinking targets paired with incompatible SFT settings."""
import json
from pathlib import Path

from .paths import FramError


def validate_training_contract(root: Path, parsed: dict[str, list[str]]) -> dict:
    checked, thinking = 0, 0
    for value in parsed.get('dataset', []):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or path.suffix != '.jsonl':
            continue  # Remote IDs / exported caches are checked in their own preprocessing.
        with path.open(encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                    assistants = [m['content'] for m in row.get('messages', []) if m['role'] == 'assistant']
                    for content in assistants:
                        if not isinstance(content, str) or not content.startswith('<think>'):
                            continue
                        if content.count('<think>') != 1 or content.count('</think>') != 1:
                            raise ValueError('思考边界不完整或重复')
                        reason, answer = content[len('<think>'):].split('</think>', 1)
                        if reason.strip():
                            if not answer.strip():
                                raise ValueError('思考后没有最终答案')
                            thinking += 1
                    checked += 1
                except (ValueError, KeyError, TypeError) as exc:
                    raise FramError(f'训练数据格式错误 {path.name}:{n}：{exc}') from exc
    if thinking:
        required = {'template': ['qwen3_5'], 'enable_thinking': ['true'],
                    'add_non_thinking_prefix': ['false'], 'preserve_thinking': ['true']}
        wrong = [f'--{key} {" ".join(value)}' for key, value in required.items()
                 if parsed.get(key) != value]
        if parsed.get('loss_scale', ['default'])[0] not in {'default', 'default+ignore_empty_think', 'ignore_empty_think', 'default+ignore_think_prefix'}:
            wrong.append('--loss_scale default+ignore_empty_think')
        if wrong:
            raise FramError('数据含非空 <think>，当前 CoT 项目要求以下配置一致：' + '；'.join(wrong))
    return {'local_rows_checked': checked, 'nonempty_thinking_targets': thinking}
