"""Check thinking/data agreement and the seen/unseen evaluation protocol."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scr.data_contract import validate_training_contract
from scr.paths import FramError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from eval_protocol import final_answer, normalize_answer, request_key
from score_stratified import score


class ThinkingContractTests(unittest.TestCase):
    def test_reject_disabled_thinking_for_nonempty_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'train.jsonl'
            path.write_text(json.dumps({'messages':[{'role':'assistant','content':'<think>Visible clues derive this.</think>\n\nanswer'}]})+'\n', encoding='utf-8')
            settings = {'dataset':[str(path)], 'template':['qwen3_5'], 'enable_thinking':['false'],
                        'add_non_thinking_prefix':['false'], 'preserve_thinking':['true']}
            with self.assertRaisesRegex(FramError, '非空'):
                validate_training_contract(root, settings)
            settings['enable_thinking'] = ['true']
            self.assertEqual(validate_training_contract(root, settings)['nonempty_thinking_targets'], 1)


class EvaluationProtocolTests(unittest.TestCase):
    def manifest(self):
        return [dict(id='a', answer='Milk', normalized_answer='milk', seen_exact=True, seen_normalized=True,
                     family='en_2pics1word', image_overlap_status='no_confirmed_overlap', request_key='ka'),
                dict(id='b', answer='风雨同舟', normalized_answer='风雨同舟', seen_exact=False, seen_normalized=False,
                     family='zh_paired_homophone', image_overlap_status='pending_review', request_key='kb')]

    def test_answer_in_reasoning_does_not_count(self):
        report, _ = score(self.manifest(), [dict(id='b', response='<think>Gold 风雨同舟 is considered.</think>\n\nWrong'),
                                           dict(id='a', response='<think>Milk follows from the clues.</think>\n\nmilk')])
        self.assertTrue(report['complete'])
        self.assertEqual(report['groups']['answer_exact/seen']['accuracy_exact'], 0)
        self.assertEqual(report['groups']['answer_exact/seen']['accuracy_normalized'], 1)
        self.assertEqual(report['groups']['answer_exact/unseen']['accuracy_normalized'], 0)

    def test_missing_and_duplicate_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            score(self.manifest(), [dict(id='a', response='Milk')])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score(self.manifest(), [dict(id='a', response='Milk'), dict(id='a', response='Milk')])
        report, _ = score(self.manifest(), [], allow_partial=True)
        self.assertIsNone(report['groups']['overall']['accuracy_exact'])

    def test_truncated_thinking_is_format_failure(self):
        with self.assertRaises(ValueError):
            final_answer('<think>Unfinished even if answer appears')
        self.assertEqual(final_answer('prefilled thinking\n</think>\n\nMilk'), 'Milk')
        self.assertEqual(normalize_answer('ＭＩＬＫ  Tea'), 'milk tea')


if __name__ == '__main__':
    unittest.main()
