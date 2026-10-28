"""Exercise the launcher with real Bash and a fake Swift subprocess, never a GPU."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scr.cli import dispatch, normalize, shell
from scr.config import find_bash, load_config
from scr.paths import FramError, checkpoint_path, config_path, has_weights
from scr.runner import execute, make_plan, stop_process, write_record


try:
    BASH = find_bash()
except FramError:
    BASH = None

FAKE_SWIFT = r'''
import json, os, pathlib, sys
args = sys.argv[2:]
def value(key):
    return args[args.index(key) + 1]
print('fake Swift stdout', flush=True)
print('fake Swift stderr', file=sys.stderr, flush=True)
if os.environ.get('FRAM_TEST_FAIL'):
    sys.exit(7)
if sys.argv[1] == 'sft':
    target = pathlib.Path(value('--output_dir')) / 'checkpoint-10'
    target.mkdir()
    (target / 'config.json').write_text('{}')
    (target / 'model.safetensors').write_bytes(b'fake weights')
    (target / 'trainer_state.json').write_text('{}')
    (target / 'optimizer.pt').write_bytes(b'fake optimizer')
else:
    pathlib.Path(value('--result_path')).write_text(
        json.dumps({'response': 'answer', 'labels': 'answer'}) + '\n', encoding='utf-8')
'''


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fram test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "portable project"
        for folder in ("sh", "outputs", "data", "model"):
            (self.root / folder).mkdir(parents=True)

    def config(self, name="train", extra="", val=False):
        path = self.root / "sh" / f"{name}.sh"
        args = '--val_dataset "$FRAM_ROOT/data/val.jsonl"' if val else (
            '--model "$FRAM_ROOT/model/model with spaces" --dataset "$FRAM_ROOT/data/train.jsonl"')
        path.write_text(f"export NPROC_PER_NODE=1\nSWIFT_ARGS=({args})\n{extra}\n", encoding="utf-8", newline="\n")
        return path

    def weights(self, relative, sharded=False):
        target = self.root / "outputs" / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        if sharded:
            (target / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}), encoding="utf-8")
            for name in ("part-1.safetensors", "part-2.safetensors"):
                (target / name).write_bytes(b"dummy")
        else:
            (target / "model.safetensors").write_bytes(b"dummy")
        return target


class PathsTests(Fixture):
    def test_reject_path_escape_and_crlf(self):
        with self.assertRaises(FramError):
            config_path(self.root, "../../outside")
        path = self.root / "sh" / "crlf.sh"
        path.write_bytes(b"SWIFT_ARGS=()\r\n")
        with self.assertRaisesRegex(FramError, "LF"):
            config_path(self.root, "crlf")

    def test_select_largest_step_not_lexical_order(self):
        self.weights("run/checkpoint-9")
        expected = self.weights("run/checkpoint-100")
        self.weights("run/checkpoint-20")
        self.assertEqual(checkpoint_path(self.root, "-run")[1], expected)

    def test_incomplete_shards_are_not_selected(self):
        expected = self.weights("run/checkpoint-1")
        incomplete = self.weights("run/checkpoint-2", sharded=True)
        (incomplete / "part-2.safetensors").unlink()
        self.assertFalse(has_weights(incomplete))
        self.assertEqual(checkpoint_path(self.root, "run")[1], expected)

    def test_explicit_checkpoint_and_containment(self):
        first = self.weights("run/checkpoint-1")
        self.weights("run/checkpoint-2")
        self.assertEqual(checkpoint_path(self.root, "run", "checkpoint-1")[1], first)
        with self.assertRaises(FramError):
            checkpoint_path(self.root, "run", "../../outside")
        with self.assertRaises(FramError):
            checkpoint_path(self.root, "../outside")

    def test_missing_weights_and_adapter_only_fail(self):
        target = self.root / "outputs" / "adapter"
        target.mkdir()
        (target / "adapter_model.safetensors").write_bytes(b"adapter")
        with self.assertRaisesRegex(FramError, "完整权重"):
            checkpoint_path(self.root, "adapter")

    def test_legacy_single_dash_commands(self):
        self.assertEqual(normalize(["train", "-full-SFT", "--dry-run"]), ["train", "full-SFT", "--dry-run"])
        self.assertEqual(normalize(["val", "-20261019", "-val"]), ["val", "20261019", "val"])
        self.assertEqual(normalize(["train", "--help"]), ["train", "--help"])


@unittest.skipUnless(BASH, "Bash is required for shell parameter tests")
class WorkflowTests(Fixture):
    def test_shell_array_quotes_and_exported_environment(self):
        path = self.config(extra="CUSTOM_VALUE='hello world'\nSWIFT_ARGS+=(--system 'a quoted prompt')")
        loaded = load_config(self.root, path)
        self.assertEqual(loaded.env["CUSTOM_VALUE"], "hello world")
        self.assertEqual(loaded.args[1], (self.root / "model/model with spaces").as_posix())
        self.assertEqual(loaded.args[-1], "a quoted prompt")

    def test_scalar_and_broken_shell_fail(self):
        path = self.config(extra="SWIFT_ARGS='not an array'")
        with self.assertRaisesRegex(FramError, "数组"):
            load_config(self.root, path)
        path.write_text("SWIFT_ARGS=(\n", encoding="utf-8", newline="\n")
        with self.assertRaises(FramError):
            load_config(self.root, path)

    def test_dry_run_isolated_and_does_not_import_swift(self):
        self.config()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(execute(make_plan(self.root, "train", "train", name="dry-run"), True), 0)
        self.assertFalse((self.root / "outputs/dry-run").exists())

    def test_collision_and_controlled_arguments_rejected(self):
        self.config()
        (self.root / "outputs/existing").mkdir()
        with self.assertRaisesRegex(FramError, "已存在"):
            make_plan(self.root, "train", "train", name="existing")
        self.config(extra="SWIFT_ARGS+=(--output_dir somewhere)")
        with self.assertRaisesRegex(FramError, "output_dir"):
            make_plan(self.root, "train", "train")

    def test_defaults_are_full_multimodal_training(self):
        self.config()
        plan = make_plan(self.root, "train", "train")
        for key, expected in (("--tuner_type", "full"), ("--freeze_llm", "false"),
                              ("--freeze_vit", "false"), ("--freeze_aligner", "false")):
            self.assertEqual(plan.argv[plan.argv.index(key) + 1], expected)

    def test_validation_must_not_inherit_eight_replicas(self):
        self.weights("run")
        self.config("val", extra="export NPROC_PER_NODE=8", val=True)
        with self.assertRaisesRegex(FramError, "NPROC_PER_NODE=1"):
            make_plan(self.root, "val", "val", model="run")

    def test_validation_does_not_start_torchrun(self):
        self.weights("run")
        self.config("val", val=True)
        plan = make_plan(self.root, "val", "val", model="run")
        self.assertNotIn("NPROC_PER_NODE", plan.env)
        self.assertNotIn("NNODES", plan.env)

    def fake_swift_env(self):
        fake = self.root / "fake modules"
        package = fake / "swift/cli"
        package.mkdir(parents=True)
        for path in (fake / "swift/__init__.py", package / "__init__.py"):
            path.write_text("", encoding="utf-8")
        (package / "main.py").write_text(FAKE_SWIFT, encoding="utf-8")
        return {"PYTHONPATH": str(fake), "FRAM_TEST_SECRET": "not-for-run-json"}

    def test_training_resume_validation_and_logs_with_real_subprocess(self):
        self.config()
        self.config("val", val=True)
        runtime = {"packages": {"ms-swift": "test"}}
        with patch.dict(os.environ, self.fake_swift_env()), patch("scr.runner.require_runtime", return_value=runtime), contextlib.redirect_stdout(io.StringIO()):
            train = make_plan(self.root, "train", "train", name="integration")
            self.assertEqual(execute(train), 0)
            record = json.loads((train.directory / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")
            self.assertNotIn("not-for-run-json", json.dumps(record))
            log = (train.directory / "console.log").read_text(encoding="utf-8")
            self.assertIn("stdout", log)
            self.assertIn("stderr", log)
            self.assertTrue((train.directory / "parameters.sh").is_file())
            resumed = make_plan(self.root, "train", "train", resume="integration", name="resumed")
            self.assertIn("--resume_from_checkpoint", resumed.argv)
            self.assertNotEqual(resumed.directory, train.directory)
            val = make_plan(self.root, "val", "val", model="integration")
            self.assertEqual(val.checkpoint.name, "checkpoint-10")
            self.assertEqual(execute(val), 0)
            self.assertTrue((val.directory / "predictions.jsonl").is_file())
            next_val = make_plan(self.root, "val", "val", model="integration")
            self.assertNotEqual(next_val.directory, val.directory)

    def test_failed_backend_exit_code_is_preserved(self):
        self.config(extra="export FRAM_TEST_FAIL=1")
        with patch.dict(os.environ, self.fake_swift_env()), patch("scr.runner.require_runtime", return_value={"packages": {}}), contextlib.redirect_stdout(io.StringIO()):
            plan = make_plan(self.root, "train", "train", name="failure")
            self.assertEqual(execute(plan), 7)
        record = json.loads((plan.directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["exit_code"]), ("failed", 7))

    def test_validation_automatically_scores_frozen_strata(self):
        self.weights("run")
        self.config("val", val=True, extra='export FRAM_EVAL_MANIFEST="$FRAM_ROOT/data/manifest.jsonl"')
        manifest = dict(id="a", answer="answer", normalized_answer="answer", seen_exact=False,
                        seen_normalized=False, family="en_rebus", image_overlap_status="no_confirmed_overlap", request_key="ka")
        (self.root / "data/manifest.jsonl").write_text(json.dumps(manifest)+"\n", encoding="utf-8")
        target = self.root / "tools"
        target.mkdir()
        for name in ("score_stratified.py", "eval_protocol.py"):
            shutil.copy2(Path(__file__).resolve().parents[1] / "tools" / name, target / name)
        env = self.fake_swift_env()
        fake = Path(env["PYTHONPATH"]) / "swift/cli/main.py"
        fake.write_text(FAKE_SWIFT.replace("{'response': 'answer', 'labels': 'answer'}",
            "{'id': 'a', 'response': '<think>Check.</think>\\n\\nanswer', 'labels': 'answer'}"), encoding="utf-8")
        with patch.dict(os.environ, env), patch("scr.runner.require_runtime", return_value={"packages": {}}), contextlib.redirect_stdout(io.StringIO()):
            plan = make_plan(self.root, "val", "val", model="run")
            self.assertEqual(execute(plan), 0)
        report = json.loads((plan.directory / "stratified_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["groups"]["answer_exact/unseen"]["accuracy_exact"], 1)
        self.assertTrue((plan.directory / "eval_manifest.jsonl").is_file())
        self.assertTrue((plan.directory / "stratified_metrics.csv").is_file())

    def test_interrupt_records_130_and_stops_child(self):
        self.config()
        plan = make_plan(self.root, "train", "train", name="interrupted")
        plan.argv = [sys.executable, "-c", "import time; time.sleep(30)"]
        def interrupt_running(path, record):
            if record["status"] == "running":
                raise KeyboardInterrupt
            write_record(path, record)
        with patch("scr.runner.require_runtime", return_value={"packages": {}}), patch("scr.runner.write_record", side_effect=interrupt_running), patch("scr.runner.stop_process", wraps=stop_process) as stop, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(execute(plan), 130)
        self.assertIsNotNone(stop.call_args.args[0].poll())
        record = json.loads((plan.directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "interrupted")

    def test_sigterm_restores_handler_and_returns_143(self):
        self.config()
        plan = make_plan(self.root, "train", "train", name="terminated")
        plan.argv = [sys.executable, "-c", "import time; time.sleep(30)"]
        previous = signal.getsignal(signal.SIGTERM)
        def terminate_running(path, record):
            if record["status"] == "running":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            write_record(path, record)
        with patch("scr.runner.require_runtime", return_value={"packages": {}}), patch("scr.runner.write_record", side_effect=terminate_running), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(execute(plan), 143)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        record = json.loads((plan.directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "interrupted")

    def test_missing_model_and_missing_config_have_nonfatal_shell_errors(self):
        self.config("val", val=True)
        with patch("builtins.input", side_effect=["train -missing", "val -missing -val", "exit"]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(shell(self.root), 0)
        self.assertIn("参数文件不存在", errors.getvalue())
        self.assertIn("模型目录不存在", errors.getvalue())

    def test_documented_examples_load_without_edits(self):
        doc = (Path(__file__).resolve().parents[1] / "docs/parameters.md").read_text(encoding="utf-8")
        blocks = [block.split("```", 1)[0] for block in doc.split("```bash\n")[1:]]
        self.assertEqual(len(blocks), 2)
        for name, block in zip(("full-SFT", "val"), blocks):
            path = self.root / "sh" / f"{name}.sh"
            path.write_text(block, encoding="utf-8", newline="\n")
        self.weights("20261019/checkpoint-10")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dispatch(["train", "-full-SFT", "--dry-run"], self.root), 0)
            self.assertEqual(dispatch(["val", "-20261019", "-val", "--dry-run"], self.root), 0)


if __name__ == "__main__":
    unittest.main()
