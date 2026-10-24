"""Plan Swift commands, own output locations, record runs, and stop process groups."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import load_config, options
from .paths import FramError, checkpoint_path, config_path, run_name
from .runtime import require_runtime


@dataclass
class Plan:
    root: Path
    mode: str
    config: Path
    directory: Path
    argv: list[str]
    env: dict[str, str]
    checkpoint: Path | None = None


class StopRequested(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


def request_stop(signum: int, _frame) -> None:
    raise StopRequested(signum)


def fresh_directory(base: Path, name: str) -> Path:
    candidate = base / name
    suffix = 2
    while candidate.exists():
        candidate = base / f"{name}-{suffix:02d}"
        suffix += 1
    return candidate


def make_plan(root: Path, mode: str, config_name: str, *, model: str | None = None,
              name: str | None = None, checkpoint: str = "latest", resume: str | None = None) -> Plan:
    path = config_path(root, config_name)
    shell = load_config(root, path)
    args = list(shell.args)
    parsed = options(args)
    if "train_type" in parsed:
        raise FramError("ms-swift 4.x 请使用 --tuner_type，不能使用旧参数 --train_type。")
    controlled = {"output_dir", "add_version", "result_path", "resume_from_checkpoint"}
    if mode == "val":
        controlled |= {"model", "adapters"}
    conflicts = controlled.intersection(parsed)
    if conflicts:
        raise FramError("这些参数由 Fram 管理，请从 SWIFT_ARGS 中移除：" + ", ".join(sorted(conflicts)))
    env = shell.env
    if env.get("NNODES", "1") != "1" or env.get("NODE_RANK", "0") != "0":
        raise FramError("本框架面向单机多卡，请将 NNODES=1、NODE_RANK=0；多机需要单独设计输出协调。")
    selected = None
    if mode == "train":
        if not parsed.get("model") or not (parsed.get("dataset") or parsed.get("cached_dataset")):
            raise FramError("训练配置必须包含 --model 和 --dataset（或 --cached_dataset）。")
        defaults = {"tuner_type": "full", "freeze_llm": "false", "freeze_vit": "false", "freeze_aligner": "false"}
        for key, value in defaults.items():
            if key not in parsed:
                args.extend([f"--{key}", value])
        if parsed.get("tuner_type", ["full"]) != ["full"]:
            raise FramError("本框架的训练/验证流程要求 --tuner_type full，以完整权重进行验证。")
        env.setdefault("NPROC_PER_NODE", "8")
        directory = root / "outputs" / run_name(name) if name else fresh_directory(
            root / "outputs", datetime.now().strftime("%Y%m%d"))
        if directory.exists():
            raise FramError(f"输出目录已存在：{directory}。请选择新的 --name；续训也会创建独立输出目录。")
        args.extend(["--output_dir", directory.as_posix(), "--add_version", "false"])
        if resume:
            _, selected = checkpoint_path(root, resume)
            if not (selected / "trainer_state.json").is_file() or not any(selected.glob("global_step*")):
                # Non-DeepSpeed runs can keep standard optimizer files instead.
                if not ((selected / "trainer_state.json").is_file() and (selected / "optimizer.pt").is_file()):
                    raise FramError("续训 checkpoint 缺少 trainer_state 或优化器状态；仅有模型权重不能完整恢复训练。")
            args.extend(["--resume_from_checkpoint", selected.as_posix()])
        method = "sft"
    elif mode == "val":
        if not model:
            raise FramError("验证需要 outputs 下的模型目录名称。")
        if not (parsed.get("val_dataset") or parsed.get("cached_val_dataset")):
            raise FramError("验证配置必须包含 --val_dataset（或 --cached_val_dataset）。")
        run, selected = checkpoint_path(root, model, checkpoint)
        directory = fresh_directory(run / "eval", datetime.now().strftime("%Y%m%d-%H%M%S"))
        # Training process settings must never accidentally spawn eight inference replicas.
        env.setdefault("NPROC_PER_NODE", "1")
        if env["NPROC_PER_NODE"] != "1":
            raise FramError("验证配置请设置 NPROC_PER_NODE=1；多 GPU 推理可使用 device_map 或 vLLM tensor parallel。")
        # Swift treats even NPROC_PER_NODE=1 as a request for torchrun. A plain
        # inference process avoids reserving a rendezvous port alongside training.
        for key in ("NPROC_PER_NODE", "NNODES", "NODE_RANK"):
            env.pop(key, None)
        args.extend(["--model", selected.as_posix(), "--result_path", (directory / "predictions.jsonl").as_posix()])
        if "load_data_args" not in parsed:
            args.extend(["--load_data_args", "false"])
        method = "infer"
    else:
        raise FramError(f"未知操作：{mode}")
    return Plan(root, mode, path, directory, [sys.executable, "-m", "swift.cli.main", method, *args], env, selected)


def show_plan(plan: Plan) -> None:
    print(f"操作：{plan.mode}\n参数：{plan.config}\n输出：{plan.directory}")
    if plan.checkpoint:
        print(f"Checkpoint：{plan.checkpoint}")
    for key in ("CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "MASTER_PORT"):
        if key in plan.env:
            print(f"{key}={plan.env[key]}")
    print("命令：" + shlex.join(plan.argv))


def write_record(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        process.wait(timeout=15)
    else:
        # A parent may exit before its torchrun workers; stop the whole group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)


def execute(plan: Plan, dry_run: bool = False) -> int:
    show_plan(plan)
    if dry_run:
        print("预演完成：未启动 Swift，未创建输出目录。参数文件已由 Bash 读取。")
        return 0
    runtime = require_runtime(plan.mode, plan.env, options(plan.argv[4:]))
    # Atomic mkdir refuses collisions, including a race with another launcher.
    try:
        plan.directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FramError(f"输出目录已被其他任务创建：{plan.directory}") from exc
    record = {
        "status": "starting", "mode": plan.mode,
        "started_at": datetime.now().astimezone().isoformat(),
        "config": str(plan.config.relative_to(plan.root)),
        "argv": plan.argv, "checkpoint": str(plan.checkpoint) if plan.checkpoint else None,
        "runtime": runtime,
        "environment": {key: plan.env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "MASTER_PORT", "IMAGE_MAX_TOKEN_NUM",
            "HF_HOME", "MODELSCOPE_CACHE", "SWIFT_CACHE_DIR", "TORCH_HOME", "TRITON_CACHE_DIR",
        ) if key in plan.env},
    }
    # Do not persist the complete environment: it may contain hub credentials.
    record_path = plan.directory / "run.json"
    write_record(record_path, record)
    process = None
    exit_code = 1
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        (plan.directory / "parameters.sh").write_bytes(plan.config.read_bytes())
        (plan.directory / "requirements.freeze.txt").write_text(
            "\n".join(f"{key}=={value}" for key, value in runtime["packages"].items() if value) + "\n",
            encoding="utf-8",
        )
        kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        with (plan.directory / "console.log").open("w", encoding="utf-8") as log:
            log.write(shlex.join(plan.argv) + "\n")
            log.flush()
            process = subprocess.Popen(plan.argv, cwd=plan.root, env=plan.env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", **kwargs)
            record.update(status="running", pid=process.pid)
            write_record(record_path, record)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = process.wait()
        if exit_code == 0 and plan.mode == "train":
            _, selected = checkpoint_path(plan.root, str(plan.directory.relative_to(plan.root / "outputs")))
            record["last_checkpoint"] = str(selected.relative_to(plan.root))
        elif exit_code == 0:
            predictions = plan.directory / "predictions.jsonl"
            if not predictions.is_file() or predictions.stat().st_size == 0:
                raise FramError("Swift 返回成功，但未生成非空 predictions.jsonl，请检查验证参数和 console.log。")
        record["status"] = "completed" if exit_code == 0 else "failed"
    except (KeyboardInterrupt, StopRequested) as exc:
        record["status"] = "interrupted"
        exit_code = 128 + (exc.signum if isinstance(exc, StopRequested) else signal.SIGINT)
        print("\n正在停止本次训练/验证的进程组……", flush=True)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        exit_code = 1
        print(f"运行失败：{exc}", file=sys.stderr)
    finally:
        try:
            if process is not None:
                stop_process(process)
                if process.stdout is not None:
                    process.stdout.close()
            record.update(exit_code=exit_code, finished_at=datetime.now().astimezone().isoformat())
            write_record(record_path, record)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"状态：{record['status']}（退出码 {exit_code}）\n日志：{plan.directory / 'console.log'}")
    return exit_code
