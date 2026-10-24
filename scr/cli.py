"""Interactive and one-shot interfaces share exactly the same execution path."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import base_environment, find_bash
from .paths import FramError
from .runner import execute, make_plan
from .runtime import inspect_runtime


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="fram", description="Qwen3.5 全量 SFT / 验证启动器（ms-swift 4.x）")
    commands = result.add_subparsers(dest="command")
    train = commands.add_parser("train", help="使用 sh/<参数名>.sh 训练")
    train.add_argument("config", help="参数名，例如 full-SFT 或 -full-SFT")
    train.add_argument("--name", help="outputs 下的新目录名称；默认 YYYYMMDD，同日自动加序号")
    train.add_argument("--resume", help="outputs 下的运行名或具体 checkpoint 路径，保留优化器状态续训")
    train.add_argument("--dry-run", action="store_true", help="读取参数并打印命令，不加载模型")
    val = commands.add_parser("val", help="对 outputs 中的完整模型进行验证集推理")
    val.add_argument("model", help="outputs 下的运行目录，例如 20261019 或 -20261019")
    val.add_argument("config", help="sh 下的验证参数名，例如 val 或 -val")
    val.add_argument("--checkpoint", default="latest", help="latest（默认：最大步数）或 run 下的相对目录")
    val.add_argument("--dry-run", action="store_true")
    commands.add_parser("configs", help="列出 sh 下的参数文件")
    commands.add_parser("runs", help="列出 outputs 下的运行")
    commands.add_parser("doctor", help="检查 Python、Bash、依赖和可见 GPU")
    commands.add_parser("shell", help="打开交互终端")
    return result


def normalize(argv: list[str]) -> list[str]:
    args = list(argv)
    positions = (1,) if args and args[0] == "train" else (1, 2) if args and args[0] == "val" else ()
    for index in positions:
        if len(args) > index and args[index].startswith("-") and not args[index].startswith("--") and args[index] != "-h":
            args[index] = args[index][1:]
    return args


def dispatch(argv: list[str], root: Path = ROOT) -> int:
    args = parser().parse_args(normalize(argv))
    if args.command == "train":
        return execute(make_plan(root, "train", args.config, name=args.name, resume=args.resume), args.dry_run)
    if args.command == "val":
        return execute(make_plan(root, "val", args.config, model=args.model, checkpoint=args.checkpoint), args.dry_run)
    if args.command == "configs":
        paths = sorted((root / "sh").glob("*.sh"))
        print("\n".join(path.stem for path in paths) if paths else "sh 目录为空，请参照 docs/parameters.md 创建参数文件。")
    elif args.command == "runs":
        paths = sorted((root / "outputs").iterdir()) if (root / "outputs").is_dir() else []
        runs = [path for path in paths if path.is_dir()]
        for path in runs:
            status = "未记录"
            record_path = path / "run.json"
            if record_path.is_file():
                try:
                    status = json.loads(record_path.read_text(encoding="utf-8")).get("status", "未知")
                except (ValueError, OSError):
                    status = "记录无法读取"
            print(f"{path.name}\t{status}")
        if not runs:
            print("尚无训练输出。")
    elif args.command == "doctor":
        print(f"项目：{root}\nPython：{sys.executable}")
        try:
            print(f"Bash：{find_bash()}")
        except FramError as exc:
            print(f"Bash：{exc}")
        info = inspect_runtime(base_environment(root))
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("真实八卡训练在 Linux 上执行；依赖和 GPU 检测以迁移后的服务器为准。")
    elif args.command in ("shell", None):
        return shell(root)
    return 0


def shell(root: Path = ROOT) -> int:
    print(f"Fram | 项目目录：{root}\n输入 help 查看命令，exit 退出。")
    while True:
        try:
            line = input("fram> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n输入 exit 可退出。")
            continue
        if not line:
            continue
        if line in ("exit", "quit"):
            return 0
        if line in ("help", "?", "--help", "-h"):
            parser().print_help()
            print("\n示例：train -full-SFT --dry-run\n      val -20261019 -val --dry-run")
            continue
        try:
            words = shlex.split(line)
            if words and words[0] == "shell":
                print("已在交互终端中。")
                continue
            dispatch(words, root)
        except SystemExit:
            # argparse errors/help must not close the interactive terminal.
            pass
        except (FramError, OSError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n操作已取消。")


def main(argv: list[str] | None = None) -> int:
    # Windows terminal code pages must not break Chinese logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        return dispatch(sys.argv[1:] if argv is None else argv)
    except (FramError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
