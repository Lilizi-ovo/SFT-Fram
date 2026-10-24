"""Resolve all user paths against the project, never the terminal directory."""

from __future__ import annotations

import json
import re
from pathlib import Path


class FramError(Exception):
    """An actionable command or environment error."""


def inside(base: Path, value: str, label: str) -> Path:
    candidate = (base / value).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise FramError(f"{label}必须位于 {base} 内：{value}")
    return candidate


def config_path(root: Path, name: str) -> Path:
    name = name.removeprefix("-")
    if not name.endswith(".sh"):
        name += ".sh"
    path = inside(root / "sh", name, "参数文件")
    if not path.is_file():
        raise FramError(f"参数文件不存在：{path}。请参照 docs/parameters.md 创建。")
    # Bash cannot reliably source a UTF-8 BOM or Windows line endings.
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise FramError(f"{path} 必须保存为 UTF-8 无 BOM、LF 换行。")
    return path


def has_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index
        if index_path.is_file():
            try:
                shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
                return bool(shards) and all((path / shard).is_file() for shard in shards)
            except (ValueError, KeyError, TypeError):
                return False
    return any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def checkpoint_path(root: Path, name: str, selector: str = "latest") -> tuple[Path, Path]:
    """Return (run directory, loadable full-weight checkpoint). No mtime ordering."""
    run = inside(root / "outputs", name.removeprefix("-"), "模型目录")
    if not run.is_dir():
        raise FramError(f"模型目录不存在：{run}")
    if selector != "latest":
        target = inside(run, selector, "checkpoint")
        if not has_weights(target):
            raise FramError(f"checkpoint 缺少完整模型配置或权重：{target}")
        return run, target
    if has_weights(run):
        return run, run
    checkpoints: list[tuple[int, Path]] = []
    for path in run.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and has_weights(path):
            # Also reject symlinks pointing outside this run.
            checkpoints.append((int(match[1]), inside(run, path.name, "checkpoint")))
    if not checkpoints:
        raise FramError(
            f"{run} 中未找到完整权重。需要 config.json 及模型权重，"
            "或 checkpoint-数字 子目录；仅有 ZeRO 分片/LoRA adapter 不能直接验证。"
            "旧版 Swift 的 v*-时间目录请直接写入模型目录参数。"
        )
    return run, max(checkpoints, key=lambda item: item[0])[1]


def run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.endswith("."):
        raise FramError("运行名称只能包含英文字母、数字、下划线、短横线和点，且不能以点结尾。")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise FramError(f"运行名称是 Windows 保留名称：{value}")
    return value
