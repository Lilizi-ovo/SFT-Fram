"""Read a trusted Bash parameter file without parsing shell syntax in Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import FramError


@dataclass
class ShellConfig:
    args: list[str]
    env: dict[str, str]


def find_bash() -> str:
    override = os.environ.get("FRAM_BASH")
    if override:
        executable = shutil.which(override)
        if executable:
            return executable
        raise FramError(f"FRAM_BASH 不可执行：{override}")
    if os.name == "nt":
        # Avoid WindowsApps' bash.exe, which opens WSL with a different Python environment.
        git = shutil.which("git")
        if git:
            for candidate in (Path(git).parent.parent / "bin/bash.exe", Path(git).parent / "bash.exe"):
                if candidate.is_file():
                    return str(candidate)
    executable = shutil.which("bash")
    if executable and "windowsapps" not in executable.lower():
        return executable
    raise FramError("未找到 Bash。Linux 请安装 bash；Windows 本机预演可安装 Git Bash 或设置 FRAM_BASH。")


def base_environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["FRAM_ROOT"] = root.as_posix()
    env["FRAM_PYTHON"] = Path(sys.executable).as_posix()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep model/dataset/compiler caches portable too. Explicit user settings win.
    for name, folder in {
        "HF_HOME": "huggingface", "MODELSCOPE_CACHE": "modelscope",
        "SWIFT_CACHE_DIR": "swift", "TORCH_HOME": "torch",
        "TRITON_CACHE_DIR": "triton", "XDG_CACHE_HOME": "xdg",
    }.items():
        env.setdefault(name, (root / "cache" / folder).as_posix())
    return env


def load_config(root: Path, path: Path) -> ShellConfig:
    script = r'''
set -euo pipefail
set -a
source "$1" >&2
set +a
if [[ "$(declare -p SWIFT_ARGS 2>/dev/null)" != "declare -a "* ]]; then
    printf '%s\n' '参数文件必须定义 Bash 数组 SWIFT_ARGS=( --参数 值 ... )。' >&2
    exit 2
fi
exec "$2" "$3" "${SWIFT_ARGS[@]}"
'''
    try:
        result = subprocess.run(
            [find_bash(), "--noprofile", "--norc", "-c", script, "fram-config",
             path.as_posix(), Path(sys.executable).as_posix(),
             (Path(__file__).parent / "config_emit.py").as_posix()],
            cwd=root, env=base_environment(root), capture_output=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FramError("参数文件读取超过 30 秒。文件中只应设置变量和 SWIFT_ARGS。") from exc
    if result.returncode:
        raise FramError(f"参数文件读取失败（退出码 {result.returncode}）：\n{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        config = ShellConfig(payload["args"], payload["env"])
    except (ValueError, KeyError, TypeError) as exc:
        raise FramError("参数文件未返回有效配置。请勿在文件中执行 exit、exec 或训练命令。") from exc
    if not config.args:
        raise FramError("SWIFT_ARGS 不能为空。")
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    # Source files may export variables, but these two always identify this launcher.
    config.env["FRAM_ROOT"] = root.as_posix()
    config.env["FRAM_PYTHON"] = Path(sys.executable).as_posix()
    return config


def options(args: list[str]) -> dict[str, list[str]]:
    """Index --key value / --key=value; preserve argv itself for Swift."""
    result: dict[str, list[str]] = {}
    current: str | None = None
    for token in args:
        if token.startswith("--"):
            key, equal, value = token.partition("=")
            key = key.replace("-", "_").lstrip("_")
            if key in result:
                raise FramError(f"参数重复：--{key}")
            result[key] = [value] if equal else []
            current = key
        elif current is None:
            raise FramError(f"参数必须以 --参数名 开始：{token}")
        else:
            result[current].append(token)
    return result
