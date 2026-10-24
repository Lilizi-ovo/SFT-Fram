"""Inspect the active interpreter after loading the shell environment."""

from __future__ import annotations

import json
import subprocess
import sys

from .paths import FramError


PROBE = r'''
import importlib.metadata as m, json, platform, sys
packages = {}
for name in ('ms-swift', 'torch', 'torchvision', 'transformers', 'deepspeed',
             'qwen-vl-utils', 'flash-linear-attention', 'causal-conv1d'):
    try:
        packages[name] = m.version(name)
    except m.PackageNotFoundError:
        packages[name] = None
info = {'python': platform.python_version(), 'platform': sys.platform,
        'executable': sys.executable, 'packages': packages, 'gpus': []}
try:
    import torch
    info['cuda'] = torch.version.cuda
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info['gpus'].append({'name': p.name, 'memory_gib': round(p.total_memory / 2**30, 1)})
except Exception as e:
    info['torch_error'] = str(e)
print(json.dumps(info))
'''


def inspect_runtime(env: dict[str, str]) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-c", PROBE], env=env, capture_output=True,
            encoding="utf-8", errors="replace", timeout=60, check=False,
        )
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        raise FramError("运行环境检查失败：Python/CUDA 初始化未成功或超过 60 秒。") from exc


def require_runtime(mode: str, env: dict[str, str], args: dict[str, list[str]]) -> dict:
    info = inspect_runtime(env)
    packages = info["packages"]
    if mode == "train" and info["platform"] != "linux":
        raise FramError("八卡分布式训练请在 Linux 服务器运行；Windows 本机可使用 --dry-run 检查配置。")
    if not packages["ms-swift"]:
        raise FramError("当前 Python 环境未安装 ms-swift。请在服务器安装 requirements.txt。")
    if not info["gpus"]:
        raise FramError(f"当前环境未检测到可用 CUDA GPU。{info.get('torch_error', '')}")
    try:
        processes = int(env.get("NPROC_PER_NODE", "1"))
    except ValueError as exc:
        raise FramError("NPROC_PER_NODE 必须是正整数。") from exc
    if not 1 <= processes <= len(info["gpus"]):
        raise FramError(f"NPROC_PER_NODE={processes}，但当前仅有 {len(info['gpus'])} 张可见 GPU。")
    if mode == "train" and args.get("deepspeed") and not packages["deepspeed"]:
        raise FramError("参数使用 DeepSpeed，但当前 Python 未安装 deepspeed。")
    return info
