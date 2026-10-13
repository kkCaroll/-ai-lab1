"""实验公共工具模块。

本模块为 `src` 下各模块共用的底座，提供三态报告机制、推理精度判定、环境
快照采集、随机种子设置、文件哈希计算与 JSON/JSONL 写入。

torch 一律在函数体内导入，不在模块顶层导入。原因是 `src/extract.py` 需要
引用本模块的状态常量，而该模块要求不依赖 torch，以便在没有 torch 的机器
上以毫秒级运行单元测试。若此处顶层导入 torch，依赖会顺着引用链被拖入抽
取模块，该性质即告失效。在顶部补一行 import torch 之前请先读到这里。

用法示例
    from src import utils

    checks = []
    utils.report(checks, "数据行数", utils.OK, "1319")
    raise SystemExit(utils.summarize(checks))
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------- 报告机制 ------------------
# 三个状态常量
OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class Check:
    """一条检查记录。name 为检查对象，detail 为观测到的取值。"""

    name: str
    status: str
    detail: str = ""


def report(checks: list[Check], name: str, status: str, detail: str = "") -> None:
    """追加一条检查记录并即时打印。"""
    checks.append(Check(name, status, detail))
    suffix = f"  {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}", flush=True)


def section(title: str) -> None:
    """打印分节标题。"""
    print(f"\n===== {title} =====", flush=True)


def summarize(checks: list[Check]) -> int:
    """打印汇总并返回退出码。"""
    section("汇总")
    counts = {status: sum(1 for item in checks if item.status == status) for status in (OK, WARN, FAIL)}
    for name, status in (
        ("失败", FAIL),
        ("警告", WARN),
    ):
        for item in checks:
            if item.status == status:
                print(f"{name}  {item.name}  {item.detail}")
    print(f"\n共计 {len(checks)} 项，OK {counts[OK]}，WARN {counts[WARN]}，FAIL {counts[FAIL]}")
    return 1 if counts[FAIL] else 0


# ----------------- 工具函数 ------------------

# 环境快照需要记录版本的库，与 requirements.txt 的固定清单一致
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "peft",
    "datasets",
    "accelerate",
    "safetensors",
    "huggingface_hub",
)


def preferred_dtype() -> torch.dtype:
    """按硬件算力选择推理精度。

    算力主版本号 8 及以上为 Ampere 及以后，原生支持 bfloat16。Volta 与 Turing
    的主版本号分别为 7 与 7.5，bfloat16 只能软件模拟，故取 float16。无 CUDA
    时取 float32，float16 在处理器上大量算子没有实现。
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32

    major = torch.cuda.get_device_capability(0)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def set_seed(seed: int) -> None:
    """设置随机种子。

    覆盖 random、numpy 与 torch 三处。torch.manual_seed 已同时为处理器与全部
    设备播种，故不再单独调用 manual_seed_all。种子不保证逐位可复现，原因与
    影响见 NOTE/04-解码策略.md。
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _package_version(name: str) -> str | None:
    """取已安装库的版本，未安装时返回 None。

    本机与容器所装库的集合不同，缺一个库不应让整个快照函数失败。
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, object]:
    """采集静态环境信息，返回字典，不落盘。

    显存峰值属于运行期测量量而非静态环境，由评测入口在生成结束后并入，
    故本函数不含该项，可在任何时刻调用。
    """
    import torch

    snapshot: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for name in TRACKED_PACKAGES:
        snapshot[f"{name}_version"] = _package_version(name)
    return snapshot


def parameter_counts(model) -> dict[str, object]:
    """统计参数量与可训练量，返回字典。

    可训练量按 requires_grad 判定，故须在方法装配之后调用。占比是风险 R15 的直接
    证据，也是记录表要填的一列，故与两项绝对量一并返回。

    本函数不导入 torch，只依赖模型对象的 parameters 迭代协议，故不破坏本模块顶部
    不导入 torch 的约定。
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": round(trainable / total, 6) if total else 0.0,
    }


def sha256_of(path: Path) -> str:
    """返回文件的 SHA256 校验和。

    分块读取而非一次读完，因为同一函数将来要校验 1.15 GiB 的权重文件。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: dict, path: Path) -> None:
    """将数据写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_jsonl(data: list[dict], path: Path) -> None:
    """将数据写入 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
