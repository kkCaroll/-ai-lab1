"""P0.4 DLC 冒烟任务脚本。

目的为在付出训练成本之前确认 DLC 侧运行环境可用。检查分五段，可分段执行

    env      解释器、驱动与 CUDA 运行时
    deps     实验所需三方库是否齐备
    vars     关键环境变量与平台注入变量
    mount    代码集与存储挂载是否可见、是否可写
    forward  加载 Qwen3-0.6B 并完成一次贪心生成

任一检查项判为 FAIL 即以非零码退出，便于在任务列表与日志中直接判断成败。
所用题目为自拟算术题，不取自实验数据集，故本脚本不依赖任何数据文件。

用法示例
    python /mnt/code/exp1/scripts/smoke_dlc.py
    python /mnt/code/exp1/scripts/smoke_dlc.py --stages env,mount --probe-write /mnt/data/exp1

注意 torch 与 transformers 均延迟导入，因此 env、vars 与 mount 三段在依赖缺失时
仍可运行并给出报告，便于区分环境问题与依赖问题。

精度选择按硬件算力判定，算力 8.0 及以上取 bfloat16，否则取 float16。不采用
torch.cuda.is_bf16_supported，该函数默认把软件模拟计入支持范围，对任意 CUDA 11
以上的设备均返回 True，无法区分原生支持与模拟实现。
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

STAGES = ("env", "deps", "vars", "mount", "forward")
DEFAULT_STAGES = ",".join(STAGES)

# 镜像自带之外，实验还需要的库
REQUIRED_PACKAGES = (
    "transformers",
    "tokenizers",
    "peft",
    "datasets",
    "accelerate",
    "safetensors",
    "huggingface_hub",
)

# 需要显式确认的变量。HuggingFace 相关变量为空时容器将直连官方站点，境内通常不可达
WATCHED_VARS = (
    "HF_ENDPOINT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "PYTHONUNBUFFERED",
    "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)

# 由平台注入的变量，用于确认注入机制生效
INJECTED_VARS = (
    "JOB_NAME",
    "WORKER_COUNT",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "POD_NAME",
    "WORKSPACE_ID",
)

# 代码集与存储挂载的候选路径，前者见任务配置的代码配置挂载路径，后者固定为 /mnt/data
DEFAULT_MOUNTS = ("/mnt/code", "/mnt/data")

SMOKE_QUESTION = (
    "A store sells pens at 3 yuan each and notebooks at 7 yuan each. "
    "Mia buys 4 pens and 5 notebooks. How much does she pay in total?"
)

SMOKE_ANSWER = "47"


@dataclass
class Check:
    """一条检查记录。name 为检查对象，detail 为观测到的取值。"""

    name: str
    status: str
    detail: str = ""


def report(checks: list[Check], name: str, status: str, detail: str = "") -> None:
    checks.append(Check(name, status, detail))
    suffix = f"  {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}", flush=True)


def section(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def stage_environment(checks: list[Check]) -> None:
    """检查解释器、NVIDIA 驱动与 CUDA 运行时。"""
    section("环境")
    report(checks, "python", OK, f"{platform.python_version()}  可执行文件 {sys.executable}")
    report(checks, "platform", OK, platform.platform())
    if hasattr(os, "uname"):
        uname = os.uname()
        report(checks, "uname", OK, f"{uname.sysname} {uname.release} {uname.machine}")

    # nvidia-smi 由宿主驱动提供，与镜像内的 CUDA 运行时相互独立，故单独确认
    smi = shutil.which("nvidia-smi")
    if smi is None:
        report(checks, "nvidia-smi", WARN, "命令不存在，无法读取驱动版本")
    else:
        try:
            done = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = (done.stdout.strip() or done.stderr.strip()).replace("\n", " | ")
            report(checks, "nvidia-smi", OK if done.returncode == 0 else WARN, output)
        except (OSError, subprocess.SubprocessError) as exc:
            report(checks, "nvidia-smi", WARN, f"调用失败 {exc!r}")

    try:
        import torch
    except ImportError as exc:
        report(checks, "torch", FAIL, f"未安装 {exc}")
        return

    report(checks, "torch", OK, torch.__version__)
    cuda_build = torch.version.cuda
    report(
        checks,
        "torch 的 CUDA 构建",
        OK if cuda_build else FAIL,
        str(cuda_build) if cuda_build else "为 CPU 构建，不可用于训练",
    )

    available = torch.cuda.is_available()
    report(checks, "CUDA 可用", OK if available else FAIL, str(available))
    if not available:
        return

    count = torch.cuda.device_count()
    report(checks, "可见 GPU 数", OK, str(count))
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        report(
            checks,
            f"GPU {index}",
            OK,
            f"{props.name}  显存 {props.total_memory / 1024**3:.1f} GiB  "
            f"算力 {props.major}.{props.minor}",
        )
    # is_bf16_supported 把软件模拟也计入支持范围，对任意 CUDA 11 以上的设备均返回
    # True，不足以判定硬件是否原生支持，故以算力为准
    capability = torch.cuda.get_device_capability(0)
    native_bf16 = capability[0] >= 8
    report(
        checks,
        "bf16 原生支持",
        OK if native_bf16 else WARN,
        str(native_bf16)
        + ("" if native_bf16 else "，算力低于 8.0，宜改用 float16"),
    )


def stage_dependencies(checks: list[Check]) -> None:
    """确认三方库齐备，缺失项即为三方库配置需要补的清单。"""
    section("依赖")
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            report(checks, name, FAIL, f"未安装 {exc}")
            continue
        report(checks, name, OK, str(getattr(module, "__version__", "版本未知")))


def stage_variables(checks: list[Check]) -> None:
    """确认关键环境变量与平台注入变量。"""
    section("环境变量")
    for name in WATCHED_VARS:
        value = os.environ.get(name)
        report(checks, name, OK if value else WARN, value or "未设置")

    present = {name: os.environ[name] for name in INJECTED_VARS if name in os.environ}
    detail = "  ".join(f"{name}={value}" for name, value in present.items())
    report(checks, "平台注入变量", OK if present else WARN, detail or "未发现")


def probe_write(checks: list[Check], directory: str) -> None:
    """在目标目录写入并回读一个小文件，以实际结果判断可写性。

    OSS 不支持对已存在文件的覆盖写，故文件名取唯一值，重复执行不会相互干扰。
    """
    if not os.path.isdir(directory):
        report(checks, "写入探测", FAIL, f"目录不存在 {directory}")
        return

    path = os.path.join(directory, f"_smoke_{uuid.uuid4().hex}.txt")
    payload = f"dlc smoke probe {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except OSError as exc:
        report(checks, "写入探测", FAIL, f"写入失败 {exc}")
        return

    try:
        with open(path, encoding="utf-8") as handle:
            readback = handle.read()
        report(checks, "写入探测", OK if readback == payload else FAIL, f"写入并回读 {path}")
    except OSError as exc:
        report(checks, "写入探测", FAIL, f"回读失败 {exc}")

    try:
        os.remove(path)
        report(checks, "写入探测清理", OK, f"已删除 {path}")
    except OSError as exc:
        report(checks, "写入探测清理", WARN, f"删除失败 {exc}，需手动清理 {path}")


def stage_mounts(checks: list[Check], paths: list[str], probe_directory: str | None) -> None:
    """确认挂载点可见，并按需探测可写性。"""
    section("挂载")
    for path in paths:
        if not os.path.isdir(path):
            report(checks, path, WARN, "目录不存在，若未配置对应挂载则属预期")
            continue
        try:
            entries = sorted(os.listdir(path))
        except OSError as exc:
            report(checks, path, FAIL, f"无法列出 {exc}")
            continue

        preview = "  ".join(entries[:8]) + ("  ..." if len(entries) > 8 else "")
        report(checks, path, OK, f"可见 {len(entries)} 项  {preview}")

        statvfs = getattr(os, "statvfs", None)
        if statvfs is None:
            continue
        try:
            stat = statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            report(
                checks,
                f"{path} 容量",
                OK,
                f"总计 {total / 1024**3:.1f} GiB  可用 {free / 1024**3:.1f} GiB",
            )
        except (OSError, AttributeError):
            pass

    if probe_directory:
        probe_write(checks, probe_directory)


def preferred_dtype(torch) -> object:
    """按硬件算力选择推理精度。

    算力 8.0 为 Ampere 及以后，原生支持 bfloat16。Volta 与 Turing 算力分别为 7.0
    与 7.5，bfloat16 只能模拟，故取 float16。
    """
    major = torch.cuda.get_device_capability(0)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def stage_forward(checks: list[Check], model_id: str, max_new_tokens: int, dtype_name: str) -> None:
    """加载模型并完成一次贪心生成，确认权重可获取且 GPU 可完成前向。"""
    section("前向")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        report(checks, "导入推理依赖", FAIL, f"{exc}")
        return

    if not torch.cuda.is_available():
        report(checks, "前向", FAIL, "CUDA 不可用，不进行前向")
        return

    if dtype_name == "auto":
        dtype = preferred_dtype(torch)
        origin = "按硬件算力自动选择"
    else:
        dtype = getattr(torch, dtype_name)
        origin = "由命令行指定"
    report(checks, "推理精度", OK, f"{dtype}，{origin}")

    try:
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model = model.to("cuda").eval()
        load_seconds = time.perf_counter() - load_start
    except Exception as exc:
        report(checks, "加载模型", FAIL, f"{type(exc).__name__} {exc}")
        return

    report(checks, "加载模型", OK, f"{model_id}  用时 {load_seconds:.1f} 秒")

    # enable_thinking 置为 False，与实验要求保持一致
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": SMOKE_QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()
    generate_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generate_seconds = time.perf_counter() - generate_start

    new_tokens = int(outputs.shape[1] - inputs["input_ids"].shape[1])
    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    speed = new_tokens / generate_seconds if generate_seconds > 0 else 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print("输出如下")
    print(completion)
    report(checks, "生成", OK, f"{new_tokens} token  用时 {generate_seconds:.1f} 秒  {speed:.1f} token/s")
    report(checks, "显存峰值", OK, f"{peak:.2f} GiB")
    hit = SMOKE_ANSWER in completion
    report(
        checks,
        "答案正确性",
        OK if hit else WARN,
        f"参考答案 {SMOKE_ANSWER} 已出现在输出中"
        if hit
        else f"参考答案 {SMOKE_ANSWER} 未在输出中直接出现，需人工核对",
    )


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


def main() -> int:
    parser = argparse.ArgumentParser(description="DLC 冒烟任务")
    parser.add_argument("--stages", default=DEFAULT_STAGES, help=f"逗号分隔，可选 {','.join(STAGES)}")
    parser.add_argument("--mounts", nargs="*", default=list(DEFAULT_MOUNTS), help="待检查的挂载路径")
    parser.add_argument("--probe-write", default=None, help="向该目录写入探测文件以验证可写性")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--hf-endpoint", default=None, help="在导入 transformers 之前设置 HF_ENDPOINT")
    args = parser.parse_args()

    selected = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = [name for name in selected if name not in STAGES]
    if unknown:
        print(f"未知的检查段 {unknown}，可选 {list(STAGES)}", file=sys.stderr)
        return 2

    # HuggingFace 客户端在导入时读取该变量，故须早于任何三方库导入
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"已将 HF_ENDPOINT 设为 {args.hf_endpoint}", flush=True)

    checks: list[Check] = []
    if "env" in selected:
        stage_environment(checks)
    if "deps" in selected:
        stage_dependencies(checks)
    if "vars" in selected:
        stage_variables(checks)
    if "mount" in selected:
        stage_mounts(checks, args.mounts, args.probe_write)
    if "forward" in selected:
        stage_forward(checks, args.model, args.max_new_tokens, args.dtype)

    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
