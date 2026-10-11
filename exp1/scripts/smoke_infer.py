"""P0.2 最小推理验证脚本。

目的为确认本机环境、依赖版本与模型权重三者可用，不做任何评测。
所用题目为自拟的算术题，不取自实验数据集，以保证本脚本不依赖任何数据文件。

用法示例
    python scripts/smoke_infer.py --model Qwen/Qwen3-0.6B --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SMOKE_QUESTION = (
    "A store sells pens at 3 yuan each and notebooks at 7 yuan each. "
    "Mia buys 4 pens and 5 notebooks. How much does she pay in total?"
)

SMOKE_ANSWER = "47"


def build_messages(question: str) -> list[dict[str, str]]:
    """构造对话消息。

    作为冒烟用途的最小模板，正式评测的 prompt 模板在 P1 阶段统一实现。
    """
    return [{"role": "user", "content": question}]


def describe_environment() -> dict[str, object]:
    """收集环境信息，用于判断失败原因是否出在依赖上。"""
    import transformers

    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        info["peft"] = None

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3 最小推理验证")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    args = parser.parse_args()

    env = describe_environment()
    print("环境信息")
    for key, value in env.items():
        print(f"  {key} = {value}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    print(f"\n加载模型 {args.model}，device 为 {device}，dtype 为 {dtype}")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(device).eval()
    load_seconds = time.perf_counter() - load_start
    print(f"加载完成，用时 {load_seconds:.1f} 秒")

    messages = build_messages(SMOKE_QUESTION)
    # enable_thinking 置为 False，与实验要求保持一致
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    print("\n输入")
    print(text)

    gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_seconds = time.perf_counter() - gen_start

    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    print("\n输出")
    print(completion)

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    print("\n统计")
    print(f"  生成 token 数 = {new_tokens}")
    print(f"  生成用时 = {gen_seconds:.1f} 秒")
    print(f"  生成速度 = {new_tokens / gen_seconds:.1f} token/s")
    print(f"  参考答案 = {SMOKE_ANSWER}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
