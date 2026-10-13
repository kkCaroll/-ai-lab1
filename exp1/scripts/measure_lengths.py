"""量训练文本与评测提示的 token 长度分布。

最大序列长度须由实测分布定，不能凭经验取整。取值过小会截掉目标末尾的 #### 行，
使模型学不到收尾；取值过大则大部分批次的大部分位置是填充符，白白消耗显存与算力。
本脚本给出分位数与若干候选上限下的截断比例，作为定值的直接依据。

训练文本由三部分拼成，即评测提示、答案正文与结束标记。提示取自 src/data.py 的
render_prompt，与评测侧同源；答案正文取自 build_target_text，已剥离计算标注；结束
标记为 <|im_end|>。两部分分别量而不只量合计，是为了看出长度主要由哪一段贡献。

只加载分词器，不加载权重，故可在处理器上以数十秒跑完。

用法示例
    cd exp1
    python -m scripts.measure_lengths --data-dir ../data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import data
from src.utils import FAIL, OK, report, section, summarize

PERCENTILES = (50, 75, 90, 95, 99, 99.9)
CAPS = (512, 768, 1024, 1536, 2048)
EXPECTED_TRAIN = 7073


def summarize_lengths(values: list[int]) -> dict[str, float]:
    """算一组长度的均值、最大值与各分位数。"""
    ordered = sorted(values)
    total = len(ordered)
    summary = {
        "mean": sum(ordered) / total,
        "max": ordered[-1],
    }
    for point in PERCENTILES:
        # 就近取位，小样本下不作插值，避免给出分布里不存在的长度
        position = min(total - 1, int(round(point / 100 * (total - 1))))
        summary[f"p{point}"] = ordered[position]
    return summary


def print_summary(name: str, summary: dict[str, float]) -> None:
    """打印一组长度的分位数表。"""
    points = "  ".join(f"p{point} {summary[f'p{point}']}" for point in PERCENTILES)
    print(f"  {name}  均值 {summary['mean']:.1f}  最大 {summary['max']}")
    print(f"    {points}")


def main() -> int:
    parser = argparse.ArgumentParser(description="量训练文本与评测提示的 token 长度分布")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    parser.add_argument("--model-path", default="models/Qwen3-0.6B", help="分词器所在目录")
    parser.add_argument("--split-path", default="configs/dev_split.json", help="开发集索引文件")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    checks = []
    section("载入")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    examples = data.load_train_split(args.data_dir, Path(args.split_path))
    print(f"  训练集 {len(examples)} 条  开发集已剔除")
    report(checks, "训练集条数", OK if len(examples) == EXPECTED_TRAIN else FAIL,
           f"{len(examples)}，期望 {EXPECTED_TRAIN}")

    empty = sum(1 for item in examples if not data.build_target_text(item))
    report(checks, "目标文本非空", OK if not empty else FAIL, f"空 {empty} 条")

    section("逐条分词")
    prompt_lengths = []
    target_lengths = []
    total_lengths = []
    for position, example in enumerate(examples):
        prompt = data.render_prompt(tokenizer, example)
        target = data.build_target_text(example)
        whole = prompt + target + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # 合计量整串而不逐段相加，避免分段边界带来的切分差异
        total_ids = tokenizer(whole, add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(total_ids) - len(prompt_ids))
        total_lengths.append(len(total_ids))
        if (position + 1) % 1000 == 0:
            print(f"  已处理 {position + 1}/{len(examples)}")

    section("长度分布")
    total_summary = summarize_lengths(total_lengths)
    print_summary("提示", summarize_lengths(prompt_lengths))
    print_summary("目标", summarize_lengths(target_lengths))
    print_summary("合计", total_summary)

    section("候选上限下的截断")
    print("  上限   超出条数   占比     被截条数占比超过百分之一即判 WARN")
    for cap in CAPS:
        over = sum(1 for value in total_lengths if value > cap)
        state = OK if over / len(total_lengths) <= 0.01 else "WARN"
        print(f"  {cap:<6} {over:<9} {over / len(total_lengths):.4f}   {state}")

    section("覆盖所需长度")
    ordered = sorted(total_lengths)
    for cover in (0.99, 0.995, 0.999, 1.0):
        position = min(len(ordered) - 1, int(cover * len(ordered)) - 1)
        print(f"  覆盖 {cover:.3%} 需 {ordered[position]} token")

    print()
    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
