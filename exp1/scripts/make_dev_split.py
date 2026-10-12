"""P1.5 开发集划分脚本。

从 GSM8K 训练集以固定种子划出 400 条作为开发集，只记索引，不复制样本。开发集供
P2 选择学习率与训练轮数，不取测试集，以免测试集信息经选择过程泄漏进模型。

原始文件不改动，正式训练集由 src/data.py 的 load_train_split 按索引剔除得到 7073
条。索引文件同时记下源文件的 SHA256，供载入端核对索引与源文件是否仍然对齐。

本脚本只在本机运行一次，产物 exp1/configs/dev_split.json 入仓库，容器内不再执行，
故直接复用 src/utils.py 的报告机制而不另写一份。

用法示例
    cd exp1
    python -m scripts.make_dev_split --data-dir ../data/raw
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils import FAIL, OK, report, section, sha256_of, summarize, write_json

SEED = 42
DEV_SIZE = 400
SOURCE_FILE = "gsm8k_train.jsonl"
EXPECTED_TOTAL = 7473
PREVIEW = 8


def main() -> int:
    parser = argparse.ArgumentParser(description="从 GSM8K 训练集划出开发集索引")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    parser.add_argument("--output", default="configs/dev_split.json", help="索引文件写出位置")
    parser.add_argument("--seed", type=int, default=SEED, help="划分所用随机种子")
    parser.add_argument("--dev-size", type=int, default=DEV_SIZE, help="开发集条数")
    args = parser.parse_args()

    checks = []
    section("源文件")

    source = Path(args.data_dir) / SOURCE_FILE
    if not source.is_file():
        report(checks, "源文件可读", FAIL, str(source))
        return summarize(checks)

    checksum = sha256_of(source)
    with source.open(encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    report(checks, "源文件行数", OK if total == EXPECTED_TOTAL else FAIL, f"{total}，期望 {EXPECTED_TOTAL}")
    print(f"  源文件校验和 {checksum}")

    section("划分")
    if not 0 < args.dev_size < total:
        report(checks, "开发集规模", FAIL, f"{args.dev_size} 不在 0 与 {total} 之间")
        return summarize(checks)

    # 用 random.Random 的独立实例，不用 datasets 的 shuffle，令划分只依赖种子与
    # Python 版本，不依赖第三方库的行为。索引排序后写出，使文件内容与 diff 稳定。
    indices = sorted(random.Random(args.seed).sample(range(total), args.dev_size))
    report(checks, "开发集规模", OK, f"{args.dev_size} 条")

    section("写出")
    record = {
        "dataset": "gsm8k",
        "split": "train",
        "source_file": SOURCE_FILE,
        "source_sha256": checksum,
        "seed": args.seed,
        "total": total,
        "dev_size": args.dev_size,
        "train_size": total - args.dev_size,
        "indices": indices,
    }
    output = Path(args.output)
    write_json(record, output)
    print(f"  已写出 {output}")
    print(f"  索引前 {PREVIEW} 项 {indices[:PREVIEW]}")

    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
