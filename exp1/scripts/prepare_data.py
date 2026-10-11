"""P0.5 数据集准备脚本。

下载 GSM8K 与 ARC-Challenge 并转换为 JSONL 落盘，供上传 OSS 后在 DLC 容器内
以挂载方式读取。转换的同时核对样本数与课程说明文档是否一致，并输出校验和，
以便上传后核对文件完整性。

本脚本只做获取与转换，不改变任何样本内容。

用法示例
    python scripts/prepare_data.py --output-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


# 说明文档给出的样本数，脚本运行后与实际值比对
SPECS = [
    {
        "name": "gsm8k_train",
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "expected": 7473,
        "role": "领域训练集",
    },
    {
        "name": "gsm8k_test",
        "path": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "expected": 1319,
        "role": "领域测试集",
    },
    {
        "name": "arc_challenge_test",
        "path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "expected": 1172,
        "role": "通用能力测试集，仅评测",
    },
]

PREVIEW_ROWS = 2


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="获取并转换实验数据集")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    mismatch = False

    for spec in SPECS:
        print(f"\n处理 {spec['name']}，来源 {spec['path']}，配置 {spec['config']}，划分 {spec['split']}")

        dataset = load_dataset(spec["path"], spec["config"], split=spec["split"])
        count = len(dataset)

        expected = spec["expected"]
        flag = "一致" if count == expected else "不一致"
        print(f"  样本数 实际 {count}，文档 {expected}，{flag}")
        if count != expected:
            mismatch = True

        target = output_dir / f"{spec['name']}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        checksum = sha256_of(target)
        size_kb = target.stat().st_size / 1024
        print(f"  已写出 {target}，{size_kb:.1f} KB")
        print(f"  校验和 {checksum[:16]}...")

        columns = list(dataset.column_names)
        print(f"  字段 {columns}")
        for index, row in enumerate(dataset.select(range(min(PREVIEW_ROWS, count)))):
            print(f"  样例 {index} {json.dumps(row, ensure_ascii=False)[:300]}")

        manifest.append(
            {
                "name": spec["name"],
                "role": spec["role"],
                "source": spec["path"],
                "config": spec["config"],
                "split": spec["split"],
                "rows": count,
                "expected_rows": expected,
                "columns": columns,
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": checksum,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n清单已写出 {manifest_path}")

    if mismatch:
        print("\n存在样本数与文档不一致的数据集，须查清后再上传。")
        return 1

    print("\n全部数据集样本数与文档一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
