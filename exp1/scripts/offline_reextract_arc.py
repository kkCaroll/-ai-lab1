"""离线重算 ARC 逐条预测的抽取结果，不加载模型。

抽取规则迭代后，已落盘的 output 字段仍是模型原样输出，可据其在本机重算 pred 与
判定，无须重跑模型。选项正文取自原始 ARC 题目文件，故须按 id 对齐。重算结果写入
同名 _v3 文件，原始预测不改动，两种口径可并列报告。若原先判对的样本重算后判错，
判 FAIL 并以非零码退出，因为那意味着新规则在既有正确结果上引入了回归。

用法示例
    cd exp1
    python -m scripts.offline_reextract_arc --data ../data/raw/arc_challenge_test.jsonl \
        --pred runs/P2-6c/M2/M2-predictions_arc.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import extract
from src.utils import FAIL, OK, WARN, report, section, summarize, write_json, write_jsonl

PREVIEW = 5
PREVIEW_CHARS = 90


def _read_jsonl(path: Path) -> list[dict]:
    """读取逐条明细，一行一条。"""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_raw(path: Path) -> dict[str, dict]:
    """载入原始题目，按 id 建索引，供取标签集合与选项正文。"""
    return {item["id"]: item for item in _read_jsonl(path)}


def process(path: Path, raw: dict[str, dict], checks: list) -> None:
    """重算一个预测文件并写出 _v3 同名产物。"""
    section(path.name)
    records = _read_jsonl(path)

    missing = [record["id"] for record in records if record["id"] not in raw]
    report(checks, "题目可按 id 对齐", OK if not missing else FAIL, f"缺 {len(missing)} 条")
    if missing:
        return

    method_counts = Counter()
    status_counts = Counter()
    changed = []
    flips_up = flips_down = 0
    correct = old_correct = 0
    truncated = 0
    for record in records:
        item = raw[record["id"]]
        outcome = extract.extract_arc(record["output"], item["choices"]["label"], item["choices"]["text"])
        hit = extract.is_correct(outcome.value, record["gold"], "arc")

        method_counts[outcome.method] += 1
        status_counts[outcome.status] += 1
        correct += hit
        old_correct += record["correct"]
        truncated += bool(record.get("truncated"))
        if outcome.value != record["pred"]:
            changed.append((record, record["extract_method"], outcome.method))
            if record["correct"] and not hit:
                flips_down += 1
            elif hit and not record["correct"]:
                flips_up += 1

        record["pred"] = outcome.value
        record["extract_status"] = outcome.status
        record["extract_method"] = outcome.method
        record["correct"] = hit

    samples = len(records)
    accuracy = correct / samples
    print(f"  准确率 {old_correct / samples:.4f} 改为 {accuracy:.4f}  正确 {old_correct} 改为 {correct}")
    print(f"  抽取失败 {status_counts[FAIL]} 条，降级 {status_counts[WARN]} 条")
    print(f"  规则命中 {dict(sorted(method_counts.items()))}")
    print(f"  取值改动 {len(changed)} 条，判对增 {flips_up} 减 {flips_down}")
    report(checks, "判对未减少", OK if flips_down == 0 else FAIL, f"减少 {flips_down} 条")
    report(checks, "抽取失败率低于阈值", OK if status_counts[FAIL] / samples <= 0.10 else FAIL,
           f"{status_counts[FAIL] / samples:.4f}")

    for record, old_method, new_method in [item for item in changed if item[0]["correct"]][:PREVIEW]:
        print(f"  [{record['index']}] gold {record['gold']} pred {record['pred']}"
              f"  {old_method} 改为 {new_method}  {' '.join(record['output'].split())[:PREVIEW_CHARS]}")

    stem = path.with_suffix("")
    metrics = {
        "source": path.name,
        "samples": samples,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "extract_failed": status_counts[FAIL],
        "extract_warned": status_counts[WARN],
        "extract_failure_rate": round(status_counts[FAIL] / samples, 4),
        "extract_method_counts": dict(sorted(method_counts.items())),
        "truncation_rate": round(truncated / samples, 4),
        "extract_rules_version": extract.EXTRACT_RULES_VERSION,
        "changed": len(changed),
        "flips_up": flips_up,
        "flips_down": flips_down,
    }
    write_jsonl(records, Path(f"{stem}_v3.jsonl"))
    write_json(metrics, Path(f"{stem}_v3.metrics.json"))
    print(f"  已写出 {stem}_v3.jsonl 与 {stem}_v3.metrics.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="按当前抽取规则离线重算 ARC 预测文件")
    parser.add_argument("--data", default="../data/raw/arc_challenge_test.jsonl", help="原始 ARC 题目文件")
    parser.add_argument("--pred", nargs="+", required=True, help="一个或多个逐条预测文件")
    args = parser.parse_args()

    section("重算")
    print(f"  抽取规则版本 {extract.EXTRACT_RULES_VERSION}")

    raw = _load_raw(Path(args.data))
    checks = []
    for name in args.pred:
        process(Path(name), raw, checks)

    print()
    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
