"""按当前抽取规则重算既有运行目录中的抽取相关产物。

抽取规则迭代后，已落盘的逐条明细中 output 字段仍是模型原样输出，可据其在本机离线
重算 pred、判定与指标，无须重跑模型，也不消耗 DLC 计量时长。准确率之外的非抽取类
字段如生成耗时与显存峰值取自原指标文件，保持原样。

重算前后逐条比对取值，若原先判对的样本出现取值改动，判 FAIL 并中止，因为那意味着
新规则在既有正确结果上引入了回归。

用法示例
    cd exp1
    python -m scripts.reextract --run-dir ../runs/R1-1 --data-dir ../data/raw
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import data, extract
from src.utils import FAIL, OK, WARN, report, section, summarize, write_json, write_jsonl

PREVIEW = 5
PREVIEW_CHARS = 90


def _read_jsonl(path: Path) -> list[dict]:
    """读取逐条明细，一行一条。"""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_examples(dataset: str, split: str, data_dir: str) -> dict[int, dict]:
    """载入题目，按 index 建索引，供取标签集合与选项正文。"""
    return {item["index"]: item for item in data.load_examples(dataset, split, data_dir)}


def _extract(text: str, example: dict, dataset: str) -> extract.ExtractResult:
    """按数据集分派抽取规则。"""
    if dataset == "gsm8k":
        return extract.extract_gsm8k(text)
    return extract.extract_arc(text, example["labels"], example["choices"])


def process(dataset: str, run_dir: Path, data_dir: str, checks: list) -> None:
    """重算一个数据集的抽取产物并就地写回。"""
    predictions_path = run_dir / f"predictions_{dataset}.jsonl"
    metrics_path = run_dir / f"metrics_{dataset}.json"
    config_path = run_dir / f"config_{dataset}.json"
    section(dataset)

    if not predictions_path.is_file():
        report(checks, "逐条明细可读", FAIL, str(predictions_path))
        return

    records = _read_jsonl(predictions_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    examples = _load_examples(dataset, metrics["split"], data_dir)

    report(checks, "明细条数与指标一致", OK if len(records) == metrics["samples"] else FAIL,
           f"{len(records)} 条，指标记 {metrics['samples']} 条")

    method_counts = Counter()
    status_counts = Counter()
    changed = []
    flips_up = flips_down = 0
    correct = 0
    for record in records:
        example = examples[record["index"]]
        outcome = _extract(record["output"], example, dataset)
        hit = extract.is_correct(outcome.value, example["gold"], dataset)

        method_counts[outcome.method] += 1
        status_counts[outcome.status] += 1
        correct += hit
        if outcome.value != record["pred"]:
            changed.append((record, record["extract_method"], outcome.method, hit))
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
    old_accuracy = metrics["accuracy"]
    print(f"  准确率 {old_accuracy:.4f} 改为 {accuracy:.4f}  正确 {metrics['correct']} 改为 {correct}")
    print(f"  规则命中 {dict(sorted(method_counts.items()))}")
    print(f"  取值改动 {len(changed)} 条，判对增 {flips_up} 减 {flips_down}")
    report(checks, "判对未减少", OK if flips_down == 0 else FAIL, f"减少 {flips_down} 条")

    for record, old_method, new_method, _ in [item for item in changed if item[3]][:PREVIEW]:
        print(f"  [{record['index']}] {record['gold']}  {record['output'][:PREVIEW_CHARS]!r}"
              f"  {old_method} 改为 {new_method}")

    failures = [record for record in records if record["extract_status"] != OK]
    metrics.update({
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "extract_failed": status_counts[FAIL],
        "extract_warned": status_counts[WARN],
        "extract_failure_rate": round(status_counts[FAIL] / samples, 4),
        "extract_method_counts": dict(sorted(method_counts.items())),
        "extract_rules_version": extract.EXTRACT_RULES_VERSION,
    })
    config["extract_rules_version"] = extract.EXTRACT_RULES_VERSION

    write_jsonl(records, predictions_path)
    write_jsonl(failures, run_dir / f"extract_failures_{dataset}.jsonl")
    write_json(metrics, metrics_path)
    write_json(config, config_path)
    print(f"  产物已就地写回 {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按当前抽取规则重算既有运行目录")
    parser.add_argument("--run-dir", required=True, help="既有运行目录")
    parser.add_argument("--dataset", default="gsm8k,arc", help="逗号分隔的数据集名")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    section("重算")
    print(f"  抽取规则版本 {extract.EXTRACT_RULES_VERSION}")

    checks = []
    for dataset in [item.strip() for item in args.dataset.split(",") if item.strip()]:
        process(dataset, run_dir, args.data_dir, checks)

    print()
    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
