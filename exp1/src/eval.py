"""统一评测入口。

加载模型并批量跑完一个数据集，把模型输出经 src/extract.py 还原为答案，与标准答案
比对后写出五类产物。M0 基线与后续各方法共用本入口，方法差异仅由 --model-path 与
--adapter 两个参数表达。提示由 src/data.py 的模板函数生成，训练与评测同源，是风险
R8 的控制点。

批量生成自行实现，理由与要点见 PLAN/01-任务分解.md 与实施计划第五节。要点为左填充、
显式位置编号、贪心解码与按提示长度分组。

生成与抽取合为一个函数 predict_examples，不由 evaluate_dataset 独享。训练过程中的中间
评测直接调用它，故训练侧的读数与独立评测出自同一条代码路径，两者可比是结构保证而非
约定。

用法示例
    cd exp1
    python -m src.eval --model-path models/Qwen3-0.6B --dataset gsm8k --data-dir ../data/raw \\
        --output-dir ../results/R1-1 --limit 8 --dtype float32 --check-consistency 4
    # 开发集读数，只对 gsm8k 成立，供超参选择取绝对对照
    python -m src.eval --model-path /mnt/data/exp1/models/Qwen3-0.6B --dataset gsm8k \\
        --split dev --data-dir /mnt/data/exp1/data/raw --output-dir /mnt/data/exp1/runs/R2-8/m0-dev
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

import torch

from src import data, extract
from src.utils import (
    FAIL,
    OK,
    WARN,
    environment_snapshot,
    preferred_dtype,
    report,
    section,
    set_seed,
    summarize,
    write_json,
    write_jsonl,
)

# 抽取失败率与截断率的告警界，前者越过上界说明抽取规则不适配当前输出形态
EXTRACT_FAIL_LIMITS = (0.01, 0.10)
TRUNCATION_LIMITS = (0.02, 0.10)

# 生成参数的按数据集默认值，依据见 PLAN/01-任务分解.md 的已定决策
DEFAULT_BATCH_SIZE = {"gsm8k": 32, "arc": 64}
DEFAULT_MAX_NEW_TOKENS = {"gsm8k": 512, "arc": 512}

PROGRESS_EVERY = 5
SAMPLE_PREVIEW = 5
PREVIEW_CHARS = 200


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="统一评测入口，批量生成并统计准确率")
    parser.add_argument("--model-path", required=True, help="基座权重目录")
    parser.add_argument("--adapter", default=None, help="PEFT 适配器目录，基座评测时留空")
    parser.add_argument("--dataset", default="gsm8k,arc", help="逗号分隔，取 gsm8k 与 arc")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    parser.add_argument("--split", default="test", choices=["test", "dev"], help="dev 只对 gsm8k 成立，取训练集划出的 400 条开发集")
    parser.add_argument("--output-dir", required=True, help="运行目录，五类产物写于此")
    parser.add_argument("--batch-size", type=int, default=None, help="留空则按数据集取默认值")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="留空则按数据集取默认值")
    parser.add_argument("--limit", type=int, default=None, help="只评测前若干条，供本机自检")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-consistency", type=int, default=0, help="取前若干条比较单条与批量生成，0 为不检查"
    )
    return parser.parse_args()


def _three_state(value: float, limits: tuple[float, float]) -> str:
    """按两个上界把比值映射为三态，越小越好。"""
    warn_at, fail_at = limits
    if value <= warn_at:
        return OK
    return WARN if value <= fail_at else FAIL


def _load_model(model_path: str, adapter: str | None, device, dtype):
    """加载分词器与模型，返回两者与加载耗时。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()
    return tokenizer, model, time.perf_counter() - start


def _make_batch(id_lists: list[list[int]], pad_id: int, device):
    """左填充若干条已分词的输入，返回输入张量、注意力掩码与位置编号。

    补在左侧是因为自回归生成读的是序列末位，补在右侧会把末位变成填充符。填充不经由
    分词器的 padding，以免其默认的右侧填充悄悄生效。位置编号由掩码逐位累减得到，使
    真实内容的编号仍从零起算，否则左填充会让整段内容的位置整体偏移。
    """
    width = max(len(ids) for ids in id_lists)
    input_ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for row, ids in enumerate(id_lists):
        length = len(ids)
        input_ids[row, width - length :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, width - length :] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
    return input_ids.to(device), attention_mask.to(device), position_ids.to(device)


def _generate(model, tokenizer, id_lists: list[list[int]], max_new_tokens: int, device):
    """贪心生成一批输入，返回每条的文本、生成 token 数与是否撞上长度上限。

    结束符只取 <|im_end|>，不取 <|endoftext|>，后者是填充符，一并当作结束符会让短
    序列在填充处被截断。生成段按首个结束符切齐，其后的填充符不计入长度。
    """
    input_ids, attention_mask, position_ids = _make_batch(id_lists, tokenizer.pad_token_id, device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    width = input_ids.shape[1]
    results = []
    for row in range(len(id_lists)):
        generated = outputs[row, width:].tolist()
        truncated = tokenizer.eos_token_id not in generated
        if not truncated:
            generated = generated[: generated.index(tokenizer.eos_token_id) + 1]
        results.append(
            (tokenizer.decode(generated, skip_special_tokens=True), len(generated), truncated)
        )
    return results


def _generate_all(model, tokenizer, encoded, max_new_tokens, device, batch_size):
    """分批生成全部输入，返回与原顺序对齐的结果与统计量。

    按提示长度排序分组，使同批的补齐量最小。显存不足时清缓存并把批大小折半，此后沿用
    折半后的取值，避免在同类批上反复触发。
    """
    order = sorted(range(len(encoded)), key=lambda item: len(encoded[item]))
    results = [None] * len(encoded)
    stats = {"batches": 0, "oom": 0}
    position = 0
    current = batch_size
    start = time.perf_counter()

    while position < len(order):
        chunk = [encoded[item] for item in order[position : position + current]]
        try:
            generated = _generate(model, tokenizer, chunk, max_new_tokens, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            stats["oom"] += 1
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f"  显存不足，批大小折半为 {current}", flush=True)
            continue

        for offset, item in enumerate(generated):
            results[order[position + offset]] = item
        position += len(chunk)
        stats["batches"] += 1

        if stats["batches"] == 1 or stats["batches"] % PROGRESS_EVERY == 0 or position == len(order):
            elapsed = time.perf_counter() - start
            rate = position / elapsed if elapsed else 0.0
            remain = (len(order) - position) / rate / 60 if rate else 0.0
            print(
                f"  批 {stats['batches']}  已完成 {position}/{len(order)}"
                f"  已用 {elapsed:.1f} 秒  预计剩余 {remain:.1f} 分",
                flush=True,
            )

    stats["seconds"] = time.perf_counter() - start
    return results, stats


def _check_consistency(model, tokenizer, encoded, max_new_tokens, device, count, checks) -> bool:
    """比较同一条样本的单条生成与批量生成，返回是否全部一致。

    两者不一致说明左填充、注意力掩码与位置编号三处至少有一处写错，而这三处写错不会
    显性报错，只会让全量结果整体偏移，故本项作为上 DLC 前的关卡。
    """
    count = min(count, len(encoded))
    subset = encoded[:count]
    single = [_generate(model, tokenizer, [ids], max_new_tokens, device)[0] for ids in subset]
    batched = _generate(model, tokenizer, subset, max_new_tokens, device)
    mismatched = [item for item in range(count) if single[item][0] != batched[item][0]]
    detail = f"{count} 条，不一致 {len(mismatched)} 条"
    report(checks, "批量与单条一致", OK if not mismatched else FAIL, detail)
    for item in mismatched[:SAMPLE_PREVIEW]:
        print(f"  单条 {single[item][0][:PREVIEW_CHARS]}")
        print(f"  批量 {batched[item][0][:PREVIEW_CHARS]}")
    return not mismatched


def predict_examples(model, tokenizer, encoded, examples, dataset, device, batch_size, max_new_tokens):
    """批量生成并抽取答案，返回逐条记录与生成统计。

    逐条记录即 predictions 文件的字段，含模型原文、抽取取值、命中的规则名与判定结果。
    本函数不写任何产物，也不打印汇总，由调用方处置。encoded 与 examples 须等长且逐位
    对齐，后者提供标准答案、选项与标签。

    进入生成前须把模型切到推理态。训练过程中的中间评测传进来的模型处于训练态，若不
    切换，dropout 会随手生效，贪心解码不再确定，与独立评测的读数不可比。缓存开关同
    此，梯度检查点会把它关掉，生成时开着可省去大量重复前向。两者用毕一律还原。
    """
    if len(encoded) != len(examples):
        raise ValueError(f"输入与样本不对齐，{len(encoded)} 与 {len(examples)}")

    was_training = model.training
    saved_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    try:
        results, stats = _generate_all(model, tokenizer, encoded, max_new_tokens, device, batch_size)
    finally:
        model.train(was_training)
        model.config.use_cache = saved_use_cache

    records = []
    for position, (example, (text, gen_tokens, truncated)) in enumerate(zip(examples, results)):
        if dataset == "gsm8k":
            outcome = extract.extract_gsm8k(text)
        else:
            outcome = extract.extract_arc(text, example["labels"], example["choices"])
        records.append(
            {
                "index": example["index"],
                "id": example["id"],
                "dataset": dataset,
                "question": example["question"],
                "gold": example["gold"],
                "output": text,
                "pred": outcome.value,
                "extract_status": outcome.status,
                "extract_method": outcome.method,
                "correct": extract.is_correct(outcome.value, example["gold"], dataset),
                "prompt_tokens": len(encoded[position]),
                "gen_tokens": gen_tokens,
                "truncated": truncated,
            }
        )
    return records, stats


def evaluate_dataset(args, dataset, tokenizer, model, device, dtype, load_seconds, checks):
    """评测一个数据集，写出产物并返回指标字典，前置检查未过时返回 None。"""
    section(f"数据集 {dataset}")
    if args.split == "dev":
        # 开发集不是独立文件，条数由载入端与索引记录核对，此处不再重复
        examples = data.load_dev_split(args.data_dir)
        expected = len(examples)
        if args.limit is not None:
            examples = examples[: args.limit]
    else:
        examples = data.load_examples(dataset, args.split, args.data_dir, limit=args.limit)
        expected = data.DATA_COUNTS[(dataset, args.split)]

    if args.limit is None:
        report(checks, "数据行数", OK if len(examples) == expected else FAIL, f"{len(examples)}，期望 {expected}")
    else:
        report(checks, "数据行数", OK, f"{len(examples)}，本次限流 {args.limit} 条，未与 {expected} 核对")

    missing = sum(1 for item in examples if item["gold"] is None)
    report(checks, "标准答案完整", OK if not missing else FAIL, f"缺失 {missing} 条")

    if dataset == "arc":
        counted = Counter(item["gold"] for item in examples)
        print(f"  标签分布 {dict(sorted(counted.items()))}")
    else:
        print(f"  标准答案去重 {len({item['gold'] for item in examples})} 种")

    prompts = [data.render_prompt(tokenizer, item) for item in examples]
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    lengths = [len(ids) for ids in encoded]
    prompt_digest = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    # 与数据集无关的模板指纹，训练侧另落一份同名同义的取值，两者相等即 R8 成立
    template_digest = data.template_digest(tokenizer)
    print(f"  提示长度 最长 {max(lengths)}，均值 {sum(lengths) / len(lengths):.1f} token")
    print(f"  提示校验和 {prompt_digest[:16]}  模板指纹 {template_digest[:16]}")

    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[dataset]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[dataset]

    if args.check_consistency:
        section("一致性自检")
        if not _check_consistency(
            model, tokenizer, encoded, max_new_tokens, device, args.check_consistency, checks
        ):
            return None

    section("推理")
    print(f"  设备 {device}  精度 {dtype}  批大小 {batch_size}  生成上限 {max_new_tokens}")
    print(f"  模型加载 {load_seconds:.1f} 秒")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    records, stats = predict_examples(
        model, tokenizer, encoded, examples, dataset, device, batch_size, max_new_tokens
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print(f"  生成 {stats['seconds']:.1f} 秒  批次 {stats['batches']}  显存峰值 {peak_gb:.2f} GiB")

    failures = [item for item in records if item["extract_status"] != OK]
    method_counts = Counter(item["extract_method"] for item in records)
    status_counts = Counter(item["extract_status"] for item in records)

    samples = len(records)
    correct = sum(1 for item in records if item["correct"])
    gen_tokens_total = sum(item["gen_tokens"] for item in records)
    accuracy = correct / samples
    failure_rate = status_counts[FAIL] / samples
    truncation_rate = sum(1 for item in records if item["truncated"]) / samples

    section("结果")
    print(f"  准确率 {accuracy:.4f}  正确 {correct}/{samples}")
    print(f"  规则命中 {dict(sorted(method_counts.items()))}")
    if dataset == "gsm8k":
        print(f"  标记命中率 {method_counts['marker'] / samples:.4f}")
    report(
        checks,
        "抽取失败率",
        _three_state(failure_rate, EXTRACT_FAIL_LIMITS),
        f"{failure_rate:.4f}，{status_counts[FAIL]} 条",
    )
    report(checks, "截断率", _three_state(truncation_rate, TRUNCATION_LIMITS), f"{truncation_rate:.4f}")
    covered = [item["index"] for item in records] == [item["index"] for item in examples]
    report(checks, "预测逐条对齐", OK if covered else FAIL, f"{samples} 条，期望 {len(examples)} 条")

    section("样例")
    if not failures:
        print("  无抽取失败或降级的样本")
    for record in failures[:SAMPLE_PREVIEW]:
        print(f"  [{record['extract_status']}] {record['id']}  标准答案 {record['gold']}")
        print(f"    输出 {' '.join(record['output'].split())[:PREVIEW_CHARS]}")
        print(f"    抽取 {record['pred']}  规则 {record['extract_method']}")

    metrics = {
        "dataset": dataset,
        "split": args.split,
        "samples": samples,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "extract_failed": status_counts[FAIL],
        "extract_warned": status_counts[WARN],
        "extract_failure_rate": round(failure_rate, 4),
        "extract_method_counts": dict(sorted(method_counts.items())),
        "truncated": sum(1 for item in records if item["truncated"]),
        "truncation_rate": round(truncation_rate, 4),
        "prompt_tokens_max": max(lengths),
        "prompt_tokens_mean": round(sum(lengths) / samples, 1),
        "gen_tokens_total": gen_tokens_total,
        "gen_tokens_mean": round(gen_tokens_total / samples, 1),
        "tokens_per_second": round(gen_tokens_total / stats["seconds"], 1),
        "peak_gpu_memory_gb": round(peak_gb, 2),
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(stats["seconds"], 1),
        "batch_size": batch_size,
        "oom_events": stats["oom"],
        "limit": args.limit,
        "prompt_version": data.PROMPT_VERSION,
        "extract_rules_version": extract.EXTRACT_RULES_VERSION,
        "prompt_sha256": prompt_digest,
    }
    config = {
        "dataset": dataset,
        "split": args.split,
        "model_path": args.model_path,
        "adapter": args.adapter,
        "prompt_version": data.PROMPT_VERSION,
        "extract_rules_version": extract.EXTRACT_RULES_VERSION,
        "prompt_sha256": prompt_digest,
        "prompt_template_sha256": template_digest,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "device": str(device),
            "dtype": str(dtype),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "seed": args.seed,
        "limit": args.limit,
        "data_dir": args.data_dir,
    }

    output_dir = Path(args.output_dir)
    write_jsonl(records, output_dir / f"predictions_{dataset}.jsonl")
    write_jsonl(failures, output_dir / f"extract_failures_{dataset}.jsonl")
    write_json(metrics, output_dir / f"metrics_{dataset}.json")
    write_json(config, output_dir / f"config_{dataset}.json")
    print(f"\n  产物已写出至 {output_dir}")
    return metrics


def main() -> int:
    """按数据集顺序评测并汇总。"""
    args = _parse_args()
    checks = []

    datasets = [item.strip() for item in args.dataset.split(",") if item.strip()]
    unknown = [item for item in datasets if item not in DEFAULT_BATCH_SIZE]
    if unknown or not datasets:
        print(f"数据集取值不合法 {datasets}，只接受 gsm8k 与 arc")
        return 2
    if args.split == "dev" and datasets != ["gsm8k"]:
        print(f"--split dev 只对 gsm8k 成立，本次给出 {datasets}")
        return 2

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.dtype != "auto":
        dtype = getattr(torch, args.dtype)
    elif device.type == "cpu":
        # 处理器上 float16 大量算子没有实现
        dtype = torch.float32
    else:
        dtype = preferred_dtype()

    section("环境")
    print(f"  设备 {device}  精度 {dtype}  种子 {args.seed}")
    set_seed(args.seed)

    tokenizer, model, load_seconds = _load_model(args.model_path, args.adapter, device, dtype)
    report(checks, "模型加载", OK, f"{load_seconds:.1f} 秒")

    peak_gb = 0.0
    for dataset in datasets:
        metrics = evaluate_dataset(
            args, dataset, tokenizer, model, device, dtype, load_seconds, checks
        )
        # 任一数据集未过前置检查即中止，产物不完整时整任务换新目录重跑
        if metrics is None:
            break
        peak_gb = max(peak_gb, metrics["peak_gpu_memory_gb"])
        snapshot = environment_snapshot()
        snapshot["peak_gpu_memory_gb"] = round(peak_gb, 2)
        write_json(snapshot, Path(args.output_dir) / "env.json")

    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
