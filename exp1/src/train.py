"""统一训练入口。

读入三层配置，装配方法模块，在 GSM8K 训练集上微调，每轮在开发集上评一次并按轮存点，
最后写出 config.json、env.json 与 train.log。方法差异全部由配置给出，本文件不为任一
方法开特例。

训练文本由三段拼成，即提示、答案正文与结束标记。提示取自 src/data.py 的 render_prompt，
与评测侧同源；答案正文取自 build_target_text，已剥离计算标注。

训练侧的批量与评测侧相反。评测须左填充，训练须右填充，填充一律只补到本批最长一条而非
max_seq_len，否则大部分算力耗在填充符上。提示段与目标段分别分词后拼接，不整串一次分词，
理由是整串分词在接缝处的切分可能与评测侧单独分词的提示不一致，而两侧提示逐 token 相同
正是风险 R8 的要求。标签只在目标段上计算损失，提示段一律置 -100。

中间评测直接调用 src/eval.py 的 predict_examples，故训练中的开发集读数与独立评测出自
同一条代码路径。配置的 eval.dataset 只接受 dev，取值校验写在 config.check 内，训练过程
不会触及测试集，这是风险 R7。

混合精度的取法为先按配置算出计算精度，权重一律以 float32 载入，再由 autocast 在算子内降到
半步精度，梯度缩放只在计算精度为 float16 时启用。不可把权重直接载入 float16，那样梯度同为
float16，而梯度缩放要求反缩放一步能在 float32 上进行，torch 遇 float16 梯度即报错，见
LOG/02-问题与解决.md Q12。

用法示例
    cd exp1
    python -m src.train --config configs/lora.yaml --model-path models/Qwen3-0.6B \\
        --data-dir ../data/raw --output-dir ../results/P2-lora \\
        --set run.limit=64 --set run.max_steps=4 --set eval.limit=8
"""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import time
from pathlib import Path

import torch

from src import config as config_module
from src import data, methods
from src import eval as eval_module
from src.utils import (
    FAIL,
    OK,
    Check,
    environment_snapshot,
    preferred_dtype,
    set_seed,
    summarize,
    write_json,
)

# 标签在提示段与填充位上的取值，负值不参与交叉熵
IGNORE_INDEX = -100

CHECKPOINT_DIR = "checkpoints"


class _Log:
    """把每一行同时写到标准输出与 train.log。

    每行即时冲刷，使任务被中断时已产生的日志仍可从 OSS 取回。
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def report(self, checks: list[Check], name: str, status: str, detail: str = "") -> None:
        """同 utils.report，差别是本条也进 train.log。"""
        checks.append(Check(name, status, detail))
        self(f"[{status}] {name}  {detail}".rstrip())

    def close(self) -> None:
        self._handle.close()


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    模型路径、数据目录、输出目录与运行标识四项不写入配置文件，因为同一份配置要在本机与
    容器两处生效而两处的路径不同。种子与方法名分别由 run.seed 与 method 键给出，命令行
    不再另传一份，免得同一件事有两处真源。
    """
    parser = argparse.ArgumentParser(description="统一训练入口，方法与超参全部由配置给出")
    parser.add_argument("--config", required=True, help="方法配置文件，如 configs/lora.yaml")
    parser.add_argument("--model-path", required=True, help="基座权重目录")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    parser.add_argument("--output-dir", required=True, help="运行目录，产物写于此")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="键路径=值",
        help="覆盖两层配置文件，可重复，供 P2.5 扫参与本机自检",
    )
    return parser.parse_args()


def _resolve_dtype(name: str, device) -> torch.dtype:
    """按配置取值定计算精度，auto 时按设备与算力判定。权重精度不由此项决定，恒为 float32。"""
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cpu":
        return torch.float32
    return preferred_dtype()


def _autocast(device, amp_dtype):
    """开出混合精度上下文，无需混合精度时返回空上下文。"""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def _encode_example(tokenizer, example):
    """把一条样本编成输入编号与标签，返回两个等长的列表。

    提示与目标分别分词再拼接，故提示段的编号与评测侧单独分词的编号逐位相同。结束标记接
    在目标之后，使模型学到在何处停笔。
    """
    prompt = data.render_prompt(tokenizer, example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_text = data.build_target_text(example) + tokenizer.eos_token
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    return prompt_ids + target_ids, [IGNORE_INDEX] * len(prompt_ids) + target_ids


def _make_batch(rows, pad_id, device):
    """右填充一批已编码样本，返回输入、注意力掩码与标签。

    训练侧必须右填充。损失逐位预测下一位，补在右侧时受影响的只是末尾若干填充位，其标签
    为 -100 不计入损失；补在左侧则会把提示与目标的对应关系整体挪动。填充量取本批最长。
    """
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for position, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[position, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[position, :length] = torch.tensor(row["labels"], dtype=torch.long)
        attention_mask[position, :length] = 1
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def _epoch_groups(size, batch_size, grad_accum, generator):
    """打乱样本序后切成微批，再按累积步数分组，返回若干组。

    先分组再计算是使每步的累积分母确定。末组不满时按其实际微批数等分梯度，不按 grad_accum，
    否则该步的梯度会被整体缩小。打乱用传入的生成器，种子固定则轮间次序可复现。
    """
    order = torch.randperm(size, generator=generator).tolist()
    micro = [order[start : start + batch_size] for start in range(0, size, batch_size)]
    return [micro[start : start + grad_accum] for start in range(0, len(micro), grad_accum)]


def _evaluate_dev(model, tokenizer, examples, encoded, settings, device, amp_dtype):
    """在开发集上评测一次，返回指标字典。

    predict_examples 自带推理态与缓存开关的存取还原，故此处不必另行切换。limit 置位时只
    取前若干条，供本机自检压缩耗时。生成包在计算精度上下文内，权重为 float32 时若不加这
    一层，缓存与算子在整条生成过程中都停在 float32，显存与时耗都会明显偏离独立评测。
    """
    if settings["limit"] is not None:
        examples, encoded = examples[: settings["limit"]], encoded[: settings["limit"]]
    if not examples:
        raise ValueError("开发集为空，核对 eval.limit 与 data.dev_split")

    with _autocast(device, amp_dtype):
        records, stats = eval_module.predict_examples(
            model,
            tokenizer,
            encoded,
            examples,
            "gsm8k",
            device,
            settings["batch_size"],
            settings["max_new_tokens"],
        )
    correct = sum(1 for item in records if item["correct"])
    return {
        "samples": len(records),
        "correct": correct,
        "accuracy": round(correct / len(records), 4),
        "gen_tokens_mean": round(sum(item["gen_tokens"] for item in records) / len(records), 1),
        "generate_seconds": round(stats["seconds"], 1),
    }


def main() -> int:  # noqa: C901
    """按配置完成一次训练，返回退出码。"""
    args = _parse_args()

    try:
        merged = config_module.load(Path(args.config), args.overrides)
        method_module = methods.load_module(merged["method"])
        config_module.apply_method(merged, method_module.CONFIG_SCHEMA)
        config_module.check(merged)
    except (config_module.ConfigError, KeyError) as error:
        print(f"配置不合法  {error}")
        return 2

    method = merged["method"]
    method_config = merged.get(method) or {}
    run_config = merged["run"]
    train_config = merged["train"]
    eval_config = merged["eval"]
    output_config = merged["output"]
    epochs = train_config["epochs"]
    batch_size = train_config["batch_size"]
    grad_accum = train_config["grad_accum"]
    max_grad_norm = train_config["max_grad_norm"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(output_dir / "train.log")
    checks: list[Check] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = _resolve_dtype(train_config["dtype"], device)
    amp_dtype = (
        compute_dtype
        if device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
        else None
    )
    set_seed(run_config["seed"])

    log("")
    log("===== 环境 =====")
    log(
        f"  方法 {method}  设备 {device}  参数精度 float32"
        f"  计算精度 {amp_dtype or torch.float32}  种子 {run_config['seed']}"
    )
    log(f"  配置 {args.config}  覆盖 {args.overrides or '无'}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    log("===== 载入 =====")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 权重以 float32 载入。载入 float16 会使梯度同为 float16，缩放的梯度无法反缩放
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(device)
    log(f"  模型加载 {time.perf_counter() - started:.1f} 秒")
    log.report(
        checks,
        "分词器结束标记",
        OK if tokenizer.eos_token else FAIL,
        f"{tokenizer.eos_token} 编号 {tokenizer.eos_token_id}",
    )
    if not tokenizer.eos_token:
        log("目标文本须接结束标记，缺失即中止")
        log.close()
        return 1

    model = method_module.build(model, method_config)
    if train_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        # 冻结词嵌入时梯度回传不到适配器，损失会一动不动而任务照常跑完
        model.enable_input_require_grads()
    model.train()

    description = method_module.describe(model, method_config)
    log("===== 方法 =====")
    log(f"  检查点形式 {description['checkpoint_kind']}")
    log(
        f"  参数量 总量 {description['total_parameters']:,}"
        f"  可训练 {description['trainable_parameters']:,}"
        f"  占比 {description['trainable_ratio']:.6f}"
    )
    if "targeted_module_names" in description:
        # 逐层列出可达上百项，只报处数与模块后缀，全量清单见 config.json
        names = description["targeted_module_names"]
        suffixes = sorted({item.rsplit(".", 1)[-1] for item in names})
        log(f"  挂载模块 {len(names)} 处  后缀 {'、'.join(suffixes)}")
    if "prompt_injection" in description:
        log(f"  注入位置 {description['prompt_injection']}")
    log.report(
        checks,
        "存在可训练参数",
        OK if description["trainable_parameters"] else FAIL,
        f"{description['trainable_parameters']:,}",
    )
    if not description["trainable_parameters"]:
        log.close()
        return 1

    log("===== 数据 =====")
    split_path = Path(merged["data"]["dev_split"])
    dev_examples = data.load_dev_split(args.data_dir, split_path)
    examples = data.load_train_split(args.data_dir, split_path)
    expected = data.DATA_COUNTS[("gsm8k", "train")] - len(dev_examples)
    log(f"  训练集 {len(examples)} 条  开发集 {len(dev_examples)} 条")
    if run_config["limit"] is None:
        log.report(
            checks, "训练集条数", OK if len(examples) == expected else FAIL, f"{len(examples)}，期望 {expected}"
        )
    else:
        log.report(checks, "训练集条数", OK, f"{len(examples)}，本次限流 {run_config['limit']} 条，未与 {expected} 核对")
        examples = examples[: run_config["limit"]]

    cap = merged["data"]["max_seq_len"]
    started = time.perf_counter()
    encoded = [_encode_example(tokenizer, example) for example in examples]
    rows = [{"input_ids": ids, "labels": labels} for ids, labels in encoded]
    lengths = [len(ids) for ids, _ in encoded]
    oversized = sum(1 for value in lengths if value > cap)
    log(
        f"  训练文本 最长 {max(lengths)}  均值 {sum(lengths) / len(lengths):.1f} token"
        f"  编码 {time.perf_counter() - started:.1f} 秒"
    )
    log.report(checks, "样本不超长", OK if not oversized else FAIL, f"超 {cap} 的 {oversized} 条，上限 {cap}")
    if oversized:
        log("训练侧截断会切掉目标末行的 #### 答案，故中止")
        log.close()
        return 1

    dev_encoded = [
        tokenizer(data.render_prompt(tokenizer, item), add_special_tokens=False)["input_ids"]
        for item in dev_examples
    ]
    log(f"  开发集提示 最长 {max(len(ids) for ids in dev_encoded)} token")

    parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=train_config["lr"], weight_decay=train_config["weight_decay"]
    )
    steps_per_epoch = math.ceil(math.ceil(len(rows) / batch_size) / grad_accum)
    total_steps = run_config["max_steps"] or steps_per_epoch * epochs
    warmup_steps = int(total_steps * train_config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    # 缩放只对 float16 生效，bfloat16 与 float32 的指数范围无需缩放
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None

    log("===== 训练 =====")
    log(
        f"  轮数 {epochs}  每轮步数 {steps_per_epoch}  计划总步数 {total_steps}  预热 {warmup_steps} 步"
    )
    log(
        f"  微批 {batch_size}  累积 {grad_accum}  有效批 {batch_size * grad_accum}"
        f"  学习率 {train_config['lr']:.2e}  调度 {train_config['lr_scheduler']}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(run_config["seed"])
    global_step = 0
    saved: list[Path] = []
    eval_history: list[dict] = []
    running_loss = 0.0
    steps_since_log = 0
    total_tokens = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if global_step >= total_steps:
            break
        truncated_epoch = False
        epoch_loss = 0.0
        epoch_steps = 0

        for group in _epoch_groups(len(rows), batch_size, grad_accum, generator):
            if global_step >= total_steps:
                truncated_epoch = True
                break
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for indices in group:
                input_ids, attention_mask, labels = _make_batch(
                    [rows[item] for item in indices], tokenizer.pad_token_id, device
                )
                with _autocast(device, amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    extra = method_module.extra_loss(
                        model,
                        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
                        outputs,
                        method_config,
                    )
                    if extra is not None:
                        loss = loss + extra
                scaled = loss / len(group)
                (scaler.scale(scaled) if scaler else scaled).backward()
                group_loss += float(loss.detach())
                total_tokens += int(attention_mask.sum())

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += group_loss / len(group)
            epoch_steps += 1
            running_loss += group_loss / len(group)
            steps_since_log += 1

            if global_step % train_config["log_every"] == 0 or global_step == total_steps:
                elapsed = time.perf_counter() - started
                log(
                    f"  轮 {epoch}/{epochs}  步 {global_step}/{total_steps}"
                    f"  损失 {running_loss / steps_since_log:.4f}"
                    f"  学习率 {scheduler.get_last_lr()[0]:.2e}"
                    f"  速度 {total_tokens / elapsed:.0f} token/s  已用 {elapsed / 60:.1f} 分"
                )
                running_loss = 0.0
                steps_since_log = 0

        log(f"  第 {epoch} 轮结束  平均损失 {epoch_loss / max(epoch_steps, 1):.4f}  本轮步数 {epoch_steps}")
        if truncated_epoch:
            log(f"  已达 max_steps {total_steps}，本轮样本未跑完")

        # eval.mode 取 final 时只评末次，被 max_steps 截断的那一轮也是末次
        last_epoch = epoch == epochs or global_step >= total_steps
        if eval_config["mode"] == "every_epoch" or last_epoch:
            metrics = _evaluate_dev(
                model, tokenizer, dev_examples, dev_encoded, eval_config, device, amp_dtype
            )
            metrics["epoch"] = epoch
            eval_history.append(metrics)
            log(
                f"  开发集  准确率 {metrics['accuracy']:.4f}  正确 {metrics['correct']}/{metrics['samples']}"
                f"  均长 {metrics['gen_tokens_mean']} token  用时 {metrics['generate_seconds']} 秒"
            )

        checkpoint = output_dir / CHECKPOINT_DIR / f"epoch-{epoch}"
        files = method_module.save(model, tokenizer, checkpoint)
        saved.append(checkpoint)
        log(f"  检查点 {checkpoint.relative_to(output_dir)}  文件 {len(files)} 个")
        while len(saved) > output_config["save_total_limit"]:
            stale = saved.pop(0)
            try:
                shutil.rmtree(stale)
                log(f"  删除较早的检查点 {stale.relative_to(output_dir)}")
            except OSError as error:
                log(f"  删除 {stale.relative_to(output_dir)} 未成  {error}")

    elapsed = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    log("===== 结束 =====")
    log(
        f"  总步数 {global_step}  用时 {elapsed / 60:.1f} 分  显存峰值 {peak_gb:.2f} GiB"
        f"  处理 token {total_tokens:,}  速度 {total_tokens / elapsed:.0f} token/s"
    )
    if global_step < total_steps:
        log(f"  实际步数少于计划的 {total_steps} 步，数据已跑完全部 {epochs} 轮")

    final = saved[-1] if saved else None
    write_json(
        {
            "method": method,
            "method_description": description,
            "config": merged,
            "run": {
                "model_path": args.model_path,
                "data_dir": args.data_dir,
                "output_dir": str(output_dir),
                "overrides": args.overrides,
                "device": str(device),
                "param_dtype": "float32",
                "compute_dtype": str(amp_dtype or torch.float32),
                "seed": run_config["seed"],
            },
            # 四个派生量为记录表直接引用。模板指纹与评测入口的 config_<dataset>.json 同名
            # 同义，两者相等即 R8 成立，由人工比对升级为文件等值核对。
            "derived": {
                "prompt_version": data.PROMPT_VERSION,
                "prompt_template_sha256": data.template_digest(tokenizer),
                "train_size": len(rows),
                "effective_batch_size": batch_size * grad_accum,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
            "steps_completed": global_step,
            "train_seconds": round(elapsed, 1),
            "final_checkpoint": str(final.relative_to(output_dir)) if final else None,
            "checkpoints": [str(item.relative_to(output_dir)) for item in saved],
            "eval_history": eval_history,
        },
        output_dir / "config.json",
    )
    snapshot = environment_snapshot()
    snapshot["peak_gpu_memory_gb"] = round(peak_gb, 2)
    snapshot["train_seconds"] = round(elapsed, 1)
    write_json(snapshot, output_dir / "env.json")
    log(f"  产物已写出至 {output_dir}")

    code = summarize(checks)
    log.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
