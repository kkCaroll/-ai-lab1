"""数据载入与提示构造模块。

把 data/raw 的三份 jsonl 归一化为统一结构并生成模型输入。提示模板集中于此，
训练与评测共用，是风险 R8 的控制点。

不导入 torch 与 transformers，分词器由参数传入，故可在不加载模型的机器上测试。

用法
    examples = data.load_examples("gsm8k", "test", data_dir)
    text = data.render_prompt(tokenizer, examples[0])
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.utils import sha256_of

# ----------------- 常量 ------------------

# 模板文本改动时递增，并誊录至 RECORD/01-实验配置.md 第四节。
PROMPT_VERSION = "v1"

# 两条指令都显式约束输出格式，抽取模块依赖它。GSM8K 要求末行给出 #### 标记。
GSM8K_INSTRUCTION = "Solve the problem step by step and put the final numerical answer on the last line in the form #### <number>."
ARC_INSTRUCTION = "Answer the following multiple-choice question. Reply with the option label only."

# 期望行数，载入后据此核对文件完整性。
DATA_COUNTS = {("gsm8k", "train"): 7473, ("gsm8k", "test"): 1319, ("arc", "test"): 1172}

# 不用 f"{dataset}_{split}.jsonl" 拼，ARC 的文件名含 challenge 一词。
DATA_FILES = {
    ("gsm8k", "train"): "gsm8k_train.jsonl",
    ("gsm8k", "test"): "gsm8k_test.jsonl",
    ("arc", "test"): "arc_challenge_test.jsonl",
}

# 开发集索引文件，由 scripts/make_dev_split.py 生成并随仓库分发
DEV_SPLIT_PATH = Path(__file__).resolve().parent.parent / "configs" / "dev_split.json"


def _normalize_gsm8k(row, index, split):
    """把一行 GSM8K 原始记录转成统一结构。

    id 由 dataset、split 与 index 合成，因为 GSM8K 数据本身没有标识字段。
    """
    return {
        "index": index,
        "id": f"gsm8k-{split}-{index:04d}",
        "dataset": "gsm8k",
        "question": row["question"],
        "choices": None,
        "labels": None,
        "gold": parse_gsm8k_gold(row["answer"]),
        "gold_raw": row["answer"],
    }


def _normalize_arc(row, index):
    """把一行 ARC 原始记录转成统一结构。

    choices 与 labels 取自题目自带的 text 与 label 两个平行列表，按原顺序一一对应。
    """
    return {
        "index": index,
        "id": row["id"],
        "dataset": "arc",
        "question": row["question"],
        "choices": row["choices"]["text"],
        "labels": row["choices"]["label"],
        "gold": row["answerKey"],
        "gold_raw": row["answerKey"],
    }


def load_examples(dataset, split, data_dir, limit=None):
    """载入一个数据集划分，归一化为统一结构。

    八个字段为 index、id、dataset、question、choices、labels、gold、gold_raw。
    index 取文件行号，同时是开发集划分的键，故必须按文件顺序载入，不可排序或打乱，
    否则将与 configs/dev_split.json 错位且不报错。choices 与 labels 仅 ARC 有值，
    取自题目自带的 text 与 label 两个平行列表，不可硬编码字母表，测试集中有 22 题
    的标签为数字。

    Args:
        dataset: 取 gsm8k 或 arc。
        split: 取 train 或 test，合法组合见 DATA_FILES。
        data_dir: 原始 jsonl 目录，本机为 data/raw，容器内为 /mnt/data/exp1/data/raw。
        limit: 只取前若干条，None 为全量，供本机小样本自检。

    Returns:
        统一结构的字典列表，顺序与文件相同。

    Raises:
        ValueError: dataset 与 split 的组合不在 DATA_FILES 中。
    """
    if (dataset, split) not in DATA_FILES:
        raise ValueError(f"Unsupported dataset and split: {dataset}, {split}")

    data_file = os.path.join(data_dir, DATA_FILES[(dataset, split)])
    examples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            if dataset == "gsm8k":
                examples.append(_normalize_gsm8k(row, index, split))
            else:
                examples.append(_normalize_arc(row, index))
            if limit is not None and index + 1 >= limit:
                break
    return examples


def _read_dev_record(path=DEV_SPLIT_PATH):
    """读取开发集索引文件并校验内部自洽，返回整个记录。

    只校验索引列表本身，它与源文件是否对应由 load_train_split 核对。
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = record["indices"]
    if len(indices) != record["dev_size"]:
        raise ValueError(f"索引条数 {len(indices)} 与记录 {record['dev_size']} 不符")
    if len(set(indices)) != len(indices):
        raise ValueError("索引存在重复")
    if any(not 0 <= index < record["total"] for index in indices):
        raise ValueError("索引越界")
    return record


def load_dev_indices(path=DEV_SPLIT_PATH):
    """读取并校验开发集索引文件，返回索引列表，顺序与文件一致。"""
    return _read_dev_record(path)["indices"]


def load_train_split(data_dir, split_path=DEV_SPLIT_PATH):
    """载入 GSM8K 训练集并剔除开发集，得正式训练集 7073 条。

    先核对源文件校验和与索引文件所记是否一致。若源文件换版而索引未换，剔除的将是
    另一批样本，且不会报错，只会让开发集与训练集悄悄错位，故此处显式拦下。

    Args:
        data_dir: 原始 jsonl 目录。
        split_path: 开发集索引文件路径，默认取 exp1/configs/dev_split.json。

    Returns:
        统一结构的字典列表，不含开发集样本。

    Raises:
        ValueError: 源文件校验和与索引文件所记不符。
    """
    record = _read_dev_record(split_path)
    source = Path(data_dir) / record["source_file"]
    actual = sha256_of(source)
    if actual != record["source_sha256"]:
        raise ValueError(
            f"源文件校验和不符，索引记录 {record['source_sha256'][:16]}，实得 {actual[:16]}"
        )
    dev = set(record["indices"])
    return [e for e in load_examples("gsm8k", "train", data_dir) if e["index"] not in dev]


def build_messages(example):
    """构造用户消息，训练与评测共用的唯一模板来源。

    只返回 user 一轮。评测侧置 add_generation_prompt=True 直接用，训练侧在其后追加
    assistant 一轮，两侧的用户内容逐字节相同，这是风险 R8 的证据。

    GSM8K 为指令加题干。ARC 另接选项块，每行由标签、点号、空格与正文组成，标签取自
    labels，不用字母表生成。
    """
    question = example["question"]
    if example["dataset"] == "gsm8k":
        content = f"{GSM8K_INSTRUCTION}\n\n{question}"
    else:
        options = "\n".join(f"{label}. {text}" for label, text in zip(example["labels"], example["choices"]))
        content = f"{ARC_INSTRUCTION}\n\n{question}\n\n{options}"
    return [{"role": "user", "content": content}]


def render_prompt(tokenizer, example):
    """渲染成模型可直接读入的提示串，内部只调用 build_messages。

    tokenize 与 enable_thinking 均显式写出。前者免掉跨版本的默认值差异，后者使模型
    直接作答，依据见 NOTE/01-从文本到模型输入.md 第四节。
    """
    return tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def strip_calc_annotations(answer):
    """删除 GSM8K 答案中 <<计算式=结果>> 形式的标注，只留结果。

    48/2 = <<48/2=24>>24 改写为 48/2 = 24。正则 [^>]* 不跨右尖括号，故整个标注一次
    匹配。末行 #### 标记必须保留，指令与抽取的第一级规则都依赖它。
    """
    return re.sub(r"<<[^>]*>>", "", answer)


def build_target_text(example):
    """训练侧 assistant 轮的文本，取删除计算标注后的答案。

    末行 #### 标记予以保留。仅 GSM8K 有训练价值，ARC 只用于评测。
    """
    return strip_calc_annotations(example["gold_raw"])


def parse_gsm8k_gold(answer):
    """取最后一处 #### 之后的片段，只定位不规整。

    用 rpartition 取最后一处，与抽取模块同向。规整推迟到 extract.py 的
    normalize_number，在 is_correct 中对两侧同时施加，使规整实现全局仅一份，避免
    两侧分歧系统性偏移准确率，即风险 R10。代价是 gold 保留原文写法如 1,234，与 pred
    的 1234 字形不同却判为正确。

    标记缺失时返回 None，由调用方统计条数并判 FAIL。
    """
    marker = "####"
    if marker not in answer:
        return None
    return answer.rpartition(marker)[2].strip()
