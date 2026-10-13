"""data.py 的单元测试。

只测纯函数，喂手写数据或临时文件，不读 data/ 也不加载分词器。原因是 data/ 不入
仓库，本机与容器路径不同，写死任一个都会在另一处失败。三份真实文件的行数与
schema 核对属于一次性命令，不进本测试。
"""

import hashlib
import json

import pytest

from src.data import (
    ARC_INSTRUCTION,
    DATA_COUNTS,
    DATA_FILES,
    GSM8K_INSTRUCTION,
    _normalize_arc,
    _normalize_gsm8k,
    build_messages,
    build_target_text,
    load_dev_indices,
    load_dev_split,
    load_examples,
    load_train_split,
    parse_gsm8k_gold,
    render_prompt,
    strip_calc_annotations,
    template_digest,
)

EXPECTED_KEYS = ["index", "id", "dataset", "question", "choices", "labels", "gold", "gold_raw"]

GSM8K_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)

GSM8K_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May.",
    "answer": GSM8K_ANSWER,
}

ARC_ROW = {
    "id": "Mercury_7175875",
    "question": "Which is the most likely effect of this increase in rotation?",
    "choices": {
        "text": ["Planetary density will decrease.", "Planetary days will become shorter."],
        "label": ["A", "B"],
    },
    "answerKey": "B",
}

ARC_NUM_ROW = {
    "id": "NYSEDREGENTS_2008_8_34",
    "question": "Which statement best describes the energy changes?",
    "choices": {"text": ["one", "two", "three", "four"], "label": ["1", "2", "3", "4"]},
    "answerKey": "2",
}


# ----------------- 计算标注剥离 ------------------


def test_strip_calc_annotations():
    """真实答案串上的改写结果，标注连同尖括号一并删除。"""
    assert strip_calc_annotations(GSM8K_ANSWER) == (
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72"
    )


def test_strip_keeps_currency_symbol():
    """货币符号不在删除范围内，留给抽取模块的数值规整处理。"""
    assert strip_calc_annotations("She makes 9 * 2 = $<<9*2=18>>18 a day.") == (
        "She makes 9 * 2 = $18 a day."
    )


def test_strip_keeps_final_marker():
    """末行 #### 标记必须保留，抽取的第一级规则依赖它。"""
    assert strip_calc_annotations("step\n#### 72").endswith("#### 72")


def test_strip_without_annotation_is_identity():
    """无标注时原样返回。"""
    assert strip_calc_annotations("plain text") == "plain text"


# ----------------- 标准答案定位 ------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("step\n#### 72", "72"),
        ("step\n####72", "72"),
        ("step\n#### 1,234", "1,234"),
        ("step\n#### $47", "$47"),
        ("step\n#### -3", "-3"),
        ("a\n#### 9\nb\n#### 12", "12"),
    ],
)
def test_parse_gsm8k_gold(answer, expected):
    """取最后一处 #### 之后的片段，只去首尾空白，不做数值规整。"""
    assert parse_gsm8k_gold(answer) == expected


def test_parse_gsm8k_gold_missing_marker():
    """标记缺失时返回 None，由调用方统计并判 FAIL。"""
    assert parse_gsm8k_gold("no marker here") is None


def test_parse_gsm8k_gold_keeps_thousands_separator():
    """千分位逗号保留，与 pred 的 1234 字形不同，等值判定在 is_correct 中做。"""
    assert parse_gsm8k_gold("step\n#### 1,234") == "1,234"


# ----------------- 归一化 ------------------


def test_normalize_gsm8k_schema():
    """八个字段齐备且顺序固定。"""
    example = _normalize_gsm8k(GSM8K_ROW, 7, "train")
    assert list(example) == EXPECTED_KEYS
    assert example["index"] == 7
    assert example["id"] == "gsm8k-train-0007"
    assert example["dataset"] == "gsm8k"
    assert example["choices"] is None
    assert example["labels"] is None
    assert example["gold"] == "72"
    assert example["gold_raw"] == GSM8K_ANSWER


def test_normalize_arc_schema():
    """八个字段齐备，choices 与 labels 从嵌套字典中拉平成平行列表。"""
    example = _normalize_arc(ARC_ROW, 3)
    assert list(example) == EXPECTED_KEYS
    assert example["index"] == 3
    assert example["id"] == "Mercury_7175875"
    assert example["dataset"] == "arc"
    assert example["choices"] == ARC_ROW["choices"]["text"]
    assert example["labels"] == ["A", "B"]
    assert example["gold"] == "B"
    assert example["gold_raw"] == "B"


def test_normalize_arc_numeric_labels():
    """数字标签原样保留，不替换成字母表。"""
    example = _normalize_arc(ARC_NUM_ROW, 0)
    assert example["labels"] == ["1", "2", "3", "4"]
    assert example["gold"] == "2"


# ----------------- 训练目标文本 ------------------


def test_build_target_text():
    """取剥离标注后的原文，末行标记保留。"""
    example = _normalize_gsm8k(GSM8K_ROW, 0, "test")
    assert build_target_text(example) == strip_calc_annotations(GSM8K_ANSWER)
    assert "<<" not in build_target_text(example)
    assert build_target_text(example).endswith("#### 72")


# ----------------- 消息构造 ------------------


def test_build_messages_single_user_turn():
    """恰好一轮，role 为 user，不含 system 与 assistant。"""
    messages = build_messages(_normalize_gsm8k(GSM8K_ROW, 0, "test"))
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert set(messages[0]) == {"role", "content"}


def test_build_messages_gsm8k_content():
    """指令接空行接题干，题干完整出现。"""
    example = _normalize_gsm8k(GSM8K_ROW, 0, "test")
    content = build_messages(example)[0]["content"]
    assert content == f"{GSM8K_INSTRUCTION}\n\n{GSM8K_ROW['question']}"


def test_build_messages_arc_has_labelled_options():
    """选项块逐行给出标签与正文，标签取自记录本身。"""
    example = _normalize_arc(ARC_ROW, 0)
    content = build_messages(example)[0]["content"]
    assert content.startswith(f"{ARC_INSTRUCTION}\n\n{ARC_ROW['question']}\n\n")
    assert content.endswith("A. Planetary density will decrease.\nB. Planetary days will become shorter.")
    assert "A. " in content and "B. " in content


def test_build_messages_arc_numeric_labels():
    """数字标签题的行首为数字，不退化成字母。"""
    content = build_messages(_normalize_arc(ARC_NUM_ROW, 0))[0]["content"]
    assert "\n1. one\n2. two\n3. three\n4. four" in content


# ----------------- 提示渲染的调用契约 ------------------


class FakeTokenizer:
    """记录 apply_chat_template 的入参，用于在不加载分词器时验证调用契约。"""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "RENDERED"


def test_render_prompt_call_contract():
    """模板参数固定为不做分词、加生成提示、关闭思考模式。"""
    tokenizer = FakeTokenizer()
    example = _normalize_gsm8k(GSM8K_ROW, 0, "test")
    assert render_prompt(tokenizer, example) == "RENDERED"
    messages, kwargs = tokenizer.calls[0]
    assert messages == build_messages(example)
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


# ----------------- 载入 ------------------


def test_load_examples_rejects_unknown_combination():
    """组合不在 DATA_FILES 中时立刻报错，不先去开文件。"""
    with pytest.raises(ValueError):
        load_examples("gsm8k", "dev", "no/such/directory")


def test_load_examples_gsm8k_from_tmp(tmp_path):
    """临时文件上测读入循环与 limit，index 恒为文件真实行号。"""
    rows = [{"question": f"q{i}", "answer": f"steps\n#### {i}"} for i in range(5)]
    (tmp_path / "gsm8k_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    full = load_examples("gsm8k", "test", str(tmp_path))
    assert [e["index"] for e in full] == [0, 1, 2, 3, 4]
    assert [e["gold"] for e in full] == ["0", "1", "2", "3", "4"]

    limited = load_examples("gsm8k", "test", str(tmp_path), limit=2)
    assert [e["index"] for e in limited] == [0, 1]


def test_load_examples_arc_from_tmp(tmp_path):
    """ARC 走另一条归一化分支，文件名含 challenge 一词。"""
    (tmp_path / "arc_challenge_test.jsonl").write_text(
        json.dumps(ARC_NUM_ROW) + "\n", encoding="utf-8"
    )

    examples = load_examples("arc", "test", str(tmp_path))
    assert len(examples) == 1
    assert examples[0]["id"] == "NYSEDREGENTS_2008_8_34"
    assert examples[0]["labels"] == ["1", "2", "3", "4"]
    assert examples[0]["gold"] == "2"


# ----------------- 常量自洽 ------------------


def test_data_counts_and_files_agree():
    """两份映射的键必须一致，改一处忘另一处会在这里失败。"""
    assert set(DATA_COUNTS) == set(DATA_FILES)
    assert DATA_COUNTS[("gsm8k", "train")] == 7473


# ----------------- 开发集索引与训练集剔除 ------------------


def _sha256(path):
    """独立用 hashlib 算校验和，不借被测函数自身的实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_train(tmp_path, rows=10):
    """写一份十行的最小训练集，答案形如 steps 加 #### i。"""
    source = tmp_path / "gsm8k_train.jsonl"
    body = "".join(
        json.dumps({"question": f"q{i}", "answer": f"steps\n#### {i}"}) + "\n" for i in range(rows)
    )
    source.write_text(body, encoding="utf-8")
    return source


def _write_split(tmp_path, indices, total, source, dev_size=None):
    """写一份索引文件，dev_size 留空时取索引条数，用于构造不一致的样本。"""
    record = {
        "dataset": "gsm8k",
        "split": "train",
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "seed": 42,
        "total": total,
        "dev_size": len(indices) if dev_size is None else dev_size,
        "train_size": total - len(indices),
        "indices": indices,
    }
    path = tmp_path / "dev_split.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_load_dev_indices_roundtrip(tmp_path):
    """读回索引列表，顺序与文件一致。"""
    source = _write_train(tmp_path)
    assert load_dev_indices(_write_split(tmp_path, [3, 1], 10, source)) == [3, 1]


def test_load_dev_indices_rejects_count_mismatch(tmp_path):
    """索引条数与 dev_size 记录不符时报错。"""
    source = _write_train(tmp_path)
    with pytest.raises(ValueError):
        load_dev_indices(_write_split(tmp_path, [1, 2], 10, source, dev_size=3))


def test_load_dev_indices_rejects_duplicates(tmp_path):
    """索引重复时报错。"""
    source = _write_train(tmp_path)
    with pytest.raises(ValueError):
        load_dev_indices(_write_split(tmp_path, [1, 1], 10, source))


def test_load_dev_indices_rejects_out_of_range(tmp_path):
    """索引越界时报错，总数 10 的合法索引到 9 为止。"""
    source = _write_train(tmp_path)
    with pytest.raises(ValueError):
        load_dev_indices(_write_split(tmp_path, [1, 10], 10, source))


def test_load_train_split_excludes_dev(tmp_path):
    """剔除后余 8 条，索引为原 10 条去掉 1 与 3。"""
    source = _write_train(tmp_path)
    path = _write_split(tmp_path, [1, 3], 10, source)
    kept = load_train_split(str(tmp_path), path)
    assert [e["index"] for e in kept] == [0, 2, 4, 5, 6, 7, 8, 9]
    assert all(e["dataset"] == "gsm8k" for e in kept)


def test_load_train_split_rejects_stale_checksum(tmp_path):
    """源文件与索引记录对不上时拦下，不静默剔除另一批样本。"""
    source = _write_train(tmp_path)
    path = _write_split(tmp_path, [1, 3], 10, source)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_train_split(str(tmp_path), path)


def test_load_dev_split_selects_dev(tmp_path):
    """开发集取中选而非剔除，索引为 1 与 3，且顺序与源文件相同。"""
    source = _write_train(tmp_path)
    path = _write_split(tmp_path, [3, 1], 10, source)
    dev = load_dev_split(str(tmp_path), path)
    assert [e["index"] for e in dev] == [1, 3]
    assert [e["gold"] for e in dev] == ["1", "3"]


def test_load_dev_split_and_train_split_are_complementary(tmp_path):
    """两部分并集为原文件全部样本且无交集。"""
    source = _write_train(tmp_path)
    path = _write_split(tmp_path, [3, 1], 10, source)
    dev = [e["index"] for e in load_dev_split(str(tmp_path), path)]
    kept = [e["index"] for e in load_train_split(str(tmp_path), path)]
    assert sorted(dev + kept) == list(range(10))
    assert not set(dev) & set(kept)


def test_load_dev_split_rejects_stale_checksum(tmp_path):
    """与训练集同源同校验，源文件换版即拦下。"""
    source = _write_train(tmp_path)
    path = _write_split(tmp_path, [1, 3], 10, source)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dev_split(str(tmp_path), path)


# ----------------- 模板指纹 ------------------


class _StubTokenizer:
    """只实现 apply_chat_template 的分词器替身，使指纹可在不加载模型时测。"""

    def __init__(self, suffix=""):
        self.suffix = suffix

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + self.suffix


def test_template_digest_is_stable():
    """同一分词器两次取指纹结果相同，且为 64 位十六进制。"""
    tokenizer = _StubTokenizer("<|assistant|>")
    digest = template_digest(tokenizer)
    assert digest == template_digest(tokenizer)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_template_digest_tracks_chat_template():
    """对话模板改动后指纹必须变化，否则该核对形同虚设。"""
    assert template_digest(_StubTokenizer("<|a|>")) != template_digest(_StubTokenizer("<|b|>"))
