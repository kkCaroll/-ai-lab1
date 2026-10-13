"""train.py 纯逻辑部分的单元测试。

只测三处，即文本编码、批次拼装与轮内分批。这三处是本模块里写错不会报错的部分，右填充
若写成左填充、标签若漏置 -100、分批若丢样本，训练都会照常跑完而读数整体偏移，故用例
写得比其余部分密。

不加载模型与分词器，分词器由替身顶替，故无需权重即可运行。
"""

from __future__ import annotations

import torch

from src import data
from src.train import IGNORE_INDEX, _encode_example, _epoch_groups, _make_batch

EXAMPLE = {"dataset": "gsm8k", "question": "1+1?", "gold_raw": "两数相加\n#### 2"}


class _StubTokenizer:
    """按字符映射编号的替身，使输入的切分点可以预先算出。"""

    eos_token = "<|end|>"

    def apply_chat_template(self, messages, **kwargs):
        return "<|user|>" + messages[0]["content"] + "<|assistant|>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [46 if c == "." else ord(c) % 40 for c in text]}


def _prompt_ids(tokenizer):
    """按评测侧的同一条路径复算提示编号，用作前缀比对的基准。"""
    return tokenizer(data.render_prompt(tokenizer, EXAMPLE), add_special_tokens=False)["input_ids"]


# ----------------- 文本编码 ------------------


def test_encode_labels_are_masked_over_prompt():
    """提示段的标签一律置 -100，目标段与输入逐位相同。"""
    tokenizer = _StubTokenizer()
    input_ids, labels = _encode_example(tokenizer, EXAMPLE)
    prompt_length = len(_prompt_ids(tokenizer))

    assert len(input_ids) == len(labels)
    assert labels[:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert labels[prompt_length:] == input_ids[prompt_length:]
    assert len(input_ids) > prompt_length


def test_encode_keeps_prompt_prefix_identical_to_eval_side():
    """输入的前缀须与单独分词的提示逐位相同，这是两侧同源的可核对形态。

    对齐若只做到字符串层面而分词结果不同，评测侧所见提示与训练侧的前缀就不是同一串编号，
    R8 名存实亡，故此处按编号比对而不按字符串比对。
    """
    tokenizer = _StubTokenizer()
    input_ids, _ = _encode_example(tokenizer, EXAMPLE)
    expected = _prompt_ids(tokenizer)
    assert input_ids[: len(expected)] == expected


def test_encode_appends_end_marker():
    """目标之后须接结束标记，缺它则模型学不到在何处停笔。"""
    tokenizer = _StubTokenizer()
    input_ids, labels = _encode_example(tokenizer, EXAMPLE)
    marker = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    assert input_ids[-len(marker) :] == marker
    assert labels[-len(marker) :] == marker


# ----------------- 批次拼装 ------------------


def _rows(*lengths):
    """造若干条的编码结果，编号按条区分，便于核验对齐。"""
    return [
        {"input_ids": [index] * length, "labels": [index] * length}
        for index, length in enumerate(lengths)
    ]


def test_make_batch_pads_on_the_right():
    """填充补在右侧，即真实内容占前段而填充符占后段。"""
    rows = _rows(2, 4)
    input_ids, attention_mask, _ = _make_batch(rows, pad_id=99, device=torch.device("cpu"))
    assert input_ids.shape == (2, 4)
    assert input_ids[0].tolist() == [0, 0, 99, 99]
    assert input_ids[1].tolist() == [1, 1, 1, 1]
    assert attention_mask[0].tolist() == [1, 1, 0, 0]


def test_make_batch_masks_padding_in_labels():
    """填充位的标签置 -100，短序列不以填充符为预测目标。"""
    rows = _rows(2, 4)
    _, _, labels = _make_batch(rows, pad_id=99, device=torch.device("cpu"))
    assert labels[0].tolist() == [0, 0, IGNORE_INDEX, IGNORE_INDEX]
    assert labels[1].tolist() == [1, 1, 1, 1]


def test_make_batch_width_is_batch_maximum():
    """宽度取本批最长一条，不取全局上限，否则大半算力耗在填充符上。"""
    _, _, labels = _make_batch(_rows(3, 5, 1), pad_id=99, device=torch.device("cpu"))
    assert labels.shape == (3, 5)
    assert labels[2].tolist() == [2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_make_batch_single_row_has_no_padding():
    """批内只有一条时不应产生填充位。"""
    input_ids, _, labels = _make_batch(_rows(3), pad_id=99, device=torch.device("cpu"))
    assert input_ids.tolist() == [[0, 0, 0]]
    assert labels.tolist() == [[0, 0, 0]]


# ----------------- 轮内分批 ------------------


def test_epoch_groups_cover_every_sample_once():
    """分组的并集恰为全部样本且无重无漏，丢样本不会显性报错，故须逐项核对。"""
    groups = _epoch_groups(10, batch_size=3, grad_accum=2, generator=torch.Generator().manual_seed(0))
    flat = [item for group in groups for micro in group for item in micro]
    assert sorted(flat) == list(range(10))
    assert all(len(micro) <= 3 for group in groups for micro in group)
    assert all(len(group) <= 2 for group in groups)


def test_epoch_groups_last_group_is_partial():
    """末组不满时保留其实际条数，据此等分梯度而不按 grad_accum。"""
    groups = _epoch_groups(9, batch_size=2, grad_accum=4, generator=torch.Generator().manual_seed(0))
    assert [len(group) for group in groups] == [4, 1]


def test_epoch_groups_are_reproducible_from_the_seed():
    """生成器种子相同则轮间次序相同，训练可复现。"""
    first = _epoch_groups(12, 2, 3, torch.Generator().manual_seed(7))
    second = _epoch_groups(12, 2, 3, torch.Generator().manual_seed(7))
    assert first == second


def test_epoch_groups_shuffle_actually_changes_order():
    """两个不同种子的次序不应一致，否则打乱这一步形同虚设。"""
    first = _epoch_groups(16, 4, 2, torch.Generator().manual_seed(1))
    second = _epoch_groups(16, 4, 2, torch.Generator().manual_seed(2))
    assert first != second
