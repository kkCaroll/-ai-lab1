"""评测入口的参数校验测试。

只覆盖启动阶段即返回的路径，不加载模型，故与 test_train.py 同属导入期依赖 torch 的一份。
开发集一值的样本载入路径由 P2.5 的本机自检覆盖，不适合进单测，因为那需要真实权重。
"""

from __future__ import annotations

import sys

from src import eval as eval_module


def _argv(*extra: str) -> list[str]:
    return ["eval", "--model-path", "m", "--output-dir", "o", *extra]


def test_parse_args_accepts_dev_split(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--dataset", "gsm8k", "--split", "dev"))
    assert eval_module._parse_args().split == "dev"


def test_main_rejects_dev_split_with_arc(monkeypatch, capsys):
    # 开发集只由 GSM8K 的训练集划出，与 ARC 并列属参数错误，应在载入模型之前拦下
    monkeypatch.setattr(sys, "argv", _argv("--dataset", "arc", "--split", "dev"))
    assert eval_module.main() == 2
    assert "--split dev" in capsys.readouterr().out


def test_main_rejects_unknown_dataset(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _argv("--dataset", "mbpp"))
    assert eval_module.main() == 2
    assert "取值不合法" in capsys.readouterr().out
