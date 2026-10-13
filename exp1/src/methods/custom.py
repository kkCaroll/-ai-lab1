"""M4 自研方法。方案于 P3.1 依 P2 结果确定后实现。

候选方案见 PLAN/02-风险与验证清单.md 的 R6，限定为降低 LoRA rank 或可训练参数占比、以
基座输出做 KL 正则、参数空间约束、早停四类，且不得使用 ARC-Challenge 数据。

本文件先占位，四个函数一律抛出未实现，使配置里 method 取 custom 时在装配处即中止，不会
静默跑出一份无意义的读数。方案确定后按 src/methods/__init__.py 的契约补齐，其中 KL 一类
需要冻结参考模型的方案，把参考模型挂在本模块自己的装配结果上，其参数一律置 requires_grad
为假，且不计入 describe 报出的参数量。
"""

from __future__ import annotations

METHOD = "custom"
CONFIG_SCHEMA: dict[str, tuple[object, type]] = {}

_UNIMPLEMENTED = "M4 的方案于 P3.1 确定后实现，见 PLAN/02-风险与验证清单.md 的 R6"


def build(model, config):
    """待实现。"""
    raise NotImplementedError(_UNIMPLEMENTED)


def describe(model, config):
    """待实现。"""
    raise NotImplementedError(_UNIMPLEMENTED)


def extra_loss(model, batch, outputs, config):
    """待实现。"""
    raise NotImplementedError(_UNIMPLEMENTED)


def save(model, tokenizer, output_dir):
    """待实现。"""
    raise NotImplementedError(_UNIMPLEMENTED)
