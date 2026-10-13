"""训练方法的装配模块。

每个方法模块对外暴露四个函数与两个模块级常量，签名与语义完全一致，训练入口按方法名
取用，不为任一方法开特例。

    METHOD        字符串常量，与方法名及配置文件名同名
    CONFIG_SCHEMA 本方法配置段的键空间，形如 {键: (默认取值, 类型)}
    build(model, config) -> nn.Module
    describe(model, config) -> dict
    extra_loss(model, batch, outputs, config) -> Tensor 或 None
    save(model, tokenizer, output_dir) -> list[str]

config 只传本方法自己那一段。方法模块看不到 train 与 data 两段，也无从读到数据集路径，
风险 R6 的约束由接口结构保证而非靠自觉。

用函数而不用类，与 src 下其余模块的写法一致。M4 若需要冻结参考模型一类的状态，挂在
build 返回的模型上作为子模块即可，extra_loss 收到的正是同一个模型，取得到自己的状态。
此类辅助子模块须把 requires_grad 一律置假，且不计入 describe 报出的参数量，后者指参与
训练的参数。

describe 返回的字典含六类字段，即 method、checkpoint_kind、config、total_parameters、
trainable_parameters 与 trainable_ratio，checkpoint_kind 取 full 或 adapter，用以决定
评测时应传 --model-path 还是 --adapter。

本模块在导入期不引入任何子模块，故 peft 与 torch 不会被拖入，方法模块按名在需要时导入。
"""

from __future__ import annotations

import importlib

# 方法名到模块名的对应。全参微调在序列里写作 M1 且强调全参，模块名取通用的 sft，两者
# 同义而用词不同，故此处显式登记，不做名字推导。
METHOD_MODULES = {
    "full_sft": "sft",
    "lora": "lora",
    "ptuning": "ptuning",
    "custom": "custom",
}


def load_module(method: str):
    """按方法名导入其模块，方法名未登记时报错。"""
    if method not in METHOD_MODULES:
        raise KeyError(f"方法名 {method} 未登记，可取的为 {'、'.join(sorted(METHOD_MODULES))}")
    return importlib.import_module(f"src.methods.{METHOD_MODULES[method]}")
