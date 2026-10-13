"""M1 全量 SFT。基座全部参数参与训练。

本方法没有专属超参，配置段为空，配置文件只覆写 train 段。装配即返回原模型，只把全部
参数的 requires_grad 置为真，因为自预训练权重载入的模型未必处处可训。
"""

from __future__ import annotations

from pathlib import Path

from src.utils import parameter_counts

METHOD = "full_sft"
CONFIG_SCHEMA: dict[str, tuple[object, type]] = {}


def build(model, config):
    """返回原模型，全部参数置为可训练。"""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def describe(model, config):
    """返回本方法的描述。可训练量与总量在此相等，占比恒为一。"""
    return {
        "method": METHOD,
        "checkpoint_kind": "full",
        "config": dict(config),
        **parameter_counts(model),
    }


def extra_loss(model, batch, outputs, config):
    """本方法只用语言建模损失，不追加项。"""
    return None


def save(model, tokenizer, output_dir):
    """写出完整权重与分词器，返回所写文件的相对路径。

    检查点自带分词器，使评测以 --model-path 指向该目录即可加载，不必另传一个分词器
    路径，评测入口也不必为全参微调增加参数。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return sorted(str(item.relative_to(output_dir)) for item in output_dir.rglob("*") if item.is_file())
