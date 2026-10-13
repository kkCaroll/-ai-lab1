"""M3 P-Tuning v2。在每一层前置可训练的虚拟键值对，不改动基座权重。

虚拟 token 不是拼在输入序列之前，而是作为各层的键值对注入注意力，故输入序列本身不变长，
每一层都多出一段可训练前缀。层间注入是 P-Tuning v2 区别于 P-Tuning 的要点，后者只把软
提示拼在输入嵌入之前。

选中 peft 的 PrefixTuningConfig 而非 PromptEncoderConfig 即出于此。两者在 peft 中的命名
容易混淆，故于 2026-10-13 在一份四层的微型 Qwen3 上实测确认，PromptEncoderConfig 名为
P-Tuning，其输出仍拼在输入嵌入之前，前向的序列长度随虚拟 token 数等量增加；而
PrefixTuningConfig 不改输入长度，改在每一层的注意力缓存里多出与虚拟 token 等长的键值对。
按机制对应，前者即 P-Tuning，后者即 P-Tuning v2。证据见 LOG/02-问题与解决.md 的 Q10。

prefix_projection 为假时各层前缀直接作为可训练表优化，为真时另经一个小规模多层感知机生成。
P-Tuning v2 的结论是规模变大后重参数化并非必要，故本方法的默认取假，encoder_hidden_size
在该情形下不参与计算，仅在 prefix_projection 为真时生效。
"""

from __future__ import annotations

from pathlib import Path

from src.utils import parameter_counts

METHOD = "ptuning"
CONFIG_SCHEMA: dict[str, tuple[object, type]] = {
    "num_virtual_tokens": (20, int),
    "prefix_projection": (False, bool),
    "encoder_hidden_size": (512, int),
}


def build(model, config):
    """在每一层前置虚拟键值对，返回 PeftModel。"""
    from peft import PrefixTuningConfig, get_peft_model

    settings = PrefixTuningConfig(
        num_virtual_tokens=config["num_virtual_tokens"],
        prefix_projection=config["prefix_projection"],
        encoder_hidden_size=config["encoder_hidden_size"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, settings)


def describe(model, config):
    """返回本方法的描述。

    注入位置单列一项。该性质由实测确认，非运行时反推，故记为常量而非测量值，改 peft
    版本后须重新实测而非信此项。
    """
    return {
        "method": METHOD,
        "checkpoint_kind": "adapter",
        "config": dict(config),
        "prompt_injection": "per-layer",
        **parameter_counts(model),
    }


def extra_loss(model, batch, outputs, config):
    """本方法只用语言建模损失，不追加项。"""
    return None


def save(model, tokenizer, output_dir):
    """写出适配器权重与配置，返回所写文件的相对路径。

    虚拟 token 的嵌入表随适配器一并落盘，规模与虚拟 token 数成正比，通常不足一兆。
    tokenizer 参数在此不用，评测时它取自基座路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return sorted(str(item.relative_to(output_dir)) for item in output_dir.rglob("*") if item.is_file())
