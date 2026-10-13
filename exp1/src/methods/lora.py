"""M2 LoRA。冻结基座权重，只训练旁路低秩增量矩阵。

增量矩阵挂在注意力投影上，训练完毕后适配器只有几兆，与基座权重分离存放，故评测须同时
给出基座路径与适配器路径。
"""

from __future__ import annotations

from pathlib import Path

from src.utils import parameter_counts

METHOD = "lora"
CONFIG_SCHEMA: dict[str, tuple[object, type]] = {
    "r": (8, int),
    "alpha": (16, int),
    "dropout": (0.05, float),
    "target_modules": (["q_proj", "k_proj", "v_proj", "o_proj"], list),
    "bias": ("none", str),
}


def build(model, config):
    """在基座模型上挂载 LoRA 旁路，返回 PeftModel。

    基座权重保持冻结，只有旁路的低秩矩阵参与训练。target_modules 按模块名后缀匹配，
    故写 q_proj 即可命中各层同名的投影，不必逐层列出。
    """
    from peft import LoraConfig, get_peft_model

    settings = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=list(config["target_modules"]),
        bias=config["bias"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, settings)


def describe(model, config):
    """返回本方法的描述，另记实际挂载的模块名。

    模块名由 peft 按后缀展开，与配置段里写的后缀不是一回事，故单列一项供人工核对挂
    载范围是否与预期一致。
    """
    return {
        "method": METHOD,
        "checkpoint_kind": "adapter",
        "config": dict(config),
        "targeted_module_names": sorted(set(getattr(model, "targeted_module_names", []))),
        **parameter_counts(model),
    }


def extra_loss(model, batch, outputs, config):
    """本方法只用语言建模损失，不追加项。"""
    return None


def save(model, tokenizer, output_dir):
    """写出适配器权重与配置，返回所写文件的相对路径。

    适配器只有几兆，每轮都存也不占空间。tokenizer 参数在此不用，评测时它取自基座路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return sorted(str(item.relative_to(output_dir)) for item in output_dir.rglob("*") if item.is_file())
