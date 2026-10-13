"""src/config.py 的单元测试。

覆盖三层合并的优先级、命令行取值的类型推断与两类拒绝，即未声明的键路径与不符的类型。
这两类拒绝是配置校验里最容易静默失效的两处，拼错一个键名若被放过，扫参得出的结论会
在无人察觉的情况下成为错的，故用例写得比其他部分密。

末组用例直接读取仓库内五份真实配置文件，使 base.yaml 与各方法模块的 CONFIG_SCHEMA 之间
若出现漂移，此处即报错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config, methods

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _write(root: Path, name: str, text: str) -> Path:
    """把一段 YAML 写入临时目录，返回其路径。"""
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


BASE_WITH_LR = """
run:
  seed: 42
  limit: null
  max_steps: null
data:
  dataset: gsm8k
  split: train
  max_seq_len: 512
  dev_split: configs/dev_split.json
train:
  epochs: 3
  batch_size: 16
  grad_accum: 4
  lr: 1.0e-4
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_grad_norm: 1.0
  gradient_checkpointing: true
  dtype: auto
  log_every: 10
eval:
  every_epoch: true
  dataset: dev
  batch_size: 32
  max_new_tokens: 512
  limit: null
output:
  save_total_limit: 3
  keep_optimizer_state: false
"""

METHOD_YAML = """
method: lora
train:
  lr: 2.0e-4
lora:
  r: 8
"""


@pytest.fixture
def layered(tmp_path: Path) -> Path:
    """造一份最小的两层配置，返回方法配置文件的路径。"""
    _write(tmp_path, "base.yaml", BASE_WITH_LR)
    return _write(tmp_path, "lora.yaml", METHOD_YAML)


def test_load_merges_method_over_base(layered: Path):
    merged = config.load(layered)
    assert merged["method"] == "lora"
    assert merged["train"]["lr"] == 2.0e-4
    assert merged["train"]["epochs"] == 3
    assert merged["lora"]["r"] == 8


def test_load_does_not_mutate_base(layered: Path):
    config.load(layered)
    again = config.load(layered)
    assert again["train"]["lr"] == 2.0e-4


def test_override_beats_both_layers(layered: Path):
    merged = config.load(layered, ["train.lr=5.0e-5", "train.epochs=7"])
    assert merged["train"]["lr"] == 5.0e-5
    assert merged["train"]["epochs"] == 7


def test_override_accepts_method_path_pending_apply(layered: Path):
    merged = config.load(layered, ["lora.r=16"])
    assert merged["lora"]["r"] == 16


def test_override_rejects_unknown_base_path(layered: Path):
    with pytest.raises(config.ConfigError, match="不在 schema 内"):
        config.load(layered, ["train.learning_rate=1.0e-4"])


def test_override_rejects_unknown_section(layered: Path):
    with pytest.raises(config.ConfigError, match="不在 schema 内"):
        config.load(layered, ["optimizer.name=adamw"])


def test_override_requires_two_levels(layered: Path):
    with pytest.raises(config.ConfigError, match="两级形式"):
        config.load(layered, ["epochs=3"])


def test_override_requires_equals_sign(layered: Path):
    with pytest.raises(config.ConfigError, match="缺等号"):
        config.load(layered, ["train.lr"])


def test_unknown_section_in_config_file(layered: Path):
    path = _write(layered.parent, "bad.yaml", "method: lora\noptimizer:\n  name: adamw\n")
    with pytest.raises(config.ConfigError, match="未声明的段"):
        config.load(path)


def test_unknown_key_in_config_file(layered: Path):
    path = _write(layered.parent, "bad.yaml", "method: lora\ntrain:\n  learning_rate: 1.0e-4\n")
    with pytest.raises(config.ConfigError, match="learning_rate"):
        config.load(path)


def test_unknown_key_in_method_section(layered: Path):
    """方法段的键空间不在本模块内，故在 apply_method 处才拦下。"""
    path = _write(layered.parent, "bad.yaml", "method: lora\nlora:\n  rank: 16\n")
    merged = config.load(path)
    with pytest.raises(config.ConfigError, match="rank"):
        config.apply_method(merged, {"r": (8, int)})


def test_method_key_required(tmp_path: Path):
    _write(tmp_path, "base.yaml", BASE_WITH_LR)
    path = _write(tmp_path, "nokey.yaml", "train:\n  lr: 1.0e-4\n")
    with pytest.raises(config.ConfigError, match="method"):
        config.load(path)


def test_base_missing_key_rejected(tmp_path: Path):
    _write(tmp_path, "base.yaml", BASE_WITH_LR.replace("  log_every: 10\n", ""))
    path = _write(tmp_path, "lora.yaml", METHOD_YAML)
    with pytest.raises(config.ConfigError, match="log_every"):
        config.load(path)


def test_base_missing_section_rejected(tmp_path: Path):
    _write(tmp_path, "base.yaml", "run:\n  seed: 42\n")
    path = _write(tmp_path, "lora.yaml", METHOD_YAML)
    with pytest.raises(config.ConfigError, match="缺段"):
        config.load(path)


def test_type_mismatch_in_config_file(tmp_path: Path):
    _write(tmp_path, "base.yaml", BASE_WITH_LR)
    path = _write(tmp_path, "lora.yaml", "method: lora\ntrain:\n  epochs: many\n")
    with pytest.raises(config.ConfigError, match="train.epochs"):
        config.load(path)


def test_type_mismatch_from_override(layered: Path):
    with pytest.raises(config.ConfigError, match="train.epochs"):
        config.load(layered, ["train.epochs=three"])


def test_bool_is_not_accepted_as_int(layered: Path):
    with pytest.raises(config.ConfigError, match="train.epochs"):
        config.load(layered, ["train.epochs=true"])


def test_integer_is_accepted_as_float(layered: Path):
    merged = config.load(layered, ["train.lr=1"])
    assert merged["train"]["lr"] == 1
    config.check(merged)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16", 16),
        ("0", 0),
        ("-3", -3),
        ("2.0e-4", 2.0e-4),
        ("0.05", 0.05),
        ("true", True),
        ("False", False),
        ("null", None),
        ("cosine", "cosine"),
        ("q_proj", "q_proj"),
    ],
)
def test_parse_scalar(text: str, expected: object):
    assert config.parse_scalar(text) == expected


def test_parse_scalar_keeps_type_of_boolean_and_integer_distinct():
    assert config.parse_scalar("1") is not config.parse_scalar("true")
    assert isinstance(config.parse_scalar("1"), int)
    assert isinstance(config.parse_scalar("true"), bool)


def test_apply_method_fills_defaults(layered: Path):
    merged = config.load(layered)
    config.apply_method(merged, {"r": (8, int), "alpha": (16, int)})
    assert merged["lora"] == {"r": 8, "alpha": 16}


def test_apply_method_rejects_unknown_key(layered: Path):
    merged = config.load(layered, ["lora.rankk=16"])
    with pytest.raises(config.ConfigError, match="rankk"):
        config.apply_method(merged, {"r": (8, int)})


def test_apply_method_drops_empty_section(layered: Path):
    merged = config.load(layered)
    merged.pop("lora")
    config.apply_method(merged, {})
    assert "lora" not in merged


def test_check_rejects_non_gsm8k_training(layered: Path):
    merged = config.load(layered, ["data.dataset=arc"])
    with pytest.raises(config.ConfigError, match="只接受 gsm8k"):
        config.check(merged)


def test_check_rejects_test_split_in_mid_training(layered: Path):
    merged = config.load(layered, ["eval.dataset=test"])
    with pytest.raises(config.ConfigError, match="只接受 dev"):
        config.check(merged)


def test_check_rejects_missing_lr(layered: Path):
    merged = config.load(layered, ["train.lr=null"])
    with pytest.raises(config.ConfigError, match="train.lr 未填"):
        config.check(merged)


def test_check_rejects_zero_save_limit(layered: Path):
    merged = config.load(layered, ["output.save_total_limit=0"])
    with pytest.raises(config.ConfigError, match="save_total_limit"):
        config.check(merged)


def test_check_rejects_non_positive_limit(layered: Path):
    merged = config.load(layered, ["run.limit=0"])
    with pytest.raises(config.ConfigError, match="run.limit"):
        config.check(merged)


def test_check_accepts_defaults(layered: Path):
    config.check(config.load(layered))


@pytest.mark.parametrize("method", ["full_sft", "lora", "ptuning"])
def test_real_config_files_load(method: str):
    """仓库内真实配置文件须通过加载、装配与核对三步。

    模块名由 src.methods 的登记表给出，不按方法名推导，因为全参微调的模块名为 sft。
    """
    module = __import__(
        f"src.methods.{methods.METHOD_MODULES[method]}", fromlist=["CONFIG_SCHEMA"]
    )
    merged = config.load(CONFIG_DIR / f"{method}.yaml")
    config.apply_method(merged, module.CONFIG_SCHEMA)
    config.check(merged)
    assert merged["method"] == method
    assert merged["train"]["lr"] is not None


def test_real_custom_config_is_rejected():
    """M4 的配置尚未定值，train.lr 留空须在核对处被拦下。"""
    module = __import__("src.methods.custom", fromlist=["CONFIG_SCHEMA"])
    merged = config.load(CONFIG_DIR / "custom.yaml")
    config.apply_method(merged, module.CONFIG_SCHEMA)
    with pytest.raises(config.ConfigError, match="train.lr 未填"):
        config.check(merged)
