"""训练配置的加载、合并与校验。

配置分三层，后者覆盖前者，逐段深合并而非整段替换。

    configs/base.yaml        五种方法共用的默认值
    configs/<method>.yaml    方法专属超参，段名与方法名同名
    命令行 --set 键路径=值    临时覆盖，供 P2.5 扫参与本机自检

基座段的键空间由 BASE_SCHEMA 声明，方法段的键空间由各方法模块的 CONFIG_SCHEMA 声明。
指到未声明过的键路径一律报错中止。拼错一个键而静默无效，会使扫参得出的结论成为错的，
故此处不作宽容。

方法段的键空间不在本模块内，因为方法模块在导入期即依赖 peft 与 torch，而本模块要求
不导入 torch，以便在处理器上以毫秒级验证合并与校验两处逻辑。故加载分两步，load 负责
两层文件与命令行覆盖，apply_method 由调用方在导入方法模块之后调用，补上方法段的校验
与默认填充。

用法示例
    from src import config

    merged = config.load(Path("configs/lora.yaml"), ["train.lr=1.0e-4"])
    config.apply_method(merged, lora.CONFIG_SCHEMA)
    config.check(merged)
"""

from __future__ import annotations

from pathlib import Path

import yaml

BASE_NAME = "base.yaml"
METHOD_KEY = "method"
OVERRIDE_SEPARATOR = "="
PATH_SEPARATOR = "."

# 基座段的键空间，键为段名，值为该段内每个键的默认取值与类型。默认取值置 None 表示
# 该键未定，须由方法段或命令行给出，由 check 判定，不使未填的配置静默跑完。
BASE_SCHEMA: dict[str, dict[str, tuple[object, type]]] = {
    "run": {
        "seed": (42, int),
        "limit": (None, int),
        "max_steps": (None, int),
    },
    "data": {
        "dataset": ("gsm8k", str),
        "split": ("train", str),
        "max_seq_len": (512, int),
        "dev_split": ("configs/dev_split.json", str),
    },
    "train": {
        "epochs": (3, int),
        "batch_size": (16, int),
        "grad_accum": (4, int),
        "lr": (None, float),
        "weight_decay": (0.0, float),
        "warmup_ratio": (0.03, float),
        "lr_scheduler": ("cosine", str),
        "max_grad_norm": (1.0, float),
        "gradient_checkpointing": (True, bool),
        "dtype": ("auto", str),
        "log_every": (10, int),
    },
    "eval": {
        "every_epoch": (True, bool),
        "dataset": ("dev", str),
        "batch_size": (32, int),
        "max_new_tokens": (512, int),
        "limit": (None, int),
    },
    "output": {
        "save_total_limit": (3, int),
        "keep_optimizer_state": (False, bool),
    },
}


class ConfigError(Exception):
    """配置不合法，含未知键路径、类型不符与跨段取值冲突三类。"""


def _describe(value: object) -> str:
    """把取值写成简短短语，供错误说明引用。"""
    if isinstance(value, str):
        return f"字符串 {value}"
    if isinstance(value, list):
        return f"列表 {len(value)} 项"
    if isinstance(value, dict):
        return f"映射 {len(value)} 项"
    return f"{type(value).__name__} {value}"


def parse_scalar(text: str) -> object:
    """按字面推断 --set 取值的类型。

    依次尝试布尔、空值与数值，均不成则按字符串。布尔排在最前，因为 true 与 false 都
    不是合法整数，虽不会误入数值分支，但顺序写明可免去依赖这一点。
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(stripped)
        except ValueError:
            continue
    return stripped


def _read_yaml(path: Path) -> dict:
    """读取一个 YAML 文件，返回字典。文件缺失或顶层不是映射即报错。"""
    if not path.is_file():
        raise ConfigError(f"配置文件不存在 {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} 的顶层不是映射")
    return loaded


def _deep_merge(base: dict, overlay: dict) -> dict:
    """逐段深合并，后者覆盖前者，返回新字典，不改动入参。"""
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _check_type(path: str, value: object, expected: type, errors: list[str]) -> None:
    """核对一个叶子的取值类型，不符时记入错误列表。

    空值放行，语义为未定，留待 check 判定。布尔须先拦下，因为 Python 把布尔视为整数
    的子类，不先拦则 true 会被认作合法的整数。整数可作浮点，反向不行。
    """
    if value is None:
        return
    if isinstance(value, bool):
        if expected is not bool:
            errors.append(f"{path} 为布尔 {value}，期望 {expected.__name__}")
        return
    if expected is float and isinstance(value, int):
        return
    if not isinstance(value, expected):
        errors.append(f"{path} 为{_describe(value)}，期望 {expected.__name__}")


def _validate_section(name: str, values: dict, schema: dict, errors: list[str]) -> None:
    """核对一个段的键空间与各键的取值类型。"""
    unknown = sorted(set(values) - set(schema))
    if unknown:
        allowed = "、".join(sorted(schema)) or "空"
        errors.append(f"段 {name} 含未声明的键 {'、'.join(unknown)}，可取的为 {allowed}")
    for key, (_, expected) in schema.items():
        if key in values:
            _check_type(f"{name}.{key}", values[key], expected, errors)


def apply_overrides(config: dict, overrides: list[str], method: str) -> dict:
    """把命令行覆盖项写入配置，返回同一字典。

    基座段的键路径逐一核对。方法段的键空间不在本模块内，故其键路径只核对段名，内容
    留待 apply_method 校验。
    """
    allowed = {
        f"{section}{PATH_SEPARATOR}{key}" for section, keys in BASE_SCHEMA.items() for key in keys
    }
    for item in overrides:
        if OVERRIDE_SEPARATOR not in item:
            raise ConfigError(f"--set {item} 缺等号，形式为 键路径=值")
        path, _, text = item.partition(OVERRIDE_SEPARATOR)
        path = path.strip()
        section, separator, key = path.partition(PATH_SEPARATOR)
        if not separator or not section or not key:
            raise ConfigError(f"--set {path} 须为 段名.键名 的两级形式")
        if path not in allowed and section != method:
            raise ConfigError(
                f"--set {path} 不在 schema 内，基座段可取的为 {'、'.join(sorted(allowed))}"
            )
        config.setdefault(section, {})[key] = parse_scalar(text)
    return config


def load(config_path: Path, overrides: list[str] | None = None, base_path: Path | None = None) -> dict:
    """读入两层配置并套用命令行覆盖，返回合并后的配置。

    base_path 留空时取 config_path 同目录下的 base.yaml，使本机与容器两处都不必记住
    它的位置。方法名以配置文件内的 method 键为唯一真源，命令行不再另传一份。
    """
    base_path = base_path or config_path.parent / BASE_NAME
    base = _read_yaml(base_path)
    layer = _read_yaml(config_path)

    errors: list[str] = []
    for section, schema in BASE_SCHEMA.items():
        values = base.get(section)
        if not isinstance(values, dict):
            errors.append(f"{base_path} 缺段 {section}")
            continue
        missing = sorted(set(schema) - set(values))
        if missing:
            errors.append(f"{base_path} 的段 {section} 缺键 {'、'.join(missing)}")
        _validate_section(section, values, schema, errors)
    if errors:
        raise ConfigError("；".join(errors))

    method = layer.get(METHOD_KEY)
    if not isinstance(method, str) or not method:
        raise ConfigError(f"{config_path} 缺 {METHOD_KEY} 键，方法名以该键为唯一真源")

    merged = _deep_merge(base, layer)
    apply_overrides(merged, overrides or [], method)

    unknown = sorted(set(merged) - set(BASE_SCHEMA) - {METHOD_KEY, method})
    if unknown:
        raise ConfigError(
            f"{config_path} 含未声明的段 {'、'.join(unknown)}，"
            f"可取的为 {'、'.join(sorted(BASE_SCHEMA))} 与 {method}"
        )

    errors = []
    for section, schema in BASE_SCHEMA.items():
        _validate_section(section, merged[section], schema, errors)
    if errors:
        raise ConfigError("；".join(errors))
    return merged


def apply_method(config: dict, schema: dict) -> dict:
    """按方法模块声明的键空间校验并填充方法段，返回同一字典。

    方法段为空且 schema 也为空时不留空段，使全参微调一类的配置不产生无内容的键。
    """
    name = config[METHOD_KEY]
    section = dict(config.get(name) or {})
    errors: list[str] = []
    _validate_section(name, section, schema, errors)
    if errors:
        raise ConfigError("；".join(errors))
    for key, (default, _) in schema.items():
        section.setdefault(key, default)
    if section:
        config[name] = section
    else:
        config.pop(name, None)
    return config


def check(config: dict) -> None:
    """跨段一致性核对，任一条不成立即报错中止。

    三条取自风险清单。data.dataset 只接受 gsm8k，训练侧不得触及 ARC-Challenge，见 R6。
    eval.dataset 只接受 dev，训练过程中的任何评测都不得触及测试集，见 R7。train.lr 不
    得留空，未填的配置不应静默跑完。
    """
    if config["data"]["dataset"] != "gsm8k":
        raise ConfigError(f"data.dataset 为 {config['data']['dataset']}，训练只接受 gsm8k")
    if config["eval"]["dataset"] != "dev":
        raise ConfigError(
            f"eval.dataset 为 {config['eval']['dataset']}，中间评测只接受 dev，见 R7"
        )
    if config["train"]["lr"] is None:
        raise ConfigError("train.lr 未填，须由方法段或 --set 给出")
    if config["output"]["save_total_limit"] < 1:
        raise ConfigError(f"output.save_total_limit 为 {config['output']['save_total_limit']}，至少为 1")
    for name in ("limit", "max_steps"):
        value = config["run"][name]
        if value is not None and value < 1:
            raise ConfigError(f"run.{name} 为 {value}，至少为 1")
