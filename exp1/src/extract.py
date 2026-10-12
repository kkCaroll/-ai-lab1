"""答案抽取模块。

把模型输出的自由文本还原为可比较的离散答案。GSM8K 取数值，ARC 取选项标签，
两者各用一套按序命中的级联规则，首个成功即返回。规则宽严会系统性偏移准确率，
故每条规则有名字并计数，宽严取舍见 PLAN/02-风险与验证清单.md 的 R10。

不在顶层导入 torch 与 transformers，故可在不加载模型的机器上以毫秒级跑完单测。

用法
    result = extract_gsm8k(text)
    correct = is_correct(result.value, example["gold"], example["dataset"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from src.utils import FAIL, OK, WARN

EXTRACT_RULES_VERSION = "v2"

# raw 字段的截断长度，避免少数失控输出把明细文件撑大
RAW_LIMIT = 60

# 数值本体，含千分位、小数、前导小数点、分数与混合数，不含符号。分数与混合数
# 排在前面，否则 3/4 会被拆成末位 4，1 1/2 会被拆成末位 2
_NUMBER = r"\d+\s+\d+/\d+|\d+/\d+|\d{1,3}(?:,\d{3})+|\d*\.?\d+"
_SIGNED_NUMBER = rf"[-−]?(?:{_NUMBER})"

# 引导语之后四十字符内出现的数值，取最右一次命中
_CUE_RE = re.compile(
    r"(?:answer|total|altogether|equal(?:s| to)?|therefore|thus)\b[^0-9]{0,40}?("
    + _SIGNED_NUMBER
    + r")",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(_SIGNED_NUMBER)

# ARC 的作答引导语，如 Answer: B、answer is C
_ARC_CUE_RE = re.compile(r"(?:answer|option|choice)\b", re.IGNORECASE)
# 孤立的单个标签字符，两侧不得紧邻同类字符
_ARC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9])(?![A-Za-z0-9])")
# 全文末位的孤立标签，其后只允许非字母数字字符。模型只输出一个标签时命中此式。
# 要求落在末位，是因为句首的 A 与冠词无从区分，A car is moving 不得读作答 A
_ARC_FINAL_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9])[^A-Za-z0-9]*$")
# 输出开头的标签，如 B. Planetary days will become shorter。标号之后必须紧跟空白，
# 否则 3.5 m/s 一类数值会被读成标号 3。该式同时用于正文比对前剥去前置标签
_ARC_LEAD_RE = re.compile(r"^\s*[*_`\s]*[(\[]?([A-Za-z0-9])[)\]]?\s*[.):,]\s")

# 引导语之后取多长的窗口找标签。不用一条正则跨过 is、: 等连接成分，是因为这些成分
# 本身由字母组成，正则会把 is 的 i 当成标签
ARC_CUE_WINDOW = 24
# 与选项正文比对时只看输出开头这一段
ARC_HEAD_CHARS = 80
# 参与正文比对的选项至少这么长，过短会把任意文本都匹配上
ARC_CHOICE_MIN_CHARS = 4

# 数值附近可能出现但无意义的单位词
_UNIT_RE = re.compile(
    r"\b(?:dollars?|yuan|cents?|euros?|pounds?|rmb|usd|percent)\b", re.IGNORECASE
)
_CURRENCY_RE = re.compile(r"[$¥€£]")
_FRAC_RE = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_MIXED_RE = re.compile(r"^(-?\d+)\s+(\d+)/(\d+)$")
_LEADING_DOT_RE = re.compile(r"^(-?)\.(\d+)$")
# 包裹在数值外层的符号，成对出现但不改变取值。末端可删点号，首端不可，否则前导
# 小数点的写法 .5 会被削成 5。两套字符集分别用 strip 与仅尾部的正则以区分两端，
# 因为 strip 不论给出哪套字符都会同时作用于首尾
_PADDING_HEAD = "*_`\"'()[],;:"
_PADDING_TAIL_RE = re.compile(r"[*_`\"'()\[\],.;:]+$")


@dataclass
class ExtractResult:
    """一次抽取的结果。

    value 为命中的原始片段，尚未规整，规整由 normalize_number 在判定时对两侧
    同时施加。method 为命中的规则名，供统计与人工核对；raw 为命中处的上下文。
    """

    value: str | None
    status: str
    method: str
    raw: str = ""


def _truncate(text: str) -> str:
    """截断上下文，超出部分以省略号收尾。"""
    text = " ".join(text.split())
    return text if len(text) <= RAW_LIMIT else text[:RAW_LIMIT] + "..."


def _last_number(text: str) -> str | None:
    """返回文本中最后一处数值，无则返回 None。"""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _number_result(value: str, method: str, context: str) -> ExtractResult:
    """组装 GSM8K 结果，捕获到但无法规整为数值时判 WARN。"""
    status = OK if normalize_number(value) is not None else WARN
    return ExtractResult(value, status, method, _truncate(context))


def _label_result(value: str, valid_labels: list[str], method: str, context: str) -> ExtractResult:
    """组装 ARC 结果，标签不在本题标签集合内时判 WARN。

    本模块只在末位降级路径上传入集合外的标签，其余各级的调用点均已在取用前核对
    过集合归属。
    """
    status = OK if value in valid_labels else WARN
    return ExtractResult(value, status, method, _truncate(context))


def extract_gsm8k(text: str) -> ExtractResult:
    """按三级级联从自由文本中抽取数值，首个成功即返回。

    三级依次为末处 #### 标记之后的数值、引导语之后的数值、全文末处数值。第三级
    是兜底，可能取到推理途中的中间量，故命中计数须单独记录，格式命中率是判断
    输出形态是否达标的依据。

    标记之后取不到数值的样本继续下沉而不直接判失败，因为模型可能写了标记却把
    答案留在正文里。
    """
    marker = "####"
    if marker in text:
        fragment = text.rpartition(marker)[2].strip()
        value = _last_number(fragment)
        if value is not None:
            return _number_result(value, "marker", fragment)

    hits = list(_CUE_RE.finditer(text))
    if hits:
        last = hits[-1]
        return _number_result(last.group(1), "cue", text[last.start() : last.end()])

    value = _last_number(text)
    if value is not None:
        return _number_result(value, "tail-number", text[-RAW_LIMIT:])

    return ExtractResult(None, FAIL, "failed", _truncate(text))


def extract_arc(text: str, valid_labels: list[str], choices: list[str]) -> ExtractResult:
    """按四级级联抽取选项标签，首个命中本题标签集合者即返回。

    标签集合由题目自带，不假设 A 至 D，测试集中另有数字标签题与三选项、五选项
    题。四级依次为作答引导语之后的标签、全文末位孤立标签、输出开头与选项正文
    比对、输出开头的标签。

    前两级只认本题标签集合内的字符，取到集合外的字符不就此返回，而是记下备作
    降级并继续走完余下两级。依据是 R1-1 实测，模型多以标签点号接选项正文作答，
    短输出的末位常是正文末字母如 F 与 s，若在末位就此返回，本可由正文比对接住的
    样本会被记成降级或失败。四级均未命中本题标签时，才把记下的集合外字符按降级
    返回，取值仍落盘供人工核对。

    第二级只认末位，是因为孤立字母出现在句中时与冠词、代词无从区分，宁可判失败
    也不猜。失败计入抽取失败率并落盘，属可见的损失，误判则是把错答记成普通错答，
    两种损失中前一种可被察觉。
    """
    fallback = None

    for cue in reversed(list(_ARC_CUE_RE.finditer(text))):
        window = text[cue.end() : cue.end() + ARC_CUE_WINDOW]
        candidates = _ARC_TOKEN_RE.findall(window)
        legal = [item for item in candidates if item in valid_labels]
        if legal:
            return _label_result(legal[0], valid_labels, "answer-pattern", window)
        if candidates and fallback is None:
            fallback = (candidates[0], "answer-pattern", window)

    final = _ARC_FINAL_RE.search(text)
    if final is not None:
        value = final.group(1)
        if value in valid_labels:
            return _label_result(value, valid_labels, "bare-label", text[final.start() :])
        if fallback is None:
            fallback = (value, "bare-label", text[final.start() :])

    value = _match_option_text(text, valid_labels, choices)
    if value is not None:
        return _label_result(value, valid_labels, "option-text", text[:ARC_HEAD_CHARS])

    leading = _ARC_LEAD_RE.match(text)
    if leading is not None and leading.group(1) in valid_labels:
        return _label_result(leading.group(1), valid_labels, "leading-label", text[:ARC_HEAD_CHARS])

    if fallback is not None:
        value, method, context = fallback
        return _label_result(value, valid_labels, method, context)

    return ExtractResult(None, FAIL, "failed", _truncate(text))


def _normalize_option(text: str) -> str:
    """去掉非字母数字字符并转小写，供选项正文比对。"""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _match_option_text(text: str, valid_labels: list[str], choices: list[str]) -> str | None:
    """把输出开头与各选项正文比对，命中唯一选项则返回其标签。

    模型有时复述选项原文而不给标签，也可能写成标签加点号再接原文，两种形态都试，
    后者以 _ARC_LEAD_RE 剥去前置标签。
    命中多条时取最长的一条，长度并列则放弃，不在歧义处硬猜。过短的选项不参与
    比对，以免匹配到任意文本。
    """
    head_text = text[:ARC_HEAD_CHARS]

    for candidate in (head_text, _ARC_LEAD_RE.sub("", head_text)):
        head = _normalize_option(candidate)
        if not head:
            continue

        hits = []
        for label, choice in zip(valid_labels, choices):
            body = _normalize_option(choice)
            if len(body) >= ARC_CHOICE_MIN_CHARS and head.startswith(body):
                hits.append((len(body), label))
        if not hits:
            continue

        hits.sort(reverse=True)
        if len(hits) > 1 and hits[0][0] == hits[1][0]:
            return None
        return hits[0][1]

    return None


def normalize_number(fragment: str | None) -> str | None:
    """把一段数值文本规整为可比较的规范形式，失败返回 None。

    覆盖千分位逗号、货币符号与单位词、百分号、两种负号、分数与 \\frac 写法、
    混合数与前导小数点。规范形式取 Python 分数的字符串，使 0.75 与 3/4 归一为
    同一串，以此做等值判定而不引入浮点容差。百分号只剥离而不除以一百。

    千分位逗号只在其右侧恰为三位数字组时删除，3,4 这类无法判定的写法一律拒绝
    而不猜测。
    """
    if not fragment:
        return None

    text = fragment.strip().strip(_PADDING_HEAD)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = _FRAC_RE.sub(r"\1/\2", text)
    text = _CURRENCY_RE.sub("", text)
    text = _UNIT_RE.sub("", text)
    text = text.replace("%", "")
    text = _THOUSANDS_RE.sub("", text)
    text = _PADDING_TAIL_RE.sub("", text).strip()

    mixed = _MIXED_RE.match(text)
    if mixed:
        whole, numerator, denominator = (int(part) for part in mixed.groups())
        if denominator == 0:
            return None
        sign = -1 if whole < 0 else 1
        return str(Fraction(sign * (abs(whole) * denominator + numerator), denominator))

    text = _LEADING_DOT_RE.sub(r"\g<1>0.\g<2>", text)
    try:
        return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(pred: str | None, gold: str | None, dataset: str) -> bool:
    """判定抽取结果与标准答案是否等值，两侧施加同一套规整。

    GSM8K 规整后按规范串比较，故 72 与 72.0 与 $72 均判等值。ARC 只做去空白与
    大写化，标签本身不参与数值规整。pred 为 None 时判否，与准确率以全样本为
    分母、抽取失败计为错误一致。
    """
    if pred is None or gold is None:
        return False

    if dataset == "arc":
        return str(pred).strip().upper() == str(gold).strip().upper()

    left = normalize_number(str(pred))
    right = normalize_number(str(gold))
    return left is not None and left == right
