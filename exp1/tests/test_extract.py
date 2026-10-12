"""extract.py 的单元测试。

只测纯字符串规则，不加载模型也不读 data/，故可在任何机器上以毫秒级跑完。抽取
规则过严或过松都会系统性偏移准确率且不报错，是风险 R10，故每条规则都有正例、
反例与边界三类用例。

用例中的字符串为手写构造或取自 LOG/02-问题与解决.md Q2 记录的真实输出。另有二十八
条真实输出冻结于 tests/fixtures/real_outputs.jsonl，覆盖七个数据集与规则名的组合，
每条都经人工核对，规则再改动而抽取值偏离冻结值时会在此暴露。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.extract import (
    EXTRACT_RULES_VERSION,
    RAW_LIMIT,
    extract_arc,
    extract_gsm8k,
    is_correct,
    normalize_number,
)
from src.utils import FAIL, OK, WARN

EXP_ROOT = Path(__file__).resolve().parent.parent

# Q2 记录的真实输出，基座模型未经格式训练，全文不含 #### 标记
Q2_REAL_OUTPUT = "**Answer:** Mia pays **47 yuan** in total"

# 首次 DLC 运行后冻结，未生成时相关用例跳过
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "real_outputs.jsonl"

ARC_LABELS = ["A", "B", "C", "D"]
ARC_CHOICES = [
    "Planetary density will decrease.",
    "Planetary days will become shorter.",
    "The planet will stop rotating.",
    "The planet will rotate faster.",
]

# 测试集中有 22 题的标签为数字，故另备一组
NUMERIC_LABELS = ["1", "2", "3", "4"]
NUMERIC_CHOICES = ["Mercury", "Venus", "Earth", "Mars"]


# ----------------- 数值规整 ------------------


@pytest.mark.parametrize(
    "fragment, expected",
    [
        ("72", "72"),
        ("$47", "47"),
        ("1,234", "1234"),
        ("50%", "50"),
        ("-3", "-3"),
        ("−5", "-5"),
        ("(47)", "47"),
        ("**47**", "47"),
        ("47 yuan", "47"),
        ("72.", "72"),
        ("  72  ", "72"),
    ],
)
def test_normalize_plain_numbers(fragment, expected):
    """整数连同外层的货币符、单位词与包裹符，规整后只留数值本身。"""
    assert normalize_number(fragment) == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        ("0.75", "3/4"),
        ("3/4", "3/4"),
        (".5", "1/2"),
        ("-.5", "-1/2"),
        ("1 1/2", "3/2"),
        ("1.5", "3/2"),
        (r"\frac{3}{4}", "3/4"),
    ],
)
def test_normalize_fractions(fragment, expected):
    """小数与分数归一为同一规范串，等值判定因此不必引入浮点容差。"""
    assert normalize_number(fragment) == expected


@pytest.mark.parametrize(
    "fragment",
    ["3,4", "1,2345", "3/0", "abc", "", "   ", "3 apples", "She earns 5 dollars", None],
)
def test_normalize_rejects(fragment):
    """无法判定的写法返回 None，不猜测。"""
    assert normalize_number(fragment) is None


def test_normalize_thousands_separator_needs_three_digits():
    """逗号只在其右侧恰为三位数字组时删除，避免把 3,4 误读成 34。"""
    assert normalize_number("1,234") == "1234"
    assert normalize_number("3,4") is None


def test_normalize_keeps_leading_dot():
    """前导小数点有意义，.5 不得被削成 5。"""
    assert normalize_number(".5") == "1/2"


# ----------------- GSM8K 抽取 ------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("steps\n#### 72", "72"),
        ("steps\n####72", "72"),
        ("steps\n#### 1,234", "1,234"),
        ("steps\n#### $47", "47"),
        ("steps\n#### 50%", "50"),
        ("steps\n#### -3", "-3"),
        ("steps\n#### 3/4", "3/4"),
        ("steps\n#### 0.5", "0.5"),
        ("steps\n#### 72.", "72"),
    ],
)
def test_extract_gsm8k_marker(text, expected):
    """标记级取末处 #### 之后的数值，原样保留，规整留给判定环节。"""
    result = extract_gsm8k(text)
    assert (result.status, result.method) == (OK, "marker")
    assert result.value == expected


def test_extract_gsm8k_marker_beats_cue():
    """标记级优先于引导语级，两级都能命中时取标记。"""
    result = extract_gsm8k("The answer is 50.\n#### 72")
    assert (result.method, result.value) == ("marker", "72")


def test_extract_gsm8k_cue_on_real_output():
    """Q2 的真实输出不含 #### 标记，由引导语级接住。"""
    result = extract_gsm8k(Q2_REAL_OUTPUT)
    assert (result.status, result.method, result.value) == (OK, "cue", "47")


def test_extract_gsm8k_cue_beats_tail():
    """引导语级优先于末值级，末句另有数值时仍取引导语后的答案。"""
    result = extract_gsm8k("The answer is 47. That is 3 less than 50.")
    assert (result.method, result.value) == ("cue", "47")


def test_extract_gsm8k_year_is_ignored():
    """正文中的年份不被取用，由引导语级先行接住。"""
    result = extract_gsm8k("In 2008 she had 12 apples. Final answer: 24.")
    assert (result.method, result.value) == ("cue", "24")


def test_extract_gsm8k_tail_number():
    """无标记也无引导语时取全文末值。"""
    result = extract_gsm8k("She has 3 cats and 12 dogs, so 15")
    assert (result.status, result.method, result.value) == (OK, "tail-number", "15")


def test_extract_gsm8k_empty_marker_falls_through():
    """写了标记但标记后无数值时下沉到下一级，不直接判失败。"""
    result = extract_gsm8k("Steps here 5 and 4.\n#### ")
    assert (result.method, result.value) == ("tail-number", "4")


def test_extract_gsm8k_no_number_fails():
    """全文无数值时判失败，调用方据此计入抽取失败率。"""
    result = extract_gsm8k("I cannot solve this problem.")
    assert (result.status, result.method, result.value) == (FAIL, "failed", None)


def test_extract_gsm8k_empty_text_fails():
    """空串同样判失败，不抛异常。"""
    assert extract_gsm8k("").status == FAIL


def test_extract_gsm8k_unicode_minus_equals_ascii():
    """U+2212 负号在判定时与 ASCII 连字符等值。"""
    result = extract_gsm8k("#### −5")
    assert result.value == "−5"
    assert is_correct(result.value, "-5", "gsm8k")


def test_extract_gsm8k_raw_is_truncated():
    """raw 截断至 RAW_LIMIT，避免少数失控输出把明细文件撑大。"""
    result = extract_gsm8k("#### 7 " + "z" * 100)
    assert result.value == "7"
    assert len(result.raw) == RAW_LIMIT + 3
    assert result.raw.endswith("...")


# ----------------- ARC 抽取 ------------------


def test_extract_arc_answer_pattern_colon():
    """Answer 加冒号的显式作答。"""
    result = extract_arc("Answer: B", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "answer-pattern", "B")


def test_extract_arc_does_not_take_verb_letter():
    """引导语后的连接成分由字母组成，不得把 is 的首字母当成答案。"""
    result = extract_arc("The answer is C because of the rotation.", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "answer-pattern", "C")


def test_extract_arc_markdown_wrapped():
    """标签被 Markdown 强调符包裹时仍可取出。"""
    result = extract_arc("**Answer:** **D**", ARC_LABELS, ARC_CHOICES)
    assert (result.method, result.value) == ("answer-pattern", "D")


def test_extract_arc_bare_label_at_end():
    """只输出一个标签时由末位级接住，尾随标点不影响。"""
    result = extract_arc("B.", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "bare-label", "B")


def test_extract_arc_article_is_not_a_label():
    """句首的 A 与冠词无从区分，不得读作答 A。"""
    result = extract_arc("A car is moving down the road.", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (FAIL, "failed", None)


def test_extract_arc_option_text_echo():
    """复述选项原文而不给标签时，映射回该选项的标签。"""
    result = extract_arc("Planetary days will become shorter.", ARC_LABELS, ARC_CHOICES)
    assert (result.method, result.value) == ("option-text", "B")


def test_extract_arc_option_text_echo_with_leading_label():
    """先给标签再复述原文的形态同样命中正文比对。"""
    result = extract_arc("B. Planetary days will become shorter.", ARC_LABELS, ARC_CHOICES)
    assert (result.method, result.value) == ("option-text", "B")


def test_extract_arc_numeric_labels():
    """数字标签题由题目自带标签集合驱动，不硬编码字母表。"""
    result = extract_arc("2", NUMERIC_LABELS, NUMERIC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "bare-label", "2")


def test_extract_arc_numeric_labels_in_cue():
    """数字标签同样可在引导语之后取出。"""
    result = extract_arc("The answer is 3", NUMERIC_LABELS, NUMERIC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "answer-pattern", "3")


def test_extract_arc_label_outside_valid_set_warns():
    """三选项题被答成 D 时降级为 WARN，取值仍记下供人工核对。

    集合外的字符不再就地返回，而是记下备作降级，待四级走完仍无命中时才取出。
    """
    result = extract_arc("Answer: D", ARC_LABELS[:3], ARC_CHOICES[:3])
    assert (result.status, result.method, result.value) == (WARN, "answer-pattern", "D")


def test_extract_arc_earlier_legal_cue_wins():
    """末次引导语窗口内只有集合外字符时回溯至更早的引导语。"""
    result = extract_arc(
        "The answer is B. However, the final answer is D.", ARC_LABELS[:3], ARC_CHOICES[:3]
    )
    assert (result.status, result.method, result.value) == (OK, "answer-pattern", "B")


def test_extract_arc_tail_letter_falls_through_to_leading_label():
    """末位级取到正文末字母时下沉，由前置标签级接住。

    形态取自 R1-1 实测，模型以标签点号接选项正文作答，正文末尾的 F 与 s 会被
    末位级读成标签。
    """
    result = extract_arc("A. 32° F  \nB. 41° F  \nC. 78° F  \nD. 98° F", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "leading-label", "A")


def test_extract_arc_tail_letter_falls_through_to_option_text():
    """末位级取到集合外字母时下沉，由正文比对接住。"""
    labels = ["A", "B", "C", "D"]
    choices = ["6.45 J", "Planetary days will become shorter.", "3.2 J", "0.5 J"]
    result = extract_arc("B. Planetary days will become shorter. J", labels, choices)
    assert (result.status, result.method, result.value) == (OK, "option-text", "B")


def test_extract_arc_leading_label_numeric_body():
    """输出以数字标签开头时，正文比对剥去标签后仍能定位。"""
    result = extract_arc("2. Venus", NUMERIC_LABELS, NUMERIC_CHOICES)
    assert (result.status, result.method, result.value) == (OK, "option-text", "2")


def test_extract_arc_label_prefix_without_space_is_not_a_label():
    """标号之后必须紧跟空白，否则 3.5 m/s 会被读成标签 3。"""
    result = extract_arc("3.5 m/s", NUMERIC_LABELS, NUMERIC_CHOICES)
    assert result.value != "3"
    assert (result.method, result.status) == ("bare-label", WARN)


def test_extract_arc_five_options_accepts_e():
    """五选项题的标签集合含 E，不按 A 至 D 截断。"""
    result = extract_arc("Answer: E", ARC_LABELS + ["E"], ARC_CHOICES + ["The planet will vanish."])
    assert (result.status, result.method, result.value) == (OK, "answer-pattern", "E")


def test_extract_arc_no_hit_fails():
    """三级均无命中时判失败，不猜。"""
    result = extract_arc("I am not sure.", ARC_LABELS, ARC_CHOICES)
    assert (result.status, result.method, result.value) == (FAIL, "failed", None)


def test_extract_arc_short_option_skipped():
    """过短的选项不参与正文比对，以免匹配到任意文本。"""
    result = extract_arc("yes", ARC_LABELS, ["yes", "no", "maybe", "perhaps"])
    assert result.status == FAIL


# ----------------- 等值判定 ------------------


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("72", "72.0"),
        ("$47", "47"),
        ("1,234", "1234"),
        ("50%", "50"),
        ("0.75", "3/4"),
        ("1 1/2", "1.5"),
    ],
)
def test_is_correct_equivalence(pred, gold):
    """字形不同但等值的两侧判为正确。"""
    assert is_correct(pred, gold, "gsm8k")


@pytest.mark.parametrize("pred, gold", [("72", "1,234"), ("abc", "72"), ("", "0")])
def test_is_correct_rejects(pred, gold):
    """不等值或无法规整的一侧判为错误。"""
    assert not is_correct(pred, gold, "gsm8k")


def test_is_correct_none_is_false():
    """抽取失败计为错误，与准确率以全样本为分母一致。"""
    assert not is_correct(None, "72", "gsm8k")
    assert not is_correct("72", None, "gsm8k")


def test_is_correct_arc_is_case_insensitive():
    """ARC 只做去空白与大写化，标签不参与数值规整。"""
    assert is_correct("b", "B", "arc")
    assert is_correct(" 2 ", "2", "arc")
    assert not is_correct("B", "C", "arc")


# ----------------- 模块约束 ------------------


def test_extract_rules_version():
    """规则版本号须与 RECORD/01-实验配置.md 所记一致，改动规则时同步递增。"""
    assert EXTRACT_RULES_VERSION == "v2"


def test_import_does_not_load_torch():
    """抽取模块不得在导入期拖入 torch，否则单测无法在无 torch 的机器上跑。

    另起进程验证，避免同一次会话中其他测试先导入 torch 而令本项失效。
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import sys, src.extract; print('torch' in sys.modules)"],
        cwd=EXP_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


# ----------------- 真实输出冻结集 ------------------


def test_frozen_real_outputs():
    """首次 DLC 运行后冻结的真实输出，用作规则回归的固定用例。

    文件每行一条，字段为 dataset、text、status、method、value，ARC 的条目另需
    labels 与 choices 两项。status、method、value 为冻结当时的抽取结果，须人工
    核对一遍再入库，此后任一条抽取结果变化都会在此暴露。文件不存在时跳过，不
    影响本机运行。
    """
    if not FIXTURE_PATH.is_file():
        pytest.skip("真实输出尚未冻结，待首次 DLC 运行后生成")

    records = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records, "冻结文件存在但内容为空"

    for record in records:
        if record["dataset"] == "gsm8k":
            result = extract_gsm8k(record["text"])
        else:
            result = extract_arc(record["text"], record["labels"], record["choices"])
        assert result.status == record["status"], record["text"]
        assert result.method == record["method"], record["text"]
        assert result.value == record["value"], record["text"]
