"""pytest 全局配置。

pytest 默认只把测试模块所在目录 exp1/tests 放进 sys.path，不会放进它的上一
级目录 exp1，而 src 包位于 exp1/src。因此测试中写 from src import utils 会
报 ModuleNotFoundError。本文件把 exp1 补进 sys.path 解决该问题。

放在 tests 目录下而非 src 目录下，是因为这个动作只对测试有意义，业务代码不
应依赖它。src 内部的相互引用统一写成 from src import ... 的绝对导入，该写法
成立的前提是导入 src 时 exp1 已在 sys.path 上。评测入口以 python -m src.eval
从 exp1 目录启动，当前目录即 exp1，天然满足。

用法
    pytest exp1/tests
    pytest exp1/tests/test_data.py -k strip
"""

from __future__ import annotations

import sys
from pathlib import Path

# 本文件位于 exp1/tests/conftest.py，上溯两级即 exp1
EXP_ROOT = Path(__file__).resolve().parent.parent

if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
