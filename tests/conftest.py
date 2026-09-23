"""pytest 引导：确保仓库根目录在 sys.path 上。

使 `pytest` 在任意工作目录下都能导入未打包安装的 wind_farm_opt 包。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
