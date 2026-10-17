# conftest.py
"""pytest 根配置：把仓库根目录加入 sys.path，使测试可以直接 import 各包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
